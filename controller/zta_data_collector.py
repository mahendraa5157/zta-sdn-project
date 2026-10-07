#!/usr/bin/python3

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet, ipv4, tcp

import time
import csv
from collections import defaultdict, deque

# ===========================================================
# CONFIGURE LABEL HERE
# 0 = NORMAL TRAFFIC
# 1 = ATTACK TRAFFIC
# ===========================================================
DATA_LABEL = 0     # <<< CHANGE THIS BEFORE COLLECTING
SEQ_LENGTH = 20    # window size for dataset
CSV_FILE = "traffic_dataset.csv"


class ZTADataCollector(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ZTADataCollector, self).__init__(*args, **kwargs)

        self.mac_to_port = {}
        self.host_buffers = defaultdict(lambda: deque(maxlen=SEQ_LENGTH))

        self.packet_stats = defaultdict(lambda: {
            "count": 0,
            "bytes": 0,
            "syn": 0,
            "ack": 0,
            "start_time": None,
        })

        self.dataset_file = open(CSV_FILE, "a", newline="")
        self.csv_writer = csv.writer(self.dataset_file)

        self.logger.info("===== ZTA DATA COLLECTION CONTROLLER STARTED =====")
        self.logger.info(f"DATA LABEL SET TO: {DATA_LABEL}")

    # --------------------------------------------------------
    # Install rule: send ALL PACKETS to CONTROLLER
    # --------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):

        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch()

        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        inst = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=0,
            match=match,
            instructions=inst
        )

        datapath.send_msg(mod)

        self.logger.info("Flow installed: ALL packets → controller")

    # --------------------------------------------------------
    # PACKET-IN HANDLER
    # --------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):

        msg = ev.msg
        datapath = msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None or eth.ethertype == 0x88cc:
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt is None:
            return

        src_ip = ip_pkt.src
        in_port = msg.match['in_port']

        # Learn MAC → Port
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][eth.src] = in_port

        # Basic forwarding
        if eth.dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][eth.dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        # ---- STATS UPDATE ----
        stats = self.packet_stats[src_ip]
        now = time.time()

        if stats["start_time"] is None:
            stats["start_time"] = now

        stats["count"] += 1
        stats["bytes"] += len(msg.data)

        # TCP flags
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        if tcp_pkt:
            if tcp_pkt.bits & tcp.TCP_SYN:
                stats["syn"] += 1
            if tcp_pkt.bits & tcp.TCP_ACK:
                stats["ack"] += 1

        # Feature extraction
        duration = max(now - stats["start_time"], 1e-6)
        avg_pkt_size = stats["bytes"] / stats["count"]
        pkt_rate = stats["count"] / duration

        syn_ratio = stats["syn"] / stats["count"]
        ack_ratio = stats["ack"] / stats["count"]
        syn_ack_diff = stats["syn"] - stats["ack"]

        feature_vec = [
            stats["count"],
            stats["bytes"],
            duration,
            avg_pkt_size,
            pkt_rate,
            syn_ratio,
            ack_ratio,
            syn_ack_diff
        ]

        self.host_buffers[src_ip].append(feature_vec)

        # ---- SAVE WINDOW WHEN FULL ----
        if len(self.host_buffers[src_ip]) == SEQ_LENGTH:

            self.csv_writer.writerow([
                stats["count"],
                stats["bytes"],
                duration,
                avg_pkt_size,
                pkt_rate,
                syn_ratio,
                ack_ratio,
                syn_ack_diff,
                DATA_LABEL     # <<< LABEL WRITTEN TO DATASET
            ])

            self.dataset_file.flush()

            self.logger.info(
                f"[SAVED] Host {src_ip} | Rate={pkt_rate:.2f} pps | Label={DATA_LABEL}"
            )

            # Reset window
            self.packet_stats[src_ip] = {
                "count": 0,
                "bytes": 0,
                "syn": 0,
                "ack": 0,
                "start_time": None
            }
            self.host_buffers[src_ip].clear()

        # Continue forwarding traffic
        actions = [parser.OFPActionOutput(out_port)]

        out = parser.OFPPacketOut(
            datapath=datapath,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )

        datapath.send_msg(out)
