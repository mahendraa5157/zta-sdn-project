#!/usr/bin/python3

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet, ipv4, tcp

import time
import requests
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================

SEQ_LENGTH = 20
INFERENCE_URL = "http://127.0.0.1:8000/predict"
TRUST_THRESHOLD = 0.5

# =========================
# CONTROLLER
# =========================

class ZTAEnforcement(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ZTAEnforcement, self).__init__(*args, **kwargs)

        self.mac_to_port = {}
        self.host_buffers = defaultdict(lambda: deque(maxlen=SEQ_LENGTH))

        self.packet_stats = defaultdict(lambda: {
            "count": 0,
            "bytes": 0,
            "syn": 0,
            "ack": 0,
            "start_time": None,
        })

        self.logger.info("ZTA Enforcement Controller Started")

    # =========================
    # Install CONTROLLER Rule
    # =========================
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

        self.logger.info("Installed CONTROLLER forwarding rule")

    # =========================
    # Packet Processing
    # =========================
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

        # ----------------------
        # Learning Switch Logic
        # ----------------------
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        src_mac = eth.src
        dst_mac = eth.dst
        in_port = msg.match['in_port']

        self.mac_to_port[dpid][src_mac] = in_port

        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        # ----------------------
        # IPv4 Processing
        # ----------------------
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt is None:
            actions = [parser.OFPActionOutput(out_port)]
            self._send_packet(datapath, msg, in_port, actions)
            return

        src_ip = ip_pkt.src

        stats = self.packet_stats[src_ip]
        now = time.time()

        if stats["start_time"] is None:
            stats["start_time"] = now

        stats["count"] += 1
        stats["bytes"] += len(msg.data)

        tcp_pkt = pkt.get_protocol(tcp.tcp)
        if tcp_pkt:
            if tcp_pkt.bits & tcp.TCP_SYN:
                stats["syn"] += 1
            if tcp_pkt.bits & tcp.TCP_ACK:
                stats["ack"] += 1

        duration = max(now - stats["start_time"], 1e-6)

        feature_vector = [
            stats["count"],
            stats["bytes"],
            duration,
            stats["bytes"] / stats["count"],
            stats["count"] / duration,
            stats["syn"] / stats["count"],
            stats["ack"] / stats["count"],
            stats["syn"] - stats["ack"]
        ]

        self.host_buffers[src_ip].append(feature_vector)

        # ----------------------
        # Run ML when window full
        # ----------------------
        if len(self.host_buffers[src_ip]) == SEQ_LENGTH:

            try:
                response = requests.post(
                    INFERENCE_URL,
                    json={
                        "host_id": src_ip,
                        "features": list(self.host_buffers[src_ip])
                    },
                    timeout=3
                )

                result = response.json()

                if "anomaly_probability" not in result:
                    self.logger.error("Invalid inference response: %s", result)
                else:
                    anomaly = result["anomaly_probability"]
                    trust = result["trust_score"]

                    self.logger.info(
                        "Host %s | Anomaly=%.3f | Trust=%.3f",
                        src_ip, anomaly, trust
                    )

                    if trust < TRUST_THRESHOLD:
                        self.logger.warning("BLOCKING HOST %s", src_ip)
                        return  # Drop packet

            except Exception as e:
                self.logger.error("Inference error: %s", str(e))

            # Reset window
            self.packet_stats[src_ip] = {
                "count": 0,
                "bytes": 0,
                "syn": 0,
                "ack": 0,
                "start_time": None
            }

            self.host_buffers[src_ip].clear()

        # Forward packet
        actions = [parser.OFPActionOutput(out_port)]
        self._send_packet(datapath, msg, in_port, actions)

    # =========================
    # Packet Out Helper
    # =========================
    def _send_packet(self, datapath, msg, in_port, actions):

        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )

        datapath.send_msg(out)

