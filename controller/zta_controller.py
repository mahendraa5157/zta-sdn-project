import csv
import datetime
import time
import threading

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (
    MAIN_DISPATCHER,
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    set_ev_cls
)
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ether_types


class ZTAController(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ZTAController, self).__init__(*args, **kwargs)

        print("ZTA Controller Started")

        self.mac_to_port = {}
        self.datapaths = {}
        self.monitor_interval = 5

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Create CSV header
        with open("flow_stats_log.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "timestamp",
                "dpid",
                "src_ip",
                "dst_ip",
                "packet_count",
                "byte_count",
                "duration"
            ])

    # -------------------------------------
    # Datapath Registration
    # -------------------------------------
    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    # -------------------------------------
    # Install Table-Miss Flow
    # -------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER,
            ofproto.OFPCML_NO_BUFFER
        )]

        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS,
            actions
        )]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst
        )

        datapath.send_msg(mod)

    # -------------------------------------
    # L3-Aware Learning Switch
    # -------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        dst = eth.dst
        src = eth.src

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install IPv4 flow rule
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if ip_pkt and out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst
            )
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )

        datapath.send_msg(out)

    # -------------------------------------
    # Flow Monitoring
    # -------------------------------------
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            time.sleep(self.monitor_interval)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        body = ev.msg.body

        with open("flow_stats_log.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for stat in body:

                # Skip table-miss rule
                if stat.priority == 0:
                    continue

                match = stat.match

                src_ip = match.get('ipv4_src', 'N/A')
                dst_ip = match.get('ipv4_dst', 'N/A')

                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    ev.msg.datapath.id,
                    src_ip,
                    dst_ip,
                    stat.packet_count,
                    stat.byte_count,
                    stat.duration_sec
                ])

