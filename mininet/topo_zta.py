#!/usr/bin/python3

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

class ZTATopo(Topo):
    def build(self):

        # Switches (explicit DPIDs required)
        s1 = self.addSwitch('s1', dpid='0000000000000001')
        s2 = self.addSwitch('s2', dpid='0000000000000002')
        s3 = self.addSwitch('s3', dpid='0000000000000003')

        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        server = self.addHost('sv', ip='10.0.0.10/24')
        attacker = self.addHost('atk', ip='10.0.0.99/24')

        # Links
        self.addLink(h1, s1)
        self.addLink(h2, s1)

        self.addLink(attacker, s2)

        self.addLink(server, s3)

        self.addLink(s1, s3)
        self.addLink(s2, s3)


if __name__ == '__main__':
    setLogLevel('info')

    topo = ZTATopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        autoSetMacs=True,
        autoStaticArp=True
    )

    # Remote Os-ken controller
    net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6653
    )

    net.start()
    print("\n*** ZTA Network Started ***\n")
    CLI(net)
    net.stop()
