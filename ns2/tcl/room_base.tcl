# Base room template script for NS2 802.11 Smart Room Simulation
# Macros starting with @... will be substituted by gen_scenarios.py

set val(chan) Channel/WirelessChannel
set val(netif) Phy/WirelessPhy
set val(mac)  Mac/802_11
set val(ifq)  Queue/DropTail/PriQueue
set val(ll)   LL
set val(ant)  Antenna/OmniAntenna
set val(rp)   NOAH            ;# NOAH prevents routing protocol chatter

# ---- Radio parameters ----
Phy/WirelessPhy set freq_ 2.437e9
Phy/WirelessPhy set Pt_ @PT@
Phy/WirelessPhy set RXThresh_ 1.0e-12     ;# -90 dBm practical receiver sensitivity
Phy/WirelessPhy set CSThresh_ 7.9e-13     ;# -91 dBm carrier sense
Phy/WirelessPhy set L_ 1.0
Mac/802_11 set RTSThreshold_ 3000         ;# Disable RTS/CTS for short frames
Mac/802_11 set dataRate_ 1Mb
Mac/802_11 set basicRate_ 1Mb

# ---- Channel and Human Body model parameters ----
Propagation/BodyShadow set pathlossExp_ @BETA@
Propagation/BodyShadow set std_db_ @STD@
Propagation/BodyShadow set dist0_ 1.0
Propagation/BodyShadow set seed_ @SEED@
Propagation/BodyShadow set loss_db_ @LOSS@
Propagation/BodyShadow set extra_std_db_ @XSTD@
Propagation/BodyShadow set fresnel_scale_ @FRES@

set ns [new Simulator]
set tf [open ns2/out/@RUN@.tr w]
$ns trace-all $tf
set nf [open ns2/out/@RUN@.nam w]
$ns namtrace-all-wireless $nf @X@ @Y@

set topo [new Topography]
$topo load_flatgrid @X@ @Y@

create-god @NN@
set prop [new Propagation/BodyShadow]

$ns node-config -adhocRouting $val(rp) \
  -llType $val(ll) \
  -macType $val(mac) \
  -ifqType $val(ifq) \
  -ifqLen 50 \
  -antType $val(ant) \
  -propInstance $prop \
  -phyType $val(netif) \
  -channelType $val(chan) \
  -topoInstance $topo \
  -agentTrace ON \
  -routerTrace OFF \
  -macTrace ON \
  -movementTrace OFF

# Create nodes: Node 0 = TX, Node 1 = R1, Node 2 = R2, Node 3.. = People
for {set i 0} {$i < @NN@} {incr i} {
  set n($i) [$ns node]
  $n($i) random-motion 0
}

# Fixed Room Positions
$n(0) set X_ 0.3; $n(0) set Y_ 0.3; $n(0) set Z_ 0.0
$n(1) set X_ 5.7; $n(1) set Y_ 0.4; $n(1) set Z_ 0.0
$n(2) set X_ 3.0; $n(2) set Y_ 4.2; $n(2) set Z_ 0.0

# PERSON_INIT
for {set i 3} {$i < @NN@} {incr i} {
  $prop person $n($i)
  $ns initial_node_pos $n($i) 0.4
}

# Attach RSSI loggers to receivers R1 and R2
[$n(1) set netif_(0)] rssi-log ns2/out/@RUN@.rssi
[$n(2) set netif_(0)] rssi-log ns2/out/@RUN@.rssi

# Beacon proxy CBR flows (offset by 5 ms to avoid MAC collisions)
foreach {rx t0} {1 0.100 2 0.105} {
  set u($rx) [new Agent/UDP]
  $ns attach-agent $n(0) $u($rx)
  set s($rx) [new Agent/Null]
  $ns attach-agent $n($rx) $s($rx)
  $ns connect $u($rx) $s($rx)

  set c($rx) [new Application/Traffic/CBR]
  $c($rx) attach-agent $u($rx)
  $c($rx) set packetSize_ 100
  $c($rx) set interval_ @INT@
  $c($rx) set random_ 0
  $ns at $t0 "$c($rx) start"
}

proc finish {} {
  global ns tf nf n
  [$n(1) set netif_(0)] rssi-close
  [$n(2) set netif_(0)] rssi-close
  $ns flush-trace
  close $tf
  close $nf
  exit 0
}

$ns at @STOP@ "finish"
# Movement commands and "$ns run" will be appended by generator
