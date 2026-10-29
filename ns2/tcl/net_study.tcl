# Network study scenario: Evaluating collection network capacity (O10)
# N receivers transmit UDP reports to Gateway 0 at rate R Hz
# D extra devices generate background traffic

set val(chan) Channel/WirelessChannel
set val(netif) Phy/WirelessPhy
set val(mac)  Mac/802_11
set val(ifq)  Queue/DropTail/PriQueue
set val(ll)   LL
set val(ant)  Antenna/OmniAntenna
set val(rp)   NOAH

# Parameters passed via command-line arguments:
# ns net_study.tcl <num_rx> <rate_hz> <num_devices> <duration_s> <run_id>
set num_rx      [lindex $argv 0]
set rate_hz     [lindex $argv 1]
set num_devices [lindex $argv 2]
set duration_s  [lindex $argv 3]
set run_id      [lindex $argv 4]

if {$num_rx == ""} { set num_rx 2 }
if {$rate_hz == ""} { set rate_hz 10 }
if {$num_devices == ""} { set num_devices 0 }
if {$duration_s == ""} { set duration_s 60 }
if {$run_id == ""} { set run_id "net_study_default" }

set total_nodes [expr 1 + $num_rx + $num_devices]

Phy/WirelessPhy set freq_ 2.437e9
Phy/WirelessPhy set Pt_ 0.1
Phy/WirelessPhy set RXThresh_ 1.0e-12
Phy/WirelessPhy set CSThresh_ 7.9e-13
Mac/802_11 set RTSThreshold_ 3000
Mac/802_11 set dataRate_ 1Mb
Mac/802_11 set basicRate_ 1Mb

set ns [new Simulator]
set tf [open ns2/out/$run_id.tr w]
$ns trace-all $tf

set topo [new Topography]
$topo load_flatgrid 10.0 10.0

create-god $total_nodes

$ns node-config -adhocRouting $val(rp) \
  -llType $val(ll) -macType $val(mac) \
  -ifqType $val(ifq) -ifqLen 50 \
  -antType $val(ant) -propType Propagation/FreeSpace \
  -phyType $val(netif) -channelType $val(chan) \
  -topoInstance $topo -agentTrace ON -routerTrace OFF -macTrace ON

# Node 0 is the Gateway
set gw [$ns node]
$gw set X_ 5.0; $gw set Y_ 5.0; $gw set Z_ 0.0

set sink [new Agent/Null]
$ns attach-agent $gw $sink

# Receiver sniffers sending reports to Gateway
set interval [expr 1.0 / $rate_hz]
for {set i 1} {$i <= $num_rx} {incr i} {
  set rx_node($i) [$ns node]
  set angle [expr ($i * 6.28318) / $num_rx]
  $rx_node($i) set X_ [expr 5.0 + 3.0 * cos($angle)]
  $rx_node($i) set Y_ [expr 5.0 + 3.0 * sin($angle)]
  $rx_node($i) set Z_ 0.0

  set udp($i) [new Agent/UDP]
  $ns attach-agent $rx_node($i) $udp($i)
  $ns connect $udp($i) $sink

  set cbr($i) [new Application/Traffic/CBR]
  $cbr($i) attach-agent $udp($i)
  $cbr($i) set packetSize_ 40
  $cbr($i) set interval_ $interval
  $cbr($i) set random_ 0.005

  set t_start [expr 1.0 + ($i * 0.002)]
  $ns at $t_start "$cbr($i) start"
}

# Extra background traffic devices
for {set j 1} {$j <= $num_devices} {incr j} {
  set dev_idx [expr 1 + $num_rx + $j - 1]
  set dev_node($j) [$ns node]
  $dev_node($j) set X_ [expr 5.0 + 4.0 * sin($j)]
  $dev_node($j) set Y_ [expr 5.0 + 4.0 * cos($j)]
  $dev_node($j) set Z_ 0.0

  set dev_udp($j) [new Agent/UDP]
  $ns attach-agent $dev_node($j) $dev_udp($j)
  $ns connect $dev_udp($j) $sink

  set dev_cbr($j) [new Application/Traffic/CBR]
  $dev_cbr($j) attach-agent $dev_udp($j)
  $dev_cbr($j) set packetSize_ 512
  $dev_cbr($j) set interval_ 0.02
  $ns at 1.5 "$dev_cbr($j) start"
}

proc finish {} {
  global ns tf
  $ns flush-trace
  close $tf
  exit 0
}

$ns at $duration_s "finish"
$ns run
