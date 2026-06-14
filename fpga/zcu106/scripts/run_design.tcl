# Run one registered activation implementation on the ZCU106 device.
#
# Required Tcl arguments:
#   activation  sigmoid|tanh
#   method      baseline|proposed
#   format      int8|uint8|fp8|fp32
#   action      synth|impl
#   period_ns   positive clock period
#   jobs        positive Vivado thread count

proc fail {message} {
    puts stderr "ERROR: $message"
    exit 2
}

if {$argc != 6} {
    fail "Usage: run_design.tcl <activation> <method> <format> <synth|impl> <period_ns> <jobs>"
}

lassign $argv activation method data_format action period_ns jobs

if {$activation ni {sigmoid tanh}} {
    fail "Unsupported activation '$activation'"
}
if {$method ni {baseline proposed}} {
    fail "Unsupported method '$method'"
}
if {$data_format ni {int8 uint8 fp8 fp32}} {
    fail "Unsupported format '$data_format'"
}
if {$action ni {synth impl}} {
    fail "Unsupported action '$action'"
}
if {![string is double -strict $period_ns] || $period_ns <= 0.0} {
    fail "Clock period must be positive"
}
if {![string is integer -strict $jobs] || $jobs < 1} {
    fail "Jobs must be a positive integer"
}

set script_dir [file dirname [file normalize [info script]]]
set flow_dir [file dirname $script_dir]
set repo_root [file dirname [file dirname $flow_dir]]
set rtl_root [file join $repo_root rtl]
set build_root [file join $flow_dir build]
set design_name "${activation}_${method}_${data_format}"
set run_dir [file join $build_root $design_name]
set reports_dir [file join $run_dir reports]
set checkpoints_dir [file join $run_dir checkpoints]

file mkdir $reports_dir
file mkdir $checkpoints_dir

set activation_dir $activation
set format_dir [string toupper $data_format]
if {$method eq "proposed"} {
    set rtl_dir [file join $rtl_root $activation_dir proposed1 $format_dir]
} else {
    set rtl_dir [file join $rtl_root $activation_dir baseline $format_dir]
}

set module_name "${activation}_top_${data_format}"
set rtl_files [glob -nocomplain -directory $rtl_dir *.v]
set wrapper_file [file join $flow_dir rtl fpga_benchmark_wrapper.sv]
set xdc_template [file join $flow_dir constraints zcu106_1ghz_benchmark.xdc]
set generated_xdc [file join $run_dir benchmark.xdc]

if {[llength $rtl_files] == 0} {
    fail "RTL files not found in: $rtl_dir"
}
if {![file exists $wrapper_file]} {
    fail "Wrapper file not found: $wrapper_file"
}

set xdc_in [open $xdc_template r]
set xdc_text [read $xdc_in]
close $xdc_in
regsub -all {period 1\.000} $xdc_text "period $period_ns" xdc_text
set half_period [expr {$period_ns / 2.0}]
regsub -all {\{0\.000 0\.500\}} $xdc_text [format "{0.000 %.3f}" $half_period] xdc_text
set xdc_out [open $generated_xdc w]
puts -nonewline $xdc_out $xdc_text
close $xdc_out

set data_width [expr {$data_format eq "fp32" ? 32 : 8}]
set part_name xczu7ev-ffvc1156-2-e
set_param general.maxThreads $jobs

create_project -in_memory -part $part_name
set_property target_language Verilog [current_project]
set_property verilog_define [list "DUT_MODULE=$module_name" "DATA_WIDTH=$data_width"] [current_fileset]

read_verilog $rtl_files
read_verilog -sv $wrapper_file
read_xdc $generated_xdc

synth_design \
    -top fpga_benchmark_wrapper \
    -part $part_name \
    -flatten_hierarchy rebuilt \
    -directive PerformanceOptimized

write_checkpoint -force [file join $checkpoints_dir post_synth.dcp]
report_utilization -file [file join $reports_dir post_synth_utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $reports_dir post_synth_timing_summary.rpt]

if {$action eq "impl"} {
    opt_design -directive ExploreWithRemap
    place_design -directive ExtraNetDelay_high
    phys_opt_design -directive AggressiveExplore
    route_design -directive AggressiveExplore
    phys_opt_design -directive AggressiveExplore

    write_checkpoint -force [file join $checkpoints_dir post_route.dcp]
    report_timing_summary -delay_type min_max -max_paths 50 \
        -file [file join $reports_dir post_route_timing_summary.rpt]
    report_timing -delay_type max -max_paths 50 -sort_by group \
        -file [file join $reports_dir post_route_setup_paths.rpt]
    report_timing -delay_type min -max_paths 50 -sort_by group \
        -file [file join $reports_dir post_route_hold_paths.rpt]
    report_utilization -hierarchical \
        -file [file join $reports_dir post_route_utilization.rpt]
    report_power -file [file join $reports_dir post_route_power.rpt]
    report_methodology -file [file join $reports_dir methodology.rpt]
    report_clock_utilization -file [file join $reports_dir clock_utilization.rpt]
}

set input_registers [get_cells -quiet -hierarchical -regexp {.*b_reg_reg.*}]
set output_registers [get_cells -quiet -hierarchical -regexp {.*z_reg.*}]
set setup_paths [get_timing_paths -quiet -delay_type max -max_paths 1 \
    -from $input_registers -to $output_registers]
set hold_paths [get_timing_paths -quiet -delay_type min -max_paths 1 \
    -from $input_registers -to $output_registers]
set wns [expr {[llength $setup_paths] ? [get_property SLACK [lindex $setup_paths 0]] : "NA"}]
set whs [expr {[llength $hold_paths] ? [get_property SLACK [lindex $hold_paths 0]] : "NA"}]
set lut_count [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
set ff_count [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
set dsp_count [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]]
set bram_count [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB*}]]

set slices 0
if {[file exists [file join $reports_dir post_route_utilization.rpt]]} {
    set fp [open [file join $reports_dir post_route_utilization.rpt] r]
    set util_data [read $fp]
    close $fp
    regexp {CLB LUTs\s*\|\s*(\d+)} $util_data -> slices
}

set total_power_w 0.0
set dynamic_power_w 0.0
set static_power_w 0.0
if {[file exists [file join $reports_dir post_route_power.rpt]]} {
    set fp [open [file join $reports_dir post_route_power.rpt] r]
    set power_data [read $fp]
    close $fp
    regexp {Total On-Chip Power \(W\)\s*\|\s*([\d\.]+)} $power_data -> total_power_w
    regexp {Dynamic \(W\)\s*\|\s*([\d\.]+)} $power_data -> dynamic_power_w
    regexp {Device Static \(W\)\s*\|\s*([\d\.]+)} $power_data -> static_power_w
}

set timing_met 0
if {$wns ne "NA" && $whs ne "NA" && $wns >= 0.0 && $whs >= 0.0} {
    set timing_met 1
}

set metrics_file [open [file join $run_dir metrics.csv] w]
puts $metrics_file "activation,method,data_format,action,part,period_ns,target_mhz,wns_ns,whs_ns,timing_met,luts,flip_flops,dsps,brams,slices,total_power_w,dynamic_power_w,static_power_w"
puts $metrics_file [join [list \
    $activation $method $data_format $action $part_name $period_ns \
    [format %.3f [expr {1000.0 / $period_ns}]] $wns $whs $timing_met \
    $lut_count $ff_count $dsp_count $bram_count $slices $total_power_w $dynamic_power_w $static_power_w] ","]
close $metrics_file

puts "RESULT design=$design_name period_ns=$period_ns WNS=$wns WHS=$whs timing_met=$timing_met"
exit 0
