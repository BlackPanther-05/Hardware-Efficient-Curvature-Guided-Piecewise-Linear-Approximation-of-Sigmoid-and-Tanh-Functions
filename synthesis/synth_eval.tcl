# synth_eval.tcl
# Arguments
set BOARD_PART [lindex $argv 0]
set TOP_MODULE [lindex $argv 1]
set SOURCE_FILES [lindex $argv 2]
set OUT_DIR [lindex $argv 3]
set GENERICS [lindex $argv 4]

# Ensure output directory exists
file mkdir $OUT_DIR

# Create project in memory
create_project -in_memory -part $BOARD_PART

# Read source files
set files_list [split $SOURCE_FILES ","]
foreach f $files_list {
    read_verilog $f
}

# Set generics if any
if {$GENERICS != "NONE"} {
    set_property generic $GENERICS [current_fileset]
}

# Run synthesis
synth_design -top $TOP_MODULE -part $BOARD_PART -mode out_of_context

# Constraints for timing (2 ns = 500 MHz)
create_clock -period 2.000 -name clk -waveform {0.000 1.000} [get_ports -quiet clk]

# Run opt, place, and route
opt_design
place_design
route_design

# Reports
report_utilization -file $OUT_DIR/${TOP_MODULE}_utilization.txt
report_timing_summary -file $OUT_DIR/${TOP_MODULE}_timing.txt
report_power -file $OUT_DIR/${TOP_MODULE}_power.txt

# Extract WNS and compute Fmax
set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
if {[llength $paths] > 0 && [get_property SLACK $paths] != ""} {
    set wns [get_property SLACK $paths]
    set fmax [expr 1000.0 / (2.000 - $wns)]
} else {
    set paths [get_timing_paths -max_paths 1 -nworst 1]
    if {[llength $paths] > 0 && [get_property DATAPATH_DELAY $paths] != ""} {
        set delay [get_property DATAPATH_DELAY $paths]
        if {$delay > 0} {
            set fmax [expr 1000.0 / $delay]
        } else {
            set fmax 0
        }
    } else {
        set fmax 0
    }
    set wns 0
}

# Extract Total On-Chip Power
set power [get_property POWER [get_board_parts -quiet]]
# Actually power is in report. We can extract from report or just use the report file.

set f [open "$OUT_DIR/summary.txt" w]
puts $f "WNS: $wns"
puts $f "FMAX: $fmax"
close $f

exit
