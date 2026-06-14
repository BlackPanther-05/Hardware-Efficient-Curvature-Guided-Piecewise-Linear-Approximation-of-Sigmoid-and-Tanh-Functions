#!/bin/bash

# run_all.sh

OUTPUT_DIR="synthesis_results"
mkdir -p $OUTPUT_DIR
CSV_FILE="fpga_summary_results.csv"

echo "Architecture,Activation,Board,Fmax_MHz,Total_Power_W,Static_Power_W,Dynamic_Power_W,LUTs,FFs,DSPs" > $CSV_FILE

BOARDS=("xc7k160tfbg676-2" "xczu7ev-ffvc1156-2-e")
ACTIVATIONS=("tanh" "sigmoid")
FORMATS=("FP32" "FP8" "INT8" "UINT8")

for BOARD in "${BOARDS[@]}"; do
    BOARD_NAME="Kintex7"
    if [[ "$BOARD" == *"xczu"* ]]; then
        BOARD_NAME="ZCU106"
    fi

    for ACT in "${ACTIVATIONS[@]}"; do
        for FMT in "${FORMATS[@]}"; do
            FMT_LOW=$(echo "$FMT" | tr '[:upper:]' '[:lower:]')
            
            ACT_LOW="tanh"
            if [ "$ACT" == "sigmoid" ]; then ACT_LOW="sigmoid"; fi

            if [ "$FMT" == "FP32" ]; then
                ARCHS=(
                    "baseline_${FMT_LOW}|${ACT}_top_${FMT_LOW}|../rtl/${ACT_LOW}/baseline/${FMT}/${ACT}_top_${FMT_LOW}.v|NONE"
                    "proposed1_${FMT_LOW}|${ACT}_top_${FMT_LOW}|../rtl/${ACT_LOW}/baseline/proposed1/${FMT}/${ACT}_top_${FMT_LOW}.v|NONE"
                    "proposed2_${FMT_LOW}|${ACT}_top_${FMT_LOW}|../rtl/${ACT_LOW}/baseline/proposed2/${FMT}/${ACT}_top_${FMT_LOW}.v|NONE"
                    "cordic_paper1_${FMT_LOW}|cordic_p1_${ACT}_top|../rtl/cordic/paper1/cordic_p1_core.v,../rtl/cordic/paper1/cordic_p1_${ACT}_top.v|NONE"
                    "cordic_paper2_${FMT_LOW}|cordic_p2_${ACT}_top|../rtl/cordic/paper2/cordic_p2_core.v,../rtl/cordic/paper2/cordic_p2_${ACT}_top.v|NONE"
                )
            else
                ARCHS=(
                    "baseline_${FMT_LOW}|${ACT}_top_${FMT_LOW}|../rtl/${ACT_LOW}/baseline/${FMT}/${ACT}_top_${FMT_LOW}.v|NONE"
                    "proposed1_${FMT_LOW}|${ACT}_top_${FMT_LOW}|../rtl/${ACT_LOW}/baseline/proposed1/${FMT}/${ACT}_top_${FMT_LOW}.v|NONE"
                    "proposed2_${FMT_LOW}|${ACT}_top_${FMT_LOW}|../rtl/${ACT_LOW}/baseline/proposed2/${FMT}/${ACT}_top_${FMT_LOW}.v|NONE"
                    "cordic_paper1_${FMT_LOW}|cordic_p1_${ACT}_top_${FMT_LOW}|../rtl/cordic/paper1/cordic_p1_core.v,../rtl/cordic/paper1/${FMT}/cordic_p1_${ACT}_top_${FMT_LOW}.v|NONE"
                    "cordic_paper2_${FMT_LOW}|cordic_p2_${ACT}_top_${FMT_LOW}|../rtl/cordic/paper2/cordic_p2_core.v,../rtl/cordic/paper2/${FMT}/cordic_p2_${ACT}_top_${FMT_LOW}.v|NONE"
                )
            fi

            for ARCH_STR in "${ARCHS[@]}"; do
                IFS='|' read -r ARCH TOP SRC GEN <<< "$ARCH_STR"
                
                # Since Proposed 1 has multiple files, check if there's a separate LUT file in its directory.
                if [[ "$ARCH" == proposed1_* ]]; then
                    if [ -f "../rtl/${ACT_LOW}/baseline/proposed1/${FMT}/lut_${ACT}_${FMT_LOW}.v" ]; then
                        SRC="../rtl/${ACT_LOW}/baseline/proposed1/${FMT}/lut_${ACT}_${FMT_LOW}.v,$SRC"
                    fi
                fi

                echo "Running $ARCH for $ACT on $BOARD_NAME ($BOARD)..."
                
                RUN_DIR="${OUTPUT_DIR}/${BOARD_NAME}_${ACT}_${ARCH}"
                mkdir -p $RUN_DIR

                vivado -mode batch -source synth_eval.tcl -tclargs $BOARD $TOP $SRC $RUN_DIR $GEN > $RUN_DIR/vivado.log 2>&1
                
                # Parse results
                UTIL_FILE="$RUN_DIR/${TOP}_utilization.txt"
                POWER_FILE="$RUN_DIR/${TOP}_power.txt"
                SUMMARY_FILE="$RUN_DIR/summary.txt"

                LUTS="0"
                FFS="0"
                DSPS="0"
                TOTAL_POWER="0"
                STATIC_POWER="0"
                DYNAMIC_POWER="0"
                FMAX="0"

                if [ -f "$UTIL_FILE" ]; then
                    LUTS=$(grep -E "Slice LUTs" $UTIL_FILE | awk -F '|' '{print $3}' | head -n 1 | tr -d ' ' | tr -d ',')
                    if [ -z "$LUTS" ]; then
                         LUTS=$(grep -E "CLB LUTs" $UTIL_FILE | awk -F '|' '{print $3}' | head -n 1 | tr -d ' ' | tr -d ',')
                    fi
                    FFS=$(grep -E "Slice Registers" $UTIL_FILE | awk -F '|' '{print $3}' | head -n 1 | tr -d ' ' | tr -d ',')
                    if [ -z "$FFS" ]; then
                         FFS=$(grep -E "CLB Registers" $UTIL_FILE | awk -F '|' '{print $3}' | head -n 1 | tr -d ' ' | tr -d ',')
                    fi
                    DSPS=$(grep -E "^\| DSPs" $UTIL_FILE | awk -F '|' '{print $3}' | head -n 1 | tr -d ' ' | tr -d ',')
                    
                    [ -z "$LUTS" ] && LUTS="0"
                    [ -z "$FFS" ] && FFS="0"
                    [ -z "$DSPS" ] && DSPS="0"
                fi

                if [ -f "$POWER_FILE" ]; then
                    TOTAL_POWER=$(grep -E "Total On-Chip Power" $POWER_FILE | awk '{print $7}')
                    [ -z "$TOTAL_POWER" ] && TOTAL_POWER="0"
                    
                    STATIC_POWER=$(grep -E "Device Static" $POWER_FILE | awk -F '|' '{print $3}' | tr -d ' ' | head -n 1)
                    if [ -z "$STATIC_POWER" ]; then
                        STATIC_POWER=$(grep -E "Device Static \(W\)" $POWER_FILE | awk '{print $NF}' | head -n 1)
                    fi
                    [ -z "$STATIC_POWER" ] && STATIC_POWER="0"

                    DYNAMIC_POWER=$(grep -E "^\| Dynamic" $POWER_FILE | awk -F '|' '{print $3}' | tr -d ' ' | head -n 1)
                    if [ -z "$DYNAMIC_POWER" ]; then
                        DYNAMIC_POWER=$(grep -E "Dynamic \(W\)" $POWER_FILE | awk '{print $NF}' | head -n 1)
                    fi
                    [ -z "$DYNAMIC_POWER" ] && DYNAMIC_POWER="0"
                fi

                if [ -f "$SUMMARY_FILE" ]; then
                    FMAX=$(grep -E "FMAX:" $SUMMARY_FILE | awk '{print $2}')
                fi

                echo "$ARCH,$ACT,$BOARD_NAME,$FMAX,$TOTAL_POWER,$STATIC_POWER,$DYNAMIC_POWER,$LUTS,$FFS,$DSPS" >> $CSV_FILE
            done
        done
    done
done

echo "All runs completed. Check $CSV_FILE"
