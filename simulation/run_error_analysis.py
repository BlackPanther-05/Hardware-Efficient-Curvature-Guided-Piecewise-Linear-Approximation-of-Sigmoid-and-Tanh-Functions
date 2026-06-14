import os
import subprocess
import re
import csv

def main():
    activations = ["tanh", "sigmoid"]
    formats = ["FP32", "FP8", "INT8", "UINT8"]
    architectures = ["baseline", "proposed1", "proposed2", "cordic_paper1", "cordic_paper2"]
    
    results = []

    for act in activations:
        for fmt in formats:
            for arch in architectures:
                # Determine the testbench file. We always use the baseline testbench.
                tb_file = f"rtl/{act}/{fmt}/tb_{act}_{fmt.lower()}_error.v"
                
                compile_files = []
                
                if arch == "baseline":
                    compile_files.append(f"rtl/{act}/{fmt}/{act}_top_{fmt.lower()}.v")
                elif arch == "proposed1":
                    compile_files.append(f"rtl/{act}/proposed1/{fmt}/{act}_top_{fmt.lower()}.v")
                    lut_file = f"rtl/{act}/proposed1/{fmt}/lut_{act}_{fmt.lower()}.v"
                    if os.path.exists(lut_file):
                        compile_files.append(lut_file)
                elif arch == "proposed2":
                    compile_files.append(f"rtl/{act}/proposed2/{fmt}/{act}_top_{fmt.lower()}.v")
                elif arch == "cordic_paper1":
                    # Use cordic specific testbench for cordic to avoid module name conflicts
                    tb_file = f"rtl/cordic/paper1/{fmt}/tb_cordic_p1_{act}_{fmt.lower()}.v"
                    compile_files.append("rtl/cordic/paper1/cordic_p1_core.v")
                    if fmt == "FP32":
                        compile_files.append(f"rtl/cordic/paper1/cordic_p1_{act}_top.v")
                        # The testbench for fp32 cordic was named tb_cordic_p1_tanh.v without fp32
                        tb_file = f"rtl/cordic/paper1/tb_cordic_p1_{act}.v"
                    else:
                        compile_files.append(f"rtl/cordic/paper1/{fmt}/cordic_p1_{act}_top_{fmt.lower()}.v")
                elif arch == "cordic_paper2":
                    tb_file = f"rtl/cordic/paper2/{fmt}/tb_cordic_p2_{act}_{fmt.lower()}.v"
                    compile_files.append("rtl/cordic/paper2/cordic_p2_core.v")
                    if fmt == "FP32":
                        compile_files.append(f"rtl/cordic/paper2/cordic_p2_{act}_top.v")
                        tb_file = f"rtl/cordic/paper2/tb_cordic_p2_{act}.v"
                    else:
                        compile_files.append(f"rtl/cordic/paper2/{fmt}/cordic_p2_{act}_top_{fmt.lower()}.v")

                # Check if all files exist
                valid = os.path.exists(tb_file)
                for f in compile_files:
                    if not os.path.exists(f):
                        valid = False
                        break
                
                if not valid:
                    continue

                cmd = ["iverilog", "-o", "sim.vvp"] + compile_files + [tb_file]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    print(f"Compilation failed for {arch} {act} {fmt}")
                    print(proc.stderr.decode('utf-8'))
                    continue
                    
                proc = subprocess.run(["vvp", "sim.vvp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = proc.stdout.decode('utf-8')
                
                max_err = "N/A"
                avg_err = "N/A"
                
                for line in output.split('\n'):
                    if "Max error" in line:
                        m = re.search(r':\s*([0-9\.eE+-]+)', line)
                        if m: max_err = m.group(1)
                    if "Avg error" in line:
                        m = re.search(r':\s*([0-9\.eE+-]+)', line)
                        if m: avg_err = m.group(1)
                        
                results.append({
                    "Architecture": arch,
                    "Activation": act,
                    "Format": fmt,
                    "Average_Error": avg_err,
                    "Max_Error": max_err
                })

    with open("error_analysis_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Architecture", "Activation", "Format", "Average_Error", "Max_Error"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("Error analysis completed. Results saved to error_analysis_results.csv")

if __name__ == "__main__":
    main()
