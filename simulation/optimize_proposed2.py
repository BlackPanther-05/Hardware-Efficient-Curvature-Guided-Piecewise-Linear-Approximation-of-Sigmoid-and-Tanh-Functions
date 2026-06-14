import glob
import os
import re

rtl_dir = '../rtl'

sigmoid_files = glob.glob(os.path.join(rtl_dir, 'sigmoid/proposed2/*/*_top_*.v'))
tanh_files = glob.glob(os.path.join(rtl_dir, 'tanh/proposed2/*/*_top_*.v'))

# Optimize Sigmoid (14 segments -> valid indices 0 to 13)
for fpath in sigmoid_files:
    with open(fpath, 'r') as f:
        content = f.read()
    
    # Change default seg in casez
    content = content.replace("default: seg = 4'd15;", "default: seg = 4'd13;")
    
    # Remove unused lut cases
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if "4'd14: begin m_lut =" in line:
            continue
        if "4'd15: begin m_lut =" in line:
            continue
        new_lines.append(line)
        
    with open(fpath, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"Optimized {fpath}")

# Optimize Tanh (9 segments -> valid indices 0 to 8)
for fpath in tanh_files:
    with open(fpath, 'r') as f:
        content = f.read()
    
    # Change default seg in casez
    content = content.replace("default: seg = 4'd15;", "default: seg = 4'd8;")
    
    # Remove unused lut cases
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if any(f"4'd{i}: begin m_lut =" in line for i in range(9, 16)):
            continue
        new_lines.append(line)
        
    with open(fpath, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"Optimized {fpath}")

