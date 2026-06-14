#!/usr/bin/env python3
import subprocess
import sys
import os

# ============================================================
# MASTER TEST RUNNER
# ============================================================
# Runs all sigmoid and tanh approximation implementations.
# Each implementation evaluates FP32, INT8, UINT8, and FP8 internally.
# Both baseline (equal-width) and proposed (curvature-weighted) are covered.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

test_files = [
    ("SIGMOID BASELINE", "sigmoid_baseline.py"),
    ("SIGMOID PROPOSED", "sigmoid_proposed.py"),
    ("TANH BASELINE", "tanh_baseline.py"),
    ("TANH PROPOSED", "tanh_proposed.py"),
]

print("\n" + "="*70)
print("MASTER TEST RUNNER - SIGMOID & TANH ALL DATA FORMATS")
print("="*70)

all_passed = True
results = []

for title, script in test_files:
    script_path = os.path.join(SCRIPT_DIR, script)
    
    if not os.path.exists(script_path):
        print(f"\n[FAIL] {title}: File not found - {script_path}")
        results.append((title, "FAIL", "File not found"))
        all_passed = False
        continue
    
    print(f"\n[RUN] {title}...")
    print("-"*70)
    
    try:
        result = subprocess.run(
            ["python3", script_path],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(result.stdout)
            results.append((title, "PASS", "Completed successfully"))
            print(f"[PASS] {title}")
        else:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            results.append((title, "FAIL", result.stderr[:100]))
            all_passed = False
            print(f"[FAIL] {title}")
    
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {title}: Timeout (30s)")
        results.append((title, "FAIL", "Timeout"))
        all_passed = False
    
    except Exception as e:
        print(f"[FAIL] {title}: {str(e)}")
        results.append((title, "FAIL", str(e)[:100]))
        all_passed = False

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"\n{'Test':30s}  {'Status':10s}")
print("-"*42)
for title, status, msg in results:
    print(f"{title:30s}  {status:10s}")

print("\n" + "="*70)
if all_passed:
    print("OVERALL STATUS: ALL TESTS PASSED")
    print("="*70)
    sys.exit(0)
else:
    print("OVERALL STATUS: SOME TESTS FAILED")
    print("="*70)
    sys.exit(1)
