import sys
import os
import csv

# We can import functions from run_activation_simulations
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_activation_simulations import build_segments

def main():
    rows = []
    
    # Baseline: 16 uniform segments
    for act in ["sigmoid", "tanh"]:
        segs = build_segments(act, "baseline", "float", 16, 8.0, 2048)
        for s in segs:
            rows.append({
                "Architecture": "Baseline",
                "Activation": act.capitalize(),
                "Segment": s.index,
                "x_start": round(s.x_start, 4),
                "x_end": round(s.x_end, 4),
                "Slope": round(s.slope, 6),
                "Intercept": round(s.intercept, 6)
            })

    # Proposed 1: 16 non-uniform segments
    for act in ["sigmoid", "tanh"]:
        segs = build_segments(act, "proposed", "float", 16, 8.0, 2048)
        for s in segs:
            rows.append({
                "Architecture": "Proposed 1",
                "Activation": act.capitalize(),
                "Segment": s.index,
                "x_start": round(s.x_start, 4),
                "x_end": round(s.x_end, 4),
                "Slope": round(s.slope, 6),
                "Intercept": round(s.intercept, 6)
            })

    # Proposed 2: 14 non-uniform segments for Sigmoid, 9 for Tanh
    for act, count in [("sigmoid", 14), ("tanh", 9)]:
        segs = build_segments(act, "proposed", "float", count, 8.0, 2048)
        for s in segs:
            rows.append({
                "Architecture": "Proposed 2",
                "Activation": act.capitalize(),
                "Segment": s.index,
                "x_start": round(s.x_start, 4),
                "x_end": round(s.x_end, 4),
                "Slope": round(s.slope, 6),
                "Intercept": round(s.intercept, 6)
            })

    csv_path = "architecture_partitions_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Architecture", "Activation", "Segment", "x_start", "x_end", "Slope", "Intercept"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {csv_path}")

if __name__ == "__main__":
    main()
