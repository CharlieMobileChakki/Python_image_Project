from pathlib import Path

print("Phase 10 files:")
for p in Path("benchmark_output/phase10").rglob("*"):
    print(f"  {p}")

print("\nFlow B validation files:")
for p in Path("benchmark_output/flow_b_validation").rglob("*"):
    print(f"  {p}")

