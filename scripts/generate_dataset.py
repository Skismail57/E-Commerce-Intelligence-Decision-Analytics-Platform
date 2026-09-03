import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.ingestion.generate_synthetic_data import SyntheticDataGenerator

SCALE = sys.argv[1] if len(sys.argv) > 1 else "medium"

presets = {
    "tiny":   (100,    50,    500,    "2024-01-01", "2024-06-30"),
    "small":  (2000,   300,   5000,   "2023-01-01", "2024-12-31"),
    "medium": (25000,  1200,  50000,  "2022-01-01", "2024-12-31"),
    "large":  (100000, 5000,  200000, "2022-01-01", "2024-12-31"),
}

if SCALE not in presets:
    print(f"Unknown scale: {SCALE}. Choices: {list(presets.keys())}")
    sys.exit(1)

nc, np_, no, sd, ed = presets[SCALE]
print(f"[SCALE: {SCALE.upper()}] Customers={nc:,}, Products={np_:,}, Orders={no:,}, Date range={sd}..{ed}")

g = SyntheticDataGenerator(
    random_state=42,
    start_date=sd,
    end_date=ed,
    num_customers=nc,
    num_products=np_,
    num_orders=no,
)
datasets = g.generate_all()
g.save_to_csv()
total = sum(len(v) for v in datasets.values())
print(f"\nDone! {total:,} total rows saved across {len(datasets)} tables.")
print("=" * 70)
for n, d in datasets.items():
    print(f"  {n:<22} {len(d):>10,} rows")
print("=" * 70)
