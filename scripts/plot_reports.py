import sys, os, pandas as pd
import matplotlib.pyplot as plt

def main(csv_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "benchmark.csv")
    if not os.path.exists(csv_path):
        print("No CSV found; run `make scores` first.")
        return
    df = pd.read_csv(csv_path)
    
    # Handle empty CSV (only headers, no data rows)
    if len(df) == 0:
        print("CSV contains only headers, no data rows. Creating placeholder figure...")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Placeholder Plot\n(No data rows in CSV yet)', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title("CodecEval-Africa Pipeline Test")
        plt.axis('off')
        plt.savefig(os.path.join(fig_dir, "placeholder_pipeline_test.png"), dpi=180, bbox_inches="tight")
        print("Saved placeholder figure: placeholder_pipeline_test.png")
        return
    
    if 'codec' in df.columns:
        plt.figure()
        df['codec'].value_counts().plot(kind='bar')
        plt.title("Counts by codec (placeholder)")
        plt.savefig(os.path.join(fig_dir, "counts_by_codec.png"), dpi=180, bbox_inches="tight")
        print("Saved placeholder figure: counts_by_codec.png")
    else:
        print("benchmark.csv has no 'codec' column yet (placeholder schema).")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
