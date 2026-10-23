"""Optional: train & save the model from the command line:  python train.py"""
from ml_pipeline import MODEL_PATH, get_data, load_or_train

df, source, is_demo = get_data()
print(f"Data: {len(df)} rows from {source}")
if is_demo:
    print("WARNING: real dataset not found - put auto-mpg.csv inside the data/ folder!")
b = load_or_train(df, source, progress=lambda f, t: print(f"[{f:4.0%}] {t}"))
print(b["results"].round(3).to_string(index=False))
print(f"\nBest model: {b['best_name']}  (saved to {MODEL_PATH})")
