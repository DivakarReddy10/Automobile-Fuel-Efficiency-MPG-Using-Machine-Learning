# ⛽ MPG Oracle – Predicting Automobile Fuel Efficiency with ML

Interactive Streamlit app: enter vehicle specs → get a predicted MPG.

## Run
```bash
pip install -r requirements.txt
# 1) put your dataset at data/auto-mpg.csv   (columns: mpg, cylinders, displacement, horsepower,
#    weight, acceleration, model year, origin, car name; '?' for missing is fine)
streamlit run app.py
```
First launch trains 5 models (~10-20 s), picks the best by 5-fold CV RMSE and saves it to
`model/mpg_model.joblib`. You can also pre-train with `python train.py`.
You can also upload the dataset from the app sidebar.

## Files
- `app.py` – Streamlit UI (animations, gauge, what-if lab, model arena, data explorer)
- `ml_pipeline.py` – loading, preprocessing pipeline, training & model selection
- `train.py` – optional CLI trainer
- `.streamlit/config.toml` – dark theme

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub (include `data/auto-mpg.csv`), then deploy `app.py`.
