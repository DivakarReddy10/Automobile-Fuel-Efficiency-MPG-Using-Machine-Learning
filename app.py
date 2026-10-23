"""
⛽ MPG Oracle - Predicting Automobile Fuel Efficiency with Machine Learning
Run with:   streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import (FEATURES, NUM_RAW, ORIGIN_LABEL, ORIGIN_MAP, get_data, load_or_train,
                         predict_row)

st.set_page_config(page_title="MPG Oracle", page_icon="⛽", layout="wide")

# --------------------------------------------------------------------------- #
# Styling & animations
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
.stApp{background:linear-gradient(-45deg,#0b1020,#10173a,#0d2238,#1a1040);
       background-size:400% 400%;animation:bgShift 18s ease infinite;}
@keyframes bgShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{to{background-position:200% center}}
@keyframes floaty{0%,100%{transform:translateY(0) rotate(-3deg)}50%{transform:translateY(-12px) rotate(3deg)}}
@keyframes drive{from{transform:translateX(-120px)}to{transform:translateX(110vw)}}
@keyframes roadMove{from{background-position:0 0}to{background-position:-80px 0}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 18px rgba(0,229,255,.25)}50%{box-shadow:0 0 42px rgba(0,229,255,.65)}}
@keyframes popIn{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.08)}100%{transform:scale(1);opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}

.hero{position:relative;overflow:hidden;padding:30px 34px 52px;border-radius:22px;margin-bottom:18px;
      background:linear-gradient(135deg,rgba(0,229,255,.12),rgba(155,89,255,.14));
      border:1px solid rgba(255,255,255,.12);animation:fadeUp .8s ease both;}
.hero h1{margin:0;font-size:2.7rem;font-weight:800;
      background:linear-gradient(90deg,#00e5ff,#a77bff,#ff7ad9,#00e5ff);background-size:200% auto;
      -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
      animation:shimmer 5s linear infinite;}
.hero p{margin:6px 0 0;color:#b9c2ee;font-size:1.05rem}
.hero .gear{position:absolute;right:28px;top:18px;font-size:3.2rem;animation:floaty 3.5s ease-in-out infinite}
.road{position:absolute;left:0;right:0;bottom:0;height:22px;background:#0a0e1d;
      border-top:2px solid rgba(255,255,255,.1)}
.road::before{content:"";position:absolute;left:0;right:0;top:9px;height:3px;
      background:repeating-linear-gradient(90deg,#ffd54f 0 40px,transparent 40px 80px);
      animation:roadMove 1s linear infinite}
.road .car{position:absolute;bottom:10px;font-size:1.7rem;animation:drive 9s linear infinite}

.card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:18px;
      padding:18px 22px;backdrop-filter:blur(8px);animation:fadeUp .7s ease both;
      transition:transform .25s,border-color .25s,box-shadow .25s}
.card:hover{transform:translateY(-4px);border-color:rgba(0,229,255,.55);box-shadow:0 10px 30px rgba(0,229,255,.15)}
.result{text-align:center;padding:26px 18px;border-radius:22px;animation:popIn .7s ease both,pulseGlow 3s ease-in-out infinite;
      background:linear-gradient(145deg,rgba(0,229,255,.15),rgba(155,89,255,.18));
      border:1px solid rgba(0,229,255,.5)}
.result .label{letter-spacing:.2em;font-size:.8rem;color:#9fb0ff;text-transform:uppercase}
.bignum{font-size:5.2rem;font-weight:900;line-height:1.05;color:#fff;text-shadow:0 0 25px rgba(0,229,255,.7)}
.bignum .count::after{content:counter(num)}
.bignum .count{counter-reset:num var(--num)}
.unit{font-size:1.3rem;color:#00e5ff;font-weight:700}
.badge{display:inline-block;margin-top:10px;padding:6px 18px;border-radius:999px;font-weight:800;
      animation:popIn .9s .3s ease both;color:#06101f}
.chip{display:inline-block;padding:4px 12px;margin:3px;border-radius:999px;font-size:.82rem;
      background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14)}
.stage{padding:8px 14px;margin:4px 0;border-left:3px solid #00e5ff;background:rgba(0,229,255,.07);
      border-radius:0 10px 10px 0;animation:fadeUp .4s ease both}
.spinner{display:inline-block;animation:spin 1s linear infinite}
.winner{padding:16px 22px;border-radius:18px;animation:popIn .8s ease both,pulseGlow 3s infinite;
      background:linear-gradient(135deg,rgba(255,213,79,.18),rgba(255,138,101,.16));
      border:1px solid rgba(255,213,79,.6);font-size:1.15rem}
div.stButton>button{border-radius:12px;font-weight:700;border:1px solid rgba(255,255,255,.18);
      transition:transform .18s,box-shadow .18s,background .18s}
div.stButton>button:hover{transform:translateY(-3px) scale(1.03);box-shadow:0 8px 22px rgba(0,229,255,.35);
      border-color:#00e5ff}
button[kind="primary"]{background:linear-gradient(90deg,#00b8d4,#7c4dff)!important;border:none!important;
      font-size:1.1rem!important;padding:.7rem 1rem!important}
button[data-baseweb="tab"]{font-weight:700;transition:transform .2s}
button[data-baseweb="tab"]:hover{transform:translateY(-2px)}
[data-testid="stMetric"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
      padding:10px 14px;border-radius:14px;animation:fadeUp .6s ease both}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><div class="gear">🚗</div>
  <h1>MPG Oracle</h1>
  <p>Describe a car. Our machine-learning pipeline tells you how far it can go on a gallon.</p>
  <div class="road"><span class="car">🚙</span></div>
</div>""", unsafe_allow_html=True)

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10))


def play_button(label="▶ Play animation", duration=60):
    return [dict(type="buttons", showactive=False, x=0.0, y=1.15, xanchor="left", buttons=[
        dict(label=label, method="animate",
             args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True,
                              transition=dict(duration=0))])])]


# --------------------------------------------------------------------------- #
# Sidebar: data + settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙️ Control Room")
    up = st.file_uploader("📂 Use your own dataset (optional)", type=["csv", "data", "txt"],
                          help="Auto-MPG format: mpg, cylinders, displacement, horsepower, weight, "
                               "acceleration, model year, origin, car name")
    st.markdown("### ⛽ Your driving")
    fuel_price = st.number_input("Fuel price (per litre)", 10.0, 500.0, 100.0, 1.0)
    annual_km = st.slider("Distance per year (km)", 1000, 50000, 12000, 500)
    st.caption("Used for the cost & CO₂ estimates.")

try:
    df, source, is_demo = get_data(up)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()


def ensure_bundle():
    key = f"bundle::{len(df)}::{source}::{float(df['mpg'].sum()):.3f}"
    if key in st.session_state:
        return st.session_state[key]
    box = st.empty()
    bar = st.progress(0.0)

    def cb(frac, text):
        box.markdown(f'<div class="stage"><span class="spinner">⚙️</span> {text}</div>',
                     unsafe_allow_html=True)
        bar.progress(min(frac, 1.0))

    b = load_or_train(df, source, save=(up is None), progress=cb)
    box.empty(); bar.empty()
    st.session_state[key] = b
    return b


bundle = ensure_bundle()
best_name = bundle["best_name"]
best = bundle["best_pipeline"]
rmse = bundle["cv_rmse"]
avg_mpg = float(df["mpg"].mean())

with st.sidebar:
    st.markdown("### 🧠 Deployed model")
    st.success(f"**{best_name}**\n\nCV RMSE ≈ {rmse:.2f} MPG")
    st.caption(f"Data: {len(df)} cars · {source}")
    if st.button("🔄 Retrain models"):
        for k in [k for k in st.session_state if str(k).startswith("bundle::")]:
            del st.session_state[k]
        from ml_pipeline import MODEL_PATH
        MODEL_PATH.unlink(missing_ok=True)
        st.rerun()

if is_demo:
    st.warning("⚠️ **Real dataset not found** – running on synthetic demo data. Put `auto-mpg.csv` "
               "into the `data/` folder (or upload it in the sidebar) to get real predictions.")

# --------------------------------------------------------------------------- #
# Session state defaults + presets
# --------------------------------------------------------------------------- #
DEFAULTS = dict(in_cyl=4, in_disp=140, in_hp=95, in_hp_unknown=False, in_wt=2600, in_acc=15.0,
                in_year=76, in_origin=1, in_name="")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)

PRESETS = {
    "🚙 Muscle Car": dict(in_cyl=8, in_disp=350, in_hp=165, in_wt=3700, in_acc=11.5, in_year=72,
                          in_origin=1, in_name="chevrolet chevelle malibu"),
    "🇯🇵 Economy Hatch": dict(in_cyl=4, in_disp=91, in_hp=67, in_wt=1850, in_acc=15.5, in_year=80,
                              in_origin=3, in_name="honda civic"),
    "🇩🇪 Euro Sedan": dict(in_cyl=4, in_disp=121, in_hp=110, in_wt=2800, in_acc=15.5, in_year=75,
                           in_origin=2, in_name="bmw 2002"),
    "🚐 Heavy Wagon": dict(in_cyl=8, in_disp=455, in_hp=225, in_wt=4300, in_acc=10.0, in_year=73,
                           in_origin=1, in_name="buick estate wagon"),
    "🌿 Eco Diesel": dict(in_cyl=4, in_disp=97, in_hp=48, in_wt=2100, in_acc=23.7, in_year=80,
                          in_origin=2, in_name="vw rabbit c diesel"),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
    st.session_state["in_hp_unknown"] = False


def current_row():
    s = st.session_state
    return dict(cylinders=s.in_cyl, displacement=s.in_disp,
                horsepower=None if s.in_hp_unknown else s.in_hp, weight=s.in_wt,
                acceleration=s.in_acc, model_year=s.in_year, origin=s.in_origin,
                car_name=s.in_name or None)


def eco_grade(m):
    if m >= 35: return "A+  Eco Champion", "#00e676"
    if m >= 28: return "A  Efficient", "#76ff03"
    if m >= 22: return "B  Average", "#ffd740"
    if m >= 17: return "C  Thirsty", "#ff9100"
    return "D  Gas Guzzler", "#ff5252"


tab_pred, tab_whatif, tab_arena, tab_data, tab_about = st.tabs(
    ["🎯 Predict", "🔮 What-If Lab", "🏁 Model Arena", "📊 Data Explorer", "🧬 How it works"])

# --------------------------------------------------------------------------- #
# TAB 1 - Predict
# --------------------------------------------------------------------------- #
with tab_pred:
    st.markdown("##### ⚡ Quick-start with a preset car")
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), key=f"pre_{name}")

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        st.markdown("#### 🛠️ Vehicle specifications")
        c1, c2 = st.columns(2)
        c1.select_slider("🔩 Cylinders", options=[3, 4, 5, 6, 8], key="in_cyl")
        c2.radio("🌍 Origin", [1, 2, 3], key="in_origin", horizontal=True,
                 format_func=lambda o: ORIGIN_LABEL[o])
        st.slider("🧪 Displacement (cu. inches)", 60, 470, key="in_disp")
        h1, h2 = st.columns([2, 1])
        h1.slider("🐎 Horsepower", 40, 250, key="in_hp", disabled=st.session_state.in_hp_unknown)
        h2.checkbox("Don't know", key="in_hp_unknown",
                    help="The pipeline will impute the missing value automatically!")
        st.slider("⚖️ Weight (lbs)", 1500, 5200, step=10, key="in_wt")
        st.slider("⏱️ Acceleration (0-60, seconds)", 7.0, 25.0, step=0.1, key="in_acc")
        st.slider("📅 Model year (19xx)", 70, 82, key="in_year")
        st.text_input("🏷️ Car name (optional – brand is used as a feature)", key="in_name",
                      placeholder="e.g. toyota corolla")
        go_btn = st.button("🚀  Predict MPG", type="primary", use_container_width=True)

    if go_btn:
        stages = ["🧹 Detecting & imputing missing values …",
                  "🧬 Engineering features (brand, power-to-weight …)",
                  "📏 Scaling numeric features …",
                  "🔤 One-hot encoding categories …",
                  f"🤖 Asking {best_name} …"]
        with right:
            ph, pb = st.empty(), st.progress(0.0)
            shown = ""
            for i, s in enumerate(stages):
                shown += f'<div class="stage">{s}</div>'
                ph.markdown(shown, unsafe_allow_html=True)
                pb.progress((i + 1) / len(stages))
                time.sleep(0.28)
            ph.empty(); pb.empty()
        row = current_row()
        pred = predict_row(best, row)
        others = {n: predict_row(p, row) for n, p in bundle["all_pipelines"].items()}
        st.session_state["last"] = dict(row=row, pred=pred, others=others, fresh=True)
        st.session_state["history"].append(
            dict(Time=time.strftime("%H:%M:%S"), Name=row["car_name"] or "—", Cyl=row["cylinders"],
                 Disp=row["displacement"], HP=row["horsepower"] if row["horsepower"] else "imputed",
                 Weight=row["weight"], Year=row["model_year"], Origin=ORIGIN_MAP[row["origin"]],
                 **{"Predicted MPG": round(pred, 1)}))

    with right:
        last = st.session_state["last"]
        if not last:
            st.markdown("""<div class="card" style="text-align:center;padding:60px 20px">
              <div style="font-size:4rem;animation:floaty 3s ease-in-out infinite">🔮</div>
              <h3>Your prediction will appear here</h3>
              <p style="color:#9fb0ff">Pick a preset or dial in your own specs, then hit
              <b>Predict MPG</b>.</p></div>""", unsafe_allow_html=True)
        else:
            pred = last["pred"]
            grade, color = eco_grade(pred)
            kmpl, l100 = pred * 0.425144, 235.215 / pred
            uid = int(time.time() * 1000) % 10_000_000
            st.markdown(f"""
            <style>@property --num{{syntax:'<integer>';initial-value:0;inherits:false}}
            @keyframes cu{uid}{{from{{--num:0}}to{{--num:{int(pred)}}}}}
            .count{{--num:{int(pred)};animation:cu{uid} 1.6s ease-out}}</style>
            <div class="result"><div class="label">Predicted fuel efficiency</div>
              <div class="bignum"><span class="count"></span><span style="font-size:3rem">.{int(round((pred - int(pred)) * 10)) % 10}</span>
              <span class="unit">MPG</span></div>
              <span class="badge" style="background:{color}">{grade}</span>
              <div style="margin-top:10px;color:#b9c2ee">± {rmse:.1f} MPG typical error ·
              {pred * 0.425144:.1f} km/L · {235.215 / pred:.1f} L/100 km</div></div>""",
                        unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=pred, number=dict(suffix=" MPG"),
                delta=dict(reference=avg_mpg, suffix=" vs dataset avg"),
                gauge=dict(axis=dict(range=[5, 50]), bar=dict(color=color, thickness=.3),
                           steps=[dict(range=[5, 17], color="rgba(255,82,82,.25)"),
                                  dict(range=[17, 22], color="rgba(255,145,0,.25)"),
                                  dict(range=[22, 28], color="rgba(255,215,64,.25)"),
                                  dict(range=[28, 35], color="rgba(118,255,3,.25)"),
                                  dict(range=[35, 50], color="rgba(0,230,118,.25)")],
                           threshold=dict(line=dict(color="white", width=3), value=avg_mpg))))
            fig.update_layout(height=270, **PLOT)
            st.plotly_chart(fig)

            litres = annual_km / kmpl
            cost = litres * fuel_price
            avg_cost = annual_km / (avg_mpg * 0.425144) * fuel_price
            pct = float((df["mpg"] < pred).mean() * 100)
            m1, m2, m3 = st.columns(3)
            m1.metric("💸 Fuel / year", f"{cost:,.0f}", f"{avg_cost - cost:+,.0f} vs avg car",
                      delta_color="normal")
            m2.metric("🌫️ CO₂ / year", f"{litres * 2.31:,.0f} kg")
            m3.metric("🏆 Beats", f"{pct:.0f}% of cars")

            if last.get("fresh"):
                if pred >= 30:
                    st.balloons()
                elif pred < 15:
                    st.snow()
                last["fresh"] = False

    # second row: model consensus + history
    last = st.session_state["last"]
    if last:
        st.markdown("#### 🗳️ What every model thinks")
        o = pd.DataFrame({"Model": list(last["others"]), "MPG": list(last["others"].values())})
        o["Deployed"] = np.where(o["Model"] == best_name, "★ Deployed", "Other")
        f2 = px.bar(o, x="MPG", y="Model", orientation="h", color="Deployed", text=o["MPG"].round(1),
                    color_discrete_map={"★ Deployed": "#00e5ff", "Other": "#7c4dff"})
        f2.update_layout(height=260, yaxis=dict(categoryorder="total ascending"), **PLOT)
        st.plotly_chart(f2)
    if st.session_state["history"]:
        with st.expander(f"🕘 Prediction history ({len(st.session_state['history'])})"):
            hdf = pd.DataFrame(st.session_state["history"])
            st.dataframe(hdf)
            hc1, hc2 = st.columns(2)
            hc1.download_button("⬇️ Download history (CSV)", hdf.to_csv(index=False).encode(),
                                "mpg_predictions.csv", "text/csv")
            if hc2.button("🗑️ Clear history"):
                st.session_state["history"] = []
                st.rerun()

# --------------------------------------------------------------------------- #
# TAB 2 - What-If lab
# --------------------------------------------------------------------------- #
with tab_whatif:
    st.markdown("### 🔮 What happens if I change…?")
    st.caption("Uses the specs currently set in the Predict tab as the baseline.")
    base = current_row()
    base_pred = predict_row(best, base)
    SWEEP = {"weight": ("⚖️ Weight (lbs)", 1500, 5200, 38),
             "horsepower": ("🐎 Horsepower", 40, 250, 36),
             "displacement": ("🧪 Displacement", 60, 470, 40),
             "acceleration": ("⏱️ Acceleration", 7, 25, 36),
             "model_year": ("📅 Model year", 70, 82, 13),
             "cylinders": ("🔩 Cylinders", 3, 8, 6)}
    feat = st.selectbox("Sweep this feature", list(SWEEP), format_func=lambda k: SWEEP[k][0])
    label, lo, hi, n = SWEEP[feat]
    xs = np.linspace(lo, hi, n)
    grid = pd.DataFrame([{**base, feat: x, "horsepower": base["horsepower"] if base["horsepower"]
                          is not None or feat == "horsepower" else np.nan} for x in xs])
    ys = best.predict(grid[FEATURES])
    cur = base[feat] if base[feat] is not None else float(np.nanmedian(df[feat]))
    frames = [go.Frame(data=[go.Scatter(x=xs[:k], y=ys[:k])], traces=[0]) for k in range(2, n + 1)]
    fig = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="lines+markers", name="Predicted MPG",
                                     line=dict(color="#00e5ff", width=4, shape="spline"),
                                     marker=dict(size=6)),
                          go.Scatter(x=[cur], y=[base_pred], mode="markers", name="Your car",
                                     marker=dict(size=18, color="#ff7ad9", symbol="star",
                                                 line=dict(color="white", width=2)))],
                    frames=frames)
    fig.update_layout(height=420, title=f"Predicted MPG vs {label}", xaxis_title=label,
                      yaxis_title="MPG", yaxis_range=[max(0, ys.min() - 4), ys.max() + 4],
                      xaxis_range=[lo, hi], updatemenus=play_button("▶ Replay curve"), **PLOT)
    st.plotly_chart(fig)

    st.markdown("#### 🌡️ Weight × Horsepower heat-map")
    ws, hs = np.linspace(1500, 5200, 26), np.linspace(40, 250, 26)
    g = pd.DataFrame([{**base, "weight": w, "horsepower": h} for h in hs for w in ws])
    z = best.predict(g[FEATURES]).reshape(len(hs), len(ws))
    hm = go.Figure([go.Contour(x=ws, y=hs, z=z, colorscale="Turbo", contours_coloring="heatmap",
                               colorbar=dict(title="MPG")),
                    go.Scatter(x=[base["weight"]], y=[base["horsepower"] or df["horsepower"].median()],
                               mode="markers", marker=dict(size=16, color="white", symbol="star",
                                                           line=dict(color="black", width=2)),
                               name="Your car")])
    hm.update_layout(height=430, xaxis_title="Weight (lbs)", yaxis_title="Horsepower", **PLOT)
    st.plotly_chart(hm)

    st.markdown("#### 🥗 Weight-diet challenge")
    cut = st.slider("Shave off weight (lbs)", 0, 800, 200, 50)
    lighter = predict_row(best, {**base, "weight": max(1500, base["weight"] - cut)})
    st.metric("MPG after weight loss", f"{lighter:.1f}", f"{lighter - base_pred:+.2f} MPG")

# --------------------------------------------------------------------------- #
# TAB 3 - Model arena
# --------------------------------------------------------------------------- #
with tab_arena:
    res = bundle["results"]
    st.markdown("### 🏁 Five algorithms entered. One was deployed.")
    r0 = res.iloc[0]
    st.markdown(f"""<div class="winner">🏆 <b>{best_name}</b> wins with the lowest cross-validated
      RMSE (<b>{r0['CV RMSE (5-fold)']:.2f}</b> MPG) · test R² <b>{r0['R² (test)']:.3f}</b> ·
      test MAE <b>{r0['MAE (test)']:.2f}</b> MPG</div>""", unsafe_allow_html=True)
    st.write("")
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"})
                 .highlight_min(subset=["RMSE (test)", "MAE (test)", "CV RMSE (5-fold)"], color="#0d5c4a")
                 .highlight_max(subset=["R² (test)"], color="#0d5c4a"))
    steps = 25
    fr = [go.Frame(data=[go.Bar(x=res["R² (test)"] * k / steps, y=res["Model"], orientation="h")],
                   traces=[0]) for k in range(1, steps + 1)]
    race = go.Figure(data=[go.Bar(x=res["R² (test)"] * 0, y=res["Model"], orientation="h",
                                  marker=dict(color=res["R² (test)"], colorscale="Viridis"))],
                     frames=fr)
    race.update_layout(height=340, title="R² race (higher is better)",
                       xaxis_range=[0, max(1.0, float(res["R² (test)"].max()) + .05)],
                       yaxis=dict(autorange="reversed"), updatemenus=play_button("🏁 Start race", 40),
                       **PLOT)
    st.plotly_chart(race)

    a1, a2 = st.columns(2)
    tp = bundle["test_pred"]
    sc = px.scatter(tp, x="actual", y="predicted", title="Actual vs predicted (hold-out set)",
                    color_discrete_sequence=["#00e5ff"], opacity=.8)
    lim = [tp.min().min() - 2, tp.max().max() + 2]
    sc.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                 line=dict(color="#ff7ad9", dash="dash"))
    sc.update_layout(height=380, **PLOT)
    a1.plotly_chart(sc)
    imp = bundle["importance"]
    bf = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance",
                color_continuous_scale="Turbo", title="Permutation importance (what drives MPG)")
    bf.update_layout(height=380, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, **PLOT)
    a2.plotly_chart(bf)
    resid = px.histogram(tp.assign(error=tp.predicted - tp.actual), x="error", nbins=25,
                         title="Residuals (predicted − actual)", color_discrete_sequence=["#a77bff"])
    resid.update_layout(height=300, **PLOT)
    st.plotly_chart(resid)

# --------------------------------------------------------------------------- #
# TAB 4 - Data explorer
# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown("### 📊 Explore the dataset")
    d = df.copy()
    d["origin_label"] = d["origin"].map(ORIGIN_LABEL).fillna("Unknown")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚗 Cars", len(d))
    k2.metric("⛽ Avg MPG", f"{d.mpg.mean():.1f}")
    k3.metric("🏅 Best MPG", f"{d.mpg.max():.1f}")
    k4.metric("🕳️ Missing cells", int(d[NUM_RAW + ["origin"]].isna().sum().sum()))

    f1, f2_ = st.columns(2)
    orig_sel = f1.multiselect("Origin", list(d["origin_label"].unique()),
                              default=list(d["origin_label"].unique()))
    cyl_sel = f2_.multiselect("Cylinders", sorted(d["cylinders"].dropna().unique()),
                              default=sorted(d["cylinders"].dropna().unique()))
    dd = d[d["origin_label"].isin(orig_sel) & d["cylinders"].isin(cyl_sel)]

    x_ax = st.selectbox("X axis", NUM_RAW, index=3)
    sc2 = px.scatter(dd, x=x_ax, y="mpg", color="origin_label", size=dd["horsepower"].fillna(
        d["horsepower"].median()), hover_name="car_name", trendline=None, opacity=.8,
        title=f"MPG vs {x_ax}")
    sc2.update_layout(height=420, **PLOT)
    st.plotly_chart(sc2)

    st.markdown("#### 🎞️ MPG through the years (press ▶)")
    an = d.copy()
    for c in ["horsepower", "weight"]:
        an[c] = an[c].fillna(an[c].median())
    an = an.dropna(subset=["model_year"]).sort_values("model_year")
    an["model_year"] = an["model_year"].astype(int)
    anim = px.scatter(an, x="weight", y="mpg", animation_frame="model_year", size="horsepower",
                      color="origin_label", hover_name="car_name", size_max=26,
                      range_x=[an.weight.min() - 100, an.weight.max() + 100],
                      range_y=[an.mpg.min() - 3, an.mpg.max() + 3])
    anim.update_layout(height=480, **PLOT)
    st.plotly_chart(anim)

    cc1, cc2 = st.columns(2)
    corr = px.imshow(d[["mpg"] + NUM_RAW].corr(), text_auto=".2f", zmin=-1, zmax=1,
                     color_continuous_scale="RdBu_r", title="Correlation heat-map")
    corr.update_layout(height=430, **PLOT)
    cc1.plotly_chart(corr)
    bx = px.box(d, x="cylinders", y="mpg", color="origin_label", title="MPG by cylinders")
    bx.update_layout(height=430, **PLOT)
    cc2.plotly_chart(bx)
    with st.expander("🔎 Raw data"):
        st.dataframe(d.drop(columns="origin_label"))

# --------------------------------------------------------------------------- #
# TAB 5 - How it works
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("### 🧬 The pipeline, step by step")
    steps = [("1️⃣ Raw input", "Your specs (missing values allowed)"),
             ("2️⃣ Feature engineering", "Brand from car name, origin name, power-to-weight, displacement per cylinder"),
             ("3️⃣ Imputation", "Numeric → median · Categorical → constant 'other'"),
             ("4️⃣ Scaling", "StandardScaler on all numeric features"),
             ("5️⃣ Encoding", "One-hot encoding of origin & brand (unknown brands ignored safely)"),
             ("6️⃣ Model", f"{best_name} – picked from 5 candidates by 5-fold CV RMSE")]
    cols = st.columns(3)
    for i, (t, dsc) in enumerate(steps):
        cols[i % 3].markdown(f'<div class="card" style="animation-delay:{i * .12}s;min-height:130px">'
                             f'<b>{t}</b><br><span style="color:#b9c2ee">{dsc}</span></div>',
                             unsafe_allow_html=True)
        if i % 3 == 2:
            cols = st.columns(3)
    st.markdown("#### 🕳️ Missing values found in the training data")
    miss = d[NUM_RAW + ["origin", "car_name"]].isna().sum().rename("Missing").to_frame()
    miss["% of rows"] = (miss["Missing"] / len(d) * 100).round(2)
    st.dataframe(miss)
    st.markdown("""
**Why one single `Pipeline`?** Preprocessing is fitted on the training folds only (no data leakage) and is
saved *inside* the model, so the exact same imputation / scaling / encoding is applied to every
prediction made in this app.

**Evaluation:** 80/20 hold-out split + 5-fold cross-validation on the training part.
The model with the lowest CV RMSE is deployed and refitted on 100 % of the data.

> The model predicts typical MPG for cars like the dataset (1970-1982). Treat it as an estimate, not a guarantee.
""")
