import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import numpy as np
from plotly.io import to_html
import zipfile
import io
import plotly.express as px
import hashlib

# ----------------------------
# Helpers
# ----------------------------
def file_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def coerce_numeric_allow_blanks(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    raw = raw.str.replace(",", "", regex=False)
    return pd.to_numeric(raw, errors="coerce")

def find_true_non_numeric_examples(original_series: pd.Series, coerced: pd.Series, max_examples: int = 5):
    raw = original_series.astype(str).str.strip()
    na_like = {"", "nan", "NaN", "none", "None", "null", "NULL", "NA", "N/A"}
    bad_mask = coerced.isna() & ~raw.isin(na_like)
    examples = raw[bad_mask].dropna().unique()
    return list(examples[:max_examples])

def ensure_timedelta_days(td_series: pd.Series) -> pd.Series:
    if hasattr(td_series, "dt") and np.issubdtype(td_series.dtype, np.timedelta64):
        return td_series.dt.total_seconds() / 86400.0
    return td_series

def normalize_backtest_frames(cv_df: pd.DataFrame | None, perf_df: pd.DataFrame | None):
    # Ensure a real 'horizon' column exists
    if cv_df is not None and not cv_df.empty:
        if "horizon" not in cv_df.columns:
            if getattr(cv_df.index, "name", None) == "horizon":
                cv_df = cv_df.reset_index()
            elif "cutoff" in cv_df.columns and "ds" in cv_df.columns:
                cv_df = cv_df.copy()
                cv_df["horizon"] = pd.to_datetime(cv_df["ds"]) - pd.to_datetime(cv_df["cutoff"])

    if perf_df is not None and not perf_df.empty:
        if "horizon" not in perf_df.columns:
            if getattr(perf_df.index, "name", None) == "horizon":
                perf_df = perf_df.reset_index()
            else:
                perf_df = perf_df.reset_index()

    return cv_df, perf_df

def bt_params_tuple(bt_initial_days, bt_period_days, bt_horizon_days, freq,
                    selected_regressors, event_regressors,
                    changepoint_prior_scale, seasonality_prior_scale, manual_changepoints,
                    seasonality_mode, use_log_y):
    return (
        int(bt_initial_days),
        int(bt_period_days),
        int(bt_horizon_days),
        str(freq),
        tuple(selected_regressors),
        tuple(sorted(event_regressors)),
        float(changepoint_prior_scale),
        float(seasonality_prior_scale),
        str(manual_changepoints).strip(),
        str(seasonality_mode),
        bool(use_log_y),
    )

def safe_log1p(y: pd.Series) -> pd.Series:
    return np.log1p(y.clip(lower=0))

def safe_expm1(y: pd.Series) -> pd.Series:
    return np.expm1(y)

def render_interpretable_backtest_views(cv_df: pd.DataFrame):
    st.subheader("Backtest: Interpretable Views")

    cv = cv_df.copy()
    if "horizon" not in cv.columns:
        st.warning("Backtest output is missing 'horizon'.")
        return
    if not {"ds", "y", "yhat"}.issubset(set(cv.columns)):
        st.warning("Backtest output is missing one of: ds, y, yhat")
        return

    cv["ds"] = pd.to_datetime(cv["ds"], errors="coerce")
    if "cutoff" in cv.columns:
        cv["cutoff"] = pd.to_datetime(cv["cutoff"], errors="coerce")

    cv["abs_err"] = (cv["y"] - cv["yhat"]).abs()
    cv["ape"] = (cv["abs_err"] / cv["y"].replace(0, np.nan)).astype(float)

    if "yhat_lower" in cv.columns and "yhat_upper" in cv.columns:
        cv["in_80"] = (cv["y"] >= cv["yhat_lower"]) & (cv["y"] <= cv["yhat_upper"])
    else:
        cv["in_80"] = np.nan

    cv["horizon_days"] = ensure_timedelta_days(cv["horizon"])
    cv["month"] = cv["ds"].dt.month

    mape = float(cv["ape"].dropna().mean() * 100) if cv["ape"].notna().any() else np.nan
    median_ape = float(cv["ape"].dropna().median() * 100) if cv["ape"].notna().any() else np.nan
    mae = float(cv["abs_err"].mean()) if cv["abs_err"].notna().any() else np.nan
    coverage = float(cv["in_80"].dropna().mean() * 100) if cv["in_80"].notna().any() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE (avg)", f"{mape:.1f}%" if np.isfinite(mape) else "n/a")
    c2.metric("APE (median)", f"{median_ape:.1f}%" if np.isfinite(median_ape) else "n/a")
    c3.metric("MAE (avg)", f"{mae:,.2f}" if np.isfinite(mae) else "n/a")
    c4.metric("80% band coverage", f"{coverage:.1f}%" if np.isfinite(coverage) else "n/a")

    # Horizon buckets (fast read)
    bins = [-np.inf, 7, 14, 30, 60, 90, 120, 180, np.inf]
    labels = ["≤7d", "8–14d", "15–30d", "31–60d", "61–90d", "91–120d", "121–180d", "180d+"]
    cv["h_bucket"] = pd.cut(cv["horizon_days"], bins=bins, labels=labels)

    bucket = cv.groupby("h_bucket", dropna=False).agg(
        n=("y", "size"),
        mape=("ape", lambda s: float(s.dropna().mean() * 100) if s.notna().any() else np.nan),
        mae=("abs_err", "mean"),
        coverage=("in_80", lambda s: float(s.dropna().mean() * 100) if s.notna().any() else np.nan),
    ).reset_index()

    st.dataframe(bucket)

def compute_baseline_history_view(model: Prophet, hist_df: pd.DataFrame, regressor_cols: list[str], use_log_y: bool):
    pred_in = hist_df[["ds"] + regressor_cols].copy() if regressor_cols else hist_df[["ds"]].copy()
    pred_out = model.predict(pred_in)

    out = pd.DataFrame({
        "ds": pd.to_datetime(pred_out["ds"]),
        "trend": pred_out["trend"].astype(float),
    })

    out = out.merge(hist_df[["ds", "y"]], on="ds", how="left").rename(columns={"y": "y_model_scale"})
    if use_log_y:
        out["y_actual"] = safe_expm1(out["y_model_scale"])
        out["baseline"] = safe_expm1(out["trend"])
    else:
        out["y_actual"] = out["y_model_scale"]
        out["baseline"] = out["trend"]

    out["ratio_actual_to_baseline"] = out["y_actual"] / out["baseline"].replace(0, np.nan)
    return out.sort_values("ds")

def run_baseline_only_cv(train_df_model_scale: pd.DataFrame,
                         cps: float, sps: float, seasonality_mode: str,
                         bt_initial_days: int, bt_period_days: int, bt_horizon_days: int):
    base_model = Prophet(
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        seasonality_mode=seasonality_mode,
    )
    base_model.fit(train_df_model_scale[["ds", "y"]])

    initial = f"{int(bt_initial_days)} days"
    period = f"{int(bt_period_days)} days"
    horizon = f"{int(bt_horizon_days)} days"

    cv = cross_validation(
        base_model,
        initial=initial,
        period=period,
        horizon=horizon,
        parallel=None,
    )
    perf = performance_metrics(cv)
    cv, perf = normalize_backtest_frames(cv, perf)
    return base_model, cv, perf

def stability_sweep_trends(train_df_model_scale: pd.DataFrame,
                           cps_values: list[float],
                           sps: float, seasonality_mode: str,
                           hist_ds: pd.Series):
    curves = []
    for cps in cps_values:
        m = Prophet(
            changepoint_prior_scale=float(cps),
            seasonality_prior_scale=float(sps),
            seasonality_mode=seasonality_mode,
        )
        m.fit(train_df_model_scale[["ds", "y"]])
        pred = m.predict(pd.DataFrame({"ds": hist_ds}))
        curves.append(pd.Series(pred["trend"].astype(float).values, name=str(cps)))
    trend_mat = pd.concat(curves, axis=1)
    return trend_mat

def suggest_event_like(cols: list[str]) -> list[str]:
    # Heuristic: people often name these like event_, shock_, promo_, campaign_
    prefixes = ("event", "shock", "promo", "campaign", "update", "algo")
    out = []
    for c in cols:
        lc = c.lower()
        if any(lc.startswith(p) for p in prefixes) or any(p in lc for p in prefixes):
            out.append(c)
    return out

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Prophet Forecasting App", layout="wide")

st.title("Prophet Forecasting App")
st.markdown(
    "This App takes Meta's Prophet Forecasting model and lets you apply it to SEO data to establish forecasts based on historic data. "
    "By Chris Green - https://www.chris-green.net/"
)

with st.expander("How to use this app"):
    st.markdown("""
    **Step 1. Upload your CSV file**  
    **Step 2. Pick regressors in the sidebar (no re-upload needed)**  
    **Step 3. Run forecast**  
    **Step 4. (Optional) Backtest + Baseline tests**
    """)

# ----------------------------
# Session init
# ----------------------------
defaults = {
    "file_hash": None,
    "data_raw": None,               # original parsed dataframe
    "cols": None,                   # list of columns
    "date_col": None,
    "metric_col": None,
    "all_regressors": [],
    # cached results keyed by params
    "bt_cv_df": None,
    "bt_perf_df": None,
    "bt_params": None,
    "bt_cv_base_df": None,
    "bt_perf_base_df": None,
    "bt_base_params": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Sidebar config
# ----------------------------
st.sidebar.header("Forecast Settings")

with st.sidebar.form("config_form", clear_on_submit=False):
    st.subheader("Target transform")
    use_log_y = st.checkbox(
        "Model on log(1+y) scale (recommended for SEO)",
        value=True,
        help="Stabilises variance and makes seasonality/impacts proportional."
    )

    st.subheader("Seasonality")
    seasonality_mode = st.selectbox(
        "Seasonality mode",
        ["multiplicative", "additive"],
        index=0,
        help="Multiplicative usually fits SEO better."
    )

    st.subheader("Model")
    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.0, 1.0, 0.05, 0.01)
    seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0, 0.5)
    manual_changepoints = st.text_area(
        "Manual Changepoints (comma-separated dates, e.g., 2024-01-01,2024-06-01)", ""
    )

    st.subheader("Data frequency & horizon")
    freq_choice = st.selectbox("Frequency", ["Infer Automatically", "Daily", "Weekly", "Monthly", "Yearly"], 0)
    forecast_periods = st.number_input(
        "Future periods to forecast (only used if file has no blank future metric rows)",
        min_value=0, value=0, step=1
    )

    st.subheader("Backtest (Cross-Validation)")
    run_backtest = st.checkbox("Run backtest (CV)", value=False)
    bt_initial_days = st.number_input("Initial training window (days)", min_value=30, value=365, step=30)
    bt_period_days = st.number_input("Period between cutoffs (days)", min_value=7, value=30, step=7)
    bt_horizon_days = st.number_input("Forecast horizon (days)", min_value=7, value=90, step=7)

    st.subheader("Baseline tests")
    run_baseline_tests = st.checkbox(
        "Run baseline tests (trend reliability)",
        value=True,
        help="Adds baseline-vs-actual, baseline-only CV, and stability sweep."
    )
    sweep_points = st.selectbox("Baseline stability sweep size", [3, 5, 7], index=1)

    applied = st.form_submit_button("Apply settings")

st.sidebar.divider()
run_clicked = st.sidebar.button("Run Forecast", type="primary")

# ----------------------------
# File upload + parse once
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file (first col = date, second col = metric, optional regressors after)",
    type=["csv"]
)

if not uploaded_file:
    st.info("Upload a CSV to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
fh = file_hash_bytes(file_bytes)

# New file -> reset dataset-specific state
if st.session_state.file_hash != fh:
    st.session_state.file_hash = fh
    st.session_state.data_raw = None
    st.session_state.cols = None
    st.session_state.date_col = None
    st.session_state.metric_col = None
    st.session_state.all_regressors = []

    # clear cached computations tied to old data
    st.session_state.bt_cv_df = None
    st.session_state.bt_perf_df = None
    st.session_state.bt_params = None
    st.session_state.bt_cv_base_df = None
    st.session_state.bt_perf_base_df = None
    st.session_state.bt_base_params = None

# Parse data if not already parsed
if st.session_state.data_raw is None:
    data = pd.read_csv(io.BytesIO(file_bytes))
    st.session_state.data_raw = data
    st.session_state.cols = list(data.columns)

    if len(data.columns) < 2:
        st.error("The file must have at least two columns: date + metric.")
        st.stop()

    st.session_state.date_col = data.columns[0]
    st.session_state.metric_col = data.columns[1]
    st.session_state.all_regressors = list(data.columns[2:])

data = st.session_state.data_raw.copy()
date_col = st.session_state.date_col
metric_col = st.session_state.metric_col
all_regressors = st.session_state.all_regressors

st.write("Uploaded Data Preview:", data.head())
st.info(f"Date column: '{date_col}', Metric column: '{metric_col}'")
st.info(f"Available regressors: {', '.join(all_regressors) if all_regressors else '(none)'}")

# ----------------------------
# Regressor selection (no re-upload needed)
# ----------------------------
st.sidebar.subheader("Regressors to include")
selected_regressors = st.sidebar.multiselect(
    "Select regressors for this run",
    options=all_regressors,
    default=st.session_state.get("selected_regressors", all_regressors),
    help="Pick which columns Prophet should use as regressors for this run."
)
st.session_state.selected_regressors = selected_regressors

suggested_events = [c for c in suggest_event_like(selected_regressors)]
st.sidebar.subheader("Event/shock regressors")
event_regressors = st.sidebar.multiselect(
    "Treat these as event flags (future defaults to 0)",
    options=selected_regressors,
    default=st.session_state.get("event_regressors", suggested_events),
    help="Event flags should NOT be carried into the future. They default to 0 for future dates."
)
st.session_state.event_regressors = event_regressors

# ----------------------------
# Validate + coerce types
# ----------------------------
try:
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if data[date_col].isna().any():
        st.error(f"Invalid dates detected in column '{date_col}'.")
        st.stop()

    metric_coerced = coerce_numeric_allow_blanks(data[metric_col])
    bad_metric = find_true_non_numeric_examples(data[metric_col], metric_coerced)
    if bad_metric:
        st.error(f"Non-numeric values detected in metric column '{metric_col}'. Examples: {', '.join(bad_metric)}")
        st.stop()
    data[metric_col] = metric_coerced

    for reg_col in selected_regressors:
        reg_coerced = coerce_numeric_allow_blanks(data[reg_col])
        bad_reg = find_true_non_numeric_examples(data[reg_col], reg_coerced)
        if bad_reg:
            st.error(f"Non-numeric values detected in regressor '{reg_col}'. Examples: {', '.join(bad_reg)}")
            st.stop()
        data[reg_col] = reg_coerced

except Exception as e:
    st.error(f"Failed to validate/coerce the uploaded file: {e}")
    st.stop()

# ----------------------------
# Only run heavy work on button click
# ----------------------------
if not run_clicked:
    st.info("Pick regressors in the sidebar, then click **Run Forecast**.")
    st.stop()

# ----------------------------
# Main logic
# ----------------------------
try:
    df = data.rename(columns={date_col: "ds", metric_col: "y_raw"}).copy()
    df = df.sort_values("ds").reset_index(drop=True)

    # Apply target transform (model scale)
    df["y"] = safe_log1p(df["y_raw"]) if use_log_y else df["y_raw"]

    # Resolve frequency
    if freq_choice == "Infer Automatically":
        inferred = pd.infer_freq(df["ds"])
        if inferred:
            freq = inferred
            st.info(f"Inferred frequency: {freq}")
        else:
            freq = "D"
            st.warning("Could not infer frequency. Defaulting to Daily ('D').")
    else:
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
        freq = freq_map[freq_choice]

    st.write("Processed Data (pre-resample):", df[["ds", "y_raw", "y"] + selected_regressors].head())

    with st.spinner("Running forecast..."):
        # Resample
        df = df.set_index("ds").asfreq(freq).reset_index()

        # Fill regressors after resampling (only selected ones)
        for reg_col in selected_regressors:
            if reg_col in df.columns and df[reg_col].isna().any():
                st.warning(
                    f"Regressor '{reg_col}' has missing values after resampling. "
                    f"Forward/back-filling so Prophet can run."
                )
                df[reg_col] = df[reg_col].ffill().bfill()

        # Split train vs future blanks (based on raw y)
        train_df = df[df["y_raw"].notna()].copy()
        missing_future_df = df[df["y_raw"].isna()].copy()

        if train_df.empty:
            st.error("No historical (non-null) metric values found to train on.")
            st.stop()

        # Build model
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
        )

        for reg_col in selected_regressors:
            model.add_regressor(reg_col)

        if manual_changepoints.strip():
            cps = [pd.to_datetime(d.strip()) for d in manual_changepoints.split(",") if d.strip()]
            model.changepoints = cps
            st.write("Using manual changepoints:", cps)

        model.fit(train_df[["ds", "y"] + selected_regressors])

        # Build future
        if not missing_future_df.empty:
            future = missing_future_df[["ds"] + selected_regressors].copy()
            st.info(f"Detected {len(missing_future_df)} blank metric rows. Forecasting those dates from your file.")
        else:
            future = model.make_future_dataframe(periods=int(forecast_periods), freq=freq)

            # Fill regressors into future: event flags -> 0, others -> carry last observed
            for reg_col in selected_regressors:
                if reg_col in event_regressors:
                    future[reg_col] = 0.0
                else:
                    last_val = (
                        train_df[reg_col].dropna().iloc[-1]
                        if reg_col in train_df.columns and train_df[reg_col].notna().any()
                        else 0.0
                    )
                    future[reg_col] = float(last_val)

        forecast = model.predict(future)

        # Convert yhat back to raw scale for display if log used
        pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        pred["ds"] = pd.to_datetime(pred["ds"])

        if use_log_y:
            pred["yhat_raw"] = safe_expm1(pred["yhat"])
            pred["yhat_lower_raw"] = safe_expm1(pred["yhat_lower"])
            pred["yhat_upper_raw"] = safe_expm1(pred["yhat_upper"])
        else:
            pred["yhat_raw"] = pred["yhat"]
            pred["yhat_lower_raw"] = pred["yhat_lower"]
            pred["yhat_upper_raw"] = pred["yhat_upper"]

        # Combine history + predictions for matrix, using raw scale
        hist = df[["ds", "y_raw"]].copy()
        combined = pd.merge(
            hist,
            pred[["ds", "yhat_raw", "yhat_lower_raw", "yhat_upper_raw"]],
            on="ds",
            how="outer"
        ).sort_values("ds")

        combined["final"] = combined["y_raw"].combine_first(combined["yhat_raw"])
        combined["year"] = combined["ds"].dt.year
        combined["month"] = combined["ds"].dt.month

        matrix = combined.pivot_table(index="month", columns="year", values="final", aggfunc="mean").round(2)
        if matrix.shape[1] >= 2:
            prev, curr = matrix.columns[-2], matrix.columns[-1]
            denom = matrix[prev].replace(0, pd.NA)
            matrix["% Change"] = ((matrix[curr] - matrix[prev]) / denom).astype(float).round(4) * 100

        st.write("Forecast Matrix (Monthly by Year):")
        st.dataframe(matrix)

        st.write("Year-on-Year Comparison Line Chart:")
        plot_mat = matrix.drop(columns=["% Change"], errors="ignore")
        yoy = plot_mat.reset_index().melt(id_vars="month", var_name="year", value_name="value")
        fig_yoy = px.line(
            yoy, x="month", y="value", color="year",
            labels={"month": "Month", "value": "Metric"}
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

        st.write("Forecast Plot:")
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.write("Decomposition Plot:")
        fig_decomp = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # -----------------------
        # Backtest (main model; cached)
        # -----------------------
        cv_df = None
        perf_df = None

        if run_backtest:
            st.subheader("Backtest Results (Main model)")

            current_params = bt_params_tuple(
                bt_initial_days, bt_period_days, bt_horizon_days,
                freq,
                selected_regressors, event_regressors,
                changepoint_prior_scale, seasonality_prior_scale, manual_changepoints,
                seasonality_mode, use_log_y
            )

            recompute = (
                st.session_state.bt_cv_df is None
                or st.session_state.bt_perf_df is None
                or st.session_state.bt_params != current_params
            )

            colR1, _ = st.columns([1, 3])
            with colR1:
                if st.button("Recompute main backtest"):
                    recompute = True

            if recompute:
                train_span_days = (train_df["ds"].max() - train_df["ds"].min()).days
                needed = int(bt_initial_days + bt_horizon_days + bt_period_days)
                if train_span_days < needed:
                    st.warning(
                        f"Not enough historical span for these backtest windows. "
                        f"Train span ≈ {train_span_days} days; need at least ≈ {needed} days. "
                        f"Reduce initial/horizon/period."
                    )
                else:
                    with st.spinner("Running main CV..."):
                        initial = f"{int(bt_initial_days)} days"
                        period = f"{int(bt_period_days)} days"
                        horizon = f"{int(bt_horizon_days)} days"

                        cv_df = cross_validation(
                            model,
                            initial=initial,
                            period=period,
                            horizon=horizon,
                            parallel=None,
                        )
                        perf_df = performance_metrics(cv_df)
                        cv_df, perf_df = normalize_backtest_frames(cv_df, perf_df)

                        st.session_state.bt_cv_df = cv_df
                        st.session_state.bt_perf_df = perf_df
                        st.session_state.bt_params = current_params

            cv_df = st.session_state.bt_cv_df
            perf_df = st.session_state.bt_perf_df
            cv_df, perf_df = normalize_backtest_frames(cv_df, perf_df)
            st.session_state.bt_cv_df = cv_df
            st.session_state.bt_perf_df = perf_df

            if cv_df is None or perf_df is None or cv_df.empty or perf_df.empty:
                st.info("Backtest results unavailable.")
            else:
                with st.expander("CV predictions (sample)"):
                    st.dataframe(cv_df.head(50))
                with st.expander("Performance metrics"):
                    st.dataframe(perf_df)
                render_interpretable_backtest_views(cv_df)

        # -----------------------
        # Baseline tests (trend reliability)
        # -----------------------
        if run_baseline_tests:
            st.header("Baseline tests")

            st.caption(
                "These tests treat baseline as the model trend. If baseline is unstable here, it will be unreliable in planning."
            )

            st.subheader("Baseline vs Actual (history)")
            baseline_hist = compute_baseline_history_view(
                model,
                train_df[["ds", "y"] + selected_regressors],
                selected_regressors,
                use_log_y
            )
            st.dataframe(baseline_hist.head(30))

            fig_base = px.line(
                baseline_hist,
                x="ds",
                y=["y_actual", "baseline"],
                labels={"value": "Metric", "variable": ""},
                title="History: Actual vs Baseline (trend)"
            )
            st.plotly_chart(fig_base, use_container_width=True)

            fig_ratio = px.line(
                baseline_hist,
                x="ds",
                y="ratio_actual_to_baseline",
                labels={"ratio_actual_to_baseline": "Actual / Baseline"},
                title="History: Actual divided by Baseline (trend)"
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

            # Baseline-only CV (no regressors)
            if run_backtest:
                st.subheader("Baseline-only backtest (no regressors)")

                base_params = (
                    int(bt_initial_days), int(bt_period_days), int(bt_horizon_days),
                    float(changepoint_prior_scale), float(seasonality_prior_scale),
                    str(seasonality_mode), bool(use_log_y)
                )
                recompute_base = (
                    st.session_state.bt_cv_base_df is None
                    or st.session_state.bt_perf_base_df is None
                    or st.session_state.bt_base_params != base_params
                )

                colB1, _ = st.columns([1, 3])
                with colB1:
                    if st.button("Recompute baseline-only CV"):
                        recompute_base = True

                if recompute_base:
                    with st.spinner("Running baseline-only CV (no regressors)..."):
                        base_model, cv_base, perf_base = run_baseline_only_cv(
                            train_df_model_scale=train_df[["ds", "y"]],
                            cps=changepoint_prior_scale,
                            sps=seasonality_prior_scale,
                            seasonality_mode=seasonality_mode,
                            bt_initial_days=int(bt_initial_days),
                            bt_period_days=int(bt_period_days),
                            bt_horizon_days=int(bt_horizon_days),
                        )
                        st.session_state.bt_cv_base_df = cv_base
                        st.session_state.bt_perf_base_df = perf_base
                        st.session_state.bt_base_params = base_params

                cv_base = st.session_state.bt_cv_base_df
                perf_base = st.session_state.bt_perf_base_df
                cv_base, perf_base = normalize_backtest_frames(cv_base, perf_base)
                st.session_state.bt_cv_base_df = cv_base
                st.session_state.bt_perf_base_df = perf_base

                if cv_base is None or perf_base is None or cv_base.empty or perf_base.empty:
                    st.info("Baseline-only CV unavailable.")
                else:
                    with st.expander("Baseline-only performance metrics"):
                        st.dataframe(perf_base)

            # Baseline stability sweep (trend sensitivity to CPS)
            st.subheader("Baseline stability sweep (trend sensitivity)")
            base = float(changepoint_prior_scale)
            if sweep_points == 3:
                cps_values = sorted({max(0.001, base * 0.5), base, min(1.0, base * 2.0)})
            elif sweep_points == 5:
                cps_values = sorted({
                    max(0.001, base * 0.25),
                    max(0.001, base * 0.5),
                    base,
                    min(1.0, base * 1.5),
                    min(1.0, base * 2.0),
                })
            else:
                cps_values = sorted({
                    max(0.001, base * 0.2),
                    max(0.001, base * 0.4),
                    max(0.001, base * 0.7),
                    base,
                    min(1.0, base * 1.3),
                    min(1.0, base * 1.7),
                    min(1.0, base * 2.2),
                })

            hist_ds = train_df["ds"].sort_values().reset_index(drop=True)
            trend_mat = stability_sweep_trends(
                train_df_model_scale=train_df[["ds", "y"]],
                cps_values=cps_values,
                sps=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                hist_ds=hist_ds
            )
            trend_mean = trend_mat.mean(axis=1)
            trend_std = trend_mat.std(axis=1)
            denom = trend_mean.replace(0, np.nan).abs()
            stability_cv = (trend_std / denom).replace([np.inf, -np.inf], np.nan)
            stability_score = float(np.nanmedian(stability_cv.values))
            st.metric("Baseline stability score (median CV of trend)", f"{stability_score:.4f}")

            # Convert to raw scale if log used
            if use_log_y:
                plot_df = pd.DataFrame({"ds": hist_ds})
                for col in trend_mat.columns:
                    plot_df[f"trend_cps_{col}"] = safe_expm1(trend_mat[col].astype(float).values)
            else:
                plot_df = pd.DataFrame({"ds": hist_ds})
                for col in trend_mat.columns:
                    plot_df[f"trend_cps_{col}"] = trend_mat[col].astype(float).values

            line_cols = [c for c in plot_df.columns if c.startswith("trend_cps_")]
            long_trend = plot_df.melt(id_vars="ds", value_vars=line_cols, var_name="variant", value_name="baseline")
            long_trend["variant"] = long_trend["variant"].str.replace("trend_cps_", "CPS=")

            fig_sweep = px.line(
                long_trend,
                x="ds",
                y="baseline",
                color="variant",
                title="Baseline (trend) across CPS variants (baseline-only models)"
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

        # -----------------------
        # Prepare download ZIP
        # -----------------------
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            csv_combined = io.StringIO()
            combined.to_csv(csv_combined, index=False)
            zf.writestr("combined_data.csv", csv_combined.getvalue())

            csv_matrix = io.StringIO()
            matrix.to_csv(csv_matrix, index=True)
            zf.writestr("forecast_matrix.csv", csv_matrix.getvalue())

            zf.writestr("yoy_comparison.html", to_html(fig_yoy, full_html=True, include_plotlyjs="cdn"))
            zf.writestr("forecast.html", to_html(fig_forecast, full_html=True, include_plotlyjs="cdn"))
            zf.writestr("components.html", to_html(fig_decomp, full_html=True, include_plotlyjs="cdn"))

            if run_backtest and cv_df is not None and perf_df is not None:
                csv_cv = io.StringIO()
                cv_df.to_csv(csv_cv, index=False)
                zf.writestr("backtest_cv_predictions.csv", csv_cv.getvalue())

                csv_perf = io.StringIO()
                perf_df.to_csv(csv_perf, index=False)
                zf.writestr("backtest_performance_metrics.csv", csv_perf.getvalue())

            if run_baseline_tests and run_backtest:
                cv_base = st.session_state.bt_cv_base_df
                perf_base = st.session_state.bt_perf_base_df
                if cv_base is not None and perf_base is not None:
                    csv_cvb = io.StringIO()
                    cv_base.to_csv(csv_cvb, index=False)
                    zf.writestr("baseline_only_backtest_cv_predictions.csv", csv_cvb.getvalue())

                    csv_perfb = io.StringIO()
                    perf_base.to_csv(csv_perfb, index=False)
                    zf.writestr("baseline_only_backtest_performance_metrics.csv", csv_perfb.getvalue())

        buffer.seek(0)
        st.success("Done. Download your results below.")
        st.download_button(
            "Download Results",
            data=buffer,
            file_name="forecast_results.zip",
            mime="application/zip",
        )

except Exception as e:
    st.error(f"Failed to process the uploaded file: {e}")
