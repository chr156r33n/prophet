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

# ----------------------------
# Helpers
# ----------------------------
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
    """Prophet CV uses Timedelta for horizon; convert to numeric days."""
    if hasattr(td_series, "dt") and np.issubdtype(td_series.dtype, np.timedelta64):
        return td_series.dt.total_seconds() / 86400.0
    return td_series

def normalize_backtest_frames(cv_df: pd.DataFrame | None, perf_df: pd.DataFrame | None):
    """
    Ensure both cv_df and perf_df have a real 'horizon' column.
    Prophet/prophet-diagnostics can return 'horizon' as an index depending on version.
    """
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

def bt_params_tuple(bt_initial_days, bt_period_days, bt_horizon_days, freq, regressor_cols,
                    changepoint_prior_scale, seasonality_prior_scale, manual_changepoints):
    return (
        int(bt_initial_days),
        int(bt_period_days),
        int(bt_horizon_days),
        str(freq),
        tuple(regressor_cols),
        float(changepoint_prior_scale),
        float(seasonality_prior_scale),
        str(manual_changepoints).strip(),
    )

def render_interpretable_backtest_views(cv_df: pd.DataFrame):
    st.subheader("Backtest: Interpretable Views")

    cv = cv_df.copy()
    if "horizon" not in cv.columns:
        st.warning("Backtest output is missing 'horizon' even after normalization. Skipping interpretable views.")
        return

    # Standard Prophet CV cols: ds, y, yhat, yhat_lower, yhat_upper, cutoff, horizon
    needed_cols = {"ds", "y", "yhat"}
    if not needed_cols.issubset(set(cv.columns)):
        st.warning(f"Backtest output is missing required columns: {needed_cols - set(cv.columns)}")
        return

    # Make sure dates are datetime
    cv["ds"] = pd.to_datetime(cv["ds"], errors="coerce")
    if "cutoff" in cv.columns:
        cv["cutoff"] = pd.to_datetime(cv["cutoff"], errors="coerce")

    # Errors
    cv["abs_err"] = (cv["y"] - cv["yhat"]).abs()
    cv["err"] = (cv["y"] - cv["yhat"])

    # Intervals if present
    if "yhat_lower" in cv.columns and "yhat_upper" in cv.columns:
        cv["in_80"] = (cv["y"] >= cv["yhat_lower"]) & (cv["y"] <= cv["yhat_upper"])
    else:
        cv["in_80"] = np.nan

    # APE
    cv["ape"] = (cv["abs_err"] / cv["y"].replace(0, np.nan)).astype(float)

    # Horizon days
    try:
        cv["horizon_days"] = ensure_timedelta_days(cv["horizon"])
    except Exception:
        # If horizon is not timedelta, attempt coercion
        cv["horizon_days"] = pd.to_numeric(cv["horizon"], errors="coerce")

    cv["month"] = cv["ds"].dt.month
    cv["year"] = cv["ds"].dt.year

    # 1) Scorecard
    mape = float(cv["ape"].dropna().mean() * 100) if cv["ape"].notna().any() else np.nan
    median_ape = float(cv["ape"].dropna().median() * 100) if cv["ape"].notna().any() else np.nan
    mae = float(cv["abs_err"].mean()) if cv["abs_err"].notna().any() else np.nan
    median_abs = float(cv["abs_err"].median()) if cv["abs_err"].notna().any() else np.nan

    coverage = float(cv["in_80"].dropna().mean() * 100) if cv["in_80"].notna().any() else np.nan
    within_10 = float((cv["ape"] <= 0.10).mean() * 100) if cv["ape"].notna().any() else np.nan
    within_20 = float((cv["ape"] <= 0.20).mean() * 100) if cv["ape"].notna().any() else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("MAPE (avg)", f"{mape:.1f}%" if np.isfinite(mape) else "n/a")
    c2.metric("APE (median)", f"{median_ape:.1f}%" if np.isfinite(median_ape) else "n/a")
    c3.metric("MAE (avg)", f"{mae:,.2f}" if np.isfinite(mae) else "n/a")
    c4.metric("Abs Err (median)", f"{median_abs:,.2f}" if np.isfinite(median_abs) else "n/a")
    c5.metric("Within ±10%", f"{within_10:.1f}%" if np.isfinite(within_10) else "n/a")
    c6.metric("80% band coverage", f"{coverage:.1f}%" if np.isfinite(coverage) else "n/a")

    # 2) Actual vs predicted
    with st.expander("Actual vs Predicted (CV predictions)", expanded=True):
        cv_plot = cv.sort_values("ds")
        fig_cv = px.line(
            cv_plot,
            x="ds",
            y=["y", "yhat"],
            labels={"value": "Metric", "variable": ""},
            title="CV: Actual vs Predicted (y vs yhat)",
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        if "yhat_lower" in cv_plot.columns and "yhat_upper" in cv_plot.columns:
            fig_bounds = px.line(
                cv_plot,
                x="ds",
                y=["yhat_lower", "yhat_upper"],
                labels={"value": "Metric", "variable": ""},
                title="CV: Prediction interval bounds (lower/upper)",
            )
            st.plotly_chart(fig_bounds, use_container_width=True)

    # 3) Error over time
    with st.expander("Error over time (absolute and % error)", expanded=True):
        fig_abs = px.line(
            cv.sort_values("ds"),
            x="ds",
            y="abs_err",
            labels={"abs_err": "Absolute error"},
            title="CV: Absolute error over time",
        )
        st.plotly_chart(fig_abs, use_container_width=True)

        cv_pct = cv.dropna(subset=["ape"]).copy()
        if not cv_pct.empty:
            cv_pct["ape_pct"] = cv_pct["ape"] * 100
            fig_ape = px.line(
                cv_pct.sort_values("ds"),
                x="ds",
                y="ape_pct",
                labels={"ape_pct": "APE (%)"},
                title="CV: Percent error (APE) over time",
            )
            st.plotly_chart(fig_ape, use_container_width=True)

    # 4) Horizon buckets
    with st.expander("Error by horizon bucket", expanded=True):
        bins = [-np.inf, 7, 14, 30, 60, 90, 120, 180, np.inf]
        labels = ["≤7d", "8–14d", "15–30d", "31–60d", "61–90d", "91–120d", "121–180d", "180d+"]
        cv["h_bucket"] = pd.cut(cv["horizon_days"], bins=bins, labels=labels)

        bucket = cv.groupby("h_bucket", dropna=False).agg(
            n=("y", "size"),
            mape=("ape", lambda s: float(s.dropna().mean() * 100) if s.notna().any() else np.nan),
            median_ape=("ape", lambda s: float(s.dropna().median() * 100) if s.notna().any() else np.nan),
            mae=("abs_err", "mean"),
            median_abs=("abs_err", "median"),
        ).reset_index()

        if "in_80" in cv.columns and cv["in_80"].notna().any():
            cov = cv.groupby("h_bucket", dropna=False)["in_80"].mean().reset_index(name="coverage")
            cov["coverage"] = cov["coverage"] * 100
            bucket = bucket.merge(cov, on="h_bucket", how="left")
        else:
            bucket["coverage"] = np.nan

        st.dataframe(bucket)

        bucket_plot = bucket.dropna(subset=["mape"]).copy()
        if not bucket_plot.empty:
            fig_bucket = px.bar(
                bucket_plot,
                x="h_bucket",
                y="mape",
                labels={"h_bucket": "Horizon bucket", "mape": "MAPE (%)"},
                title="CV: Average % error (MAPE) by horizon bucket",
            )
            st.plotly_chart(fig_bucket, use_container_width=True)

    # 5) Heatmap
    with st.expander("Heatmap: where it fails (month x horizon bucket)", expanded=True):
        if "h_bucket" not in cv.columns:
            bins = [-np.inf, 7, 14, 30, 60, 90, 120, 180, np.inf]
            labels = ["≤7d", "8–14d", "15–30d", "31–60d", "61–90d", "91–120d", "121–180d", "180d+"]
            cv["h_bucket"] = pd.cut(cv["horizon_days"], bins=bins, labels=labels)

        heat = cv.groupby(["month", "h_bucket"], dropna=False)["ape"].mean().reset_index()
        heat["mape_pct"] = heat["ape"] * 100

        fig_heat = px.density_heatmap(
            heat.dropna(subset=["h_bucket"]),
            x="h_bucket",
            y="month",
            z="mape_pct",
            histfunc="avg",
            labels={"h_bucket": "Horizon bucket", "month": "Month", "mape_pct": "MAPE (%)"},
            title="CV: Avg % error by month-of-year and horizon bucket",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # 6) Tolerance bands
    with st.expander("Accuracy bands: % of forecasts within ±10% / ±20%", expanded=True):
        if "h_bucket" not in cv.columns:
            bins = [-np.inf, 7, 14, 30, 60, 90, 120, 180, np.inf]
            labels = ["≤7d", "8–14d", "15–30d", "31–60d", "61–90d", "91–120d", "121–180d", "180d+"]
            cv["h_bucket"] = pd.cut(cv["horizon_days"], bins=bins, labels=labels)

        acc = cv.groupby("h_bucket", dropna=False).agg(
            n=("y", "size"),
            within_10=("ape", lambda s: float((s <= 0.10).mean() * 100) if s.notna().any() else np.nan),
            within_20=("ape", lambda s: float((s <= 0.20).mean() * 100) if s.notna().any() else np.nan),
        ).reset_index()

        st.dataframe(acc)

        acc_long = acc.melt(
            id_vars=["h_bucket", "n"],
            value_vars=["within_10", "within_20"],
            var_name="band",
            value_name="pct",
        )
        acc_long["band"] = acc_long["band"].map({"within_10": "Within ±10%", "within_20": "Within ±20%"})

        fig_acc = px.line(
            acc_long.dropna(subset=["pct"]),
            x="h_bucket",
            y="pct",
            color="band",
            markers=True,
            labels={"h_bucket": "Horizon bucket", "pct": "% of forecasts"},
            title="CV: Share of forecasts within tolerance bands",
        )
        st.plotly_chart(fig_acc, use_container_width=True)


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

    **Step 2. Preview and validate your data**  
    • Ensures dates parse correctly and numeric columns are numeric.  
    • Blanks in the metric column are allowed (often future periods).  

    **Step 3. Configure forecasting options**  
    • Changepoint Prior Scale: trend flexibility  
    • Seasonality Prior Scale: seasonality strength  
    • Manual Changepoints: force known shift dates  

    **Step 4. Specify data frequency**  
    • Prophet works best with evenly spaced data (resampling helps).  

    **Step 5. Run the forecast**  
    • Forecast plot, components, monthly YoY matrix  

    **Step 6. (Optional) Backtest**  
    • Rolling-origin cross-validation to score out-of-sample performance
    """)

# ----------------------------
# Sidebar config
# ----------------------------
st.sidebar.header("Forecast Settings")

# Session state init
for k, v in {
    "run_forecast": False,
    "bt_cv_df": None,
    "bt_perf_df": None,
    "bt_params": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar.form("config_form", clear_on_submit=False):
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
    run_backtest = st.checkbox("Run backtest (rolling-origin CV)", value=False)
    bt_initial_days = st.number_input("Initial training window (days)", min_value=30, value=365, step=30)
    bt_period_days = st.number_input("Period between cutoffs (days)", min_value=7, value=30, step=7)
    bt_horizon_days = st.number_input("Forecast horizon (days)", min_value=7, value=90, step=7)

    st.form_submit_button("Apply settings")

st.sidebar.divider()
if st.sidebar.button("Run Forecast", type="primary"):
    st.session_state.run_forecast = True

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file (first col = date, second col = metric, optional regressors after)",
    type=["csv"]
)

if not uploaded_file:
    st.info("Upload a CSV to begin.")
    st.stop()

# ----------------------------
# Main logic
# ----------------------------
try:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    if len(data.columns) < 2:
        st.error("The file must have at least two columns: date + metric.")
        st.stop()

    date_col = data.columns[0]
    metric_col = data.columns[1]
    regressor_cols = list(data.columns[2:])

    # Parse date
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if data[date_col].isna().any():
        st.error(f"Invalid dates detected in column '{date_col}'.")
        st.stop()

    # Coerce metric
    metric_coerced = coerce_numeric_allow_blanks(data[metric_col])
    bad_metric = find_true_non_numeric_examples(data[metric_col], metric_coerced)
    if bad_metric:
        st.error(f"Non-numeric values detected in metric column '{metric_col}'. Examples: {', '.join(bad_metric)}")
        st.stop()
    data[metric_col] = metric_coerced

    # Coerce regressors
    for reg_col in regressor_cols:
        reg_coerced = coerce_numeric_allow_blanks(data[reg_col])
        bad_reg = find_true_non_numeric_examples(data[reg_col], reg_coerced)
        if bad_reg:
            st.error(f"Non-numeric values detected in regressor '{reg_col}'. Examples: {', '.join(bad_reg)}")
            st.stop()
        data[reg_col] = reg_coerced

    st.info(f"Date column: '{date_col}', Metric column: '{metric_col}'")
    st.info(f"Regressors: {', '.join(regressor_cols) if regressor_cols else '(none)'}")

    df = data.rename(columns={date_col: "ds", metric_col: "y"}).copy()
    df = df.sort_values("ds").reset_index(drop=True)

    st.write("Processed Data (pre-resample):", df.head())

    if not st.session_state.run_forecast:
        st.info("Configure options in the sidebar, then click **Run Forecast**.")
        st.stop()

    with st.spinner("Running forecast..."):
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

        # Resample
        df = df.set_index("ds").asfreq(freq).reset_index()

        # Fill regressors after resampling
        if regressor_cols:
            for reg_col in regressor_cols:
                if reg_col in df.columns and df[reg_col].isna().any():
                    st.warning(
                        f"Regressor '{reg_col}' has missing values after resampling. "
                        f"Forward/back-filling so Prophet can run."
                    )
                    df[reg_col] = df[reg_col].ffill().bfill()

        st.write(f"Data resampled to {freq} frequency:")
        st.dataframe(df.head())

        # Split train vs future blanks
        train_df = df[df["y"].notna()].copy()
        missing_future_df = df[df["y"].isna()].copy()

        if train_df.empty:
            st.error("No historical (non-null) metric values found to train on.")
            st.stop()

        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )

        for reg_col in regressor_cols:
            model.add_regressor(reg_col)

        if manual_changepoints.strip():
            cps = [pd.to_datetime(d.strip()) for d in manual_changepoints.split(",") if d.strip()]
            model.changepoints = cps
            st.write("Using manual changepoints:", cps)

        model.fit(train_df)

        # Future
        if not missing_future_df.empty:
            future = missing_future_df[["ds"] + regressor_cols].copy()
            st.info(f"Detected {len(missing_future_df)} blank metric rows. Forecasting those dates from your file.")
        else:
            future = model.make_future_dataframe(periods=int(forecast_periods), freq=freq)
            for reg_col in regressor_cols:
                last_val = train_df[reg_col].dropna().iloc[-1] if train_df[reg_col].notna().any() else 0.0
                future[reg_col] = last_val

        forecast = model.predict(future)

        # Combine
        hist = df[["ds", "y"]].copy()
        pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        combined = pd.merge(hist, pred, on="ds", how="outer").sort_values("ds")

        combined["final"] = combined["y"].combine_first(combined["yhat"])
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
        fig_yoy = px.line(yoy, x="month", y="value", color="year",
                          labels={"month": "Month", "value": "Metric"})
        st.plotly_chart(fig_yoy, use_container_width=True)

        st.write("Forecast Plot:")
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.write("Decomposition Plot:")
        fig_decomp = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # -----------------------
        # Backtest (cached + normalized)
        # -----------------------
        cv_df = None
        perf_df = None

        if run_backtest:
            st.subheader("Backtest Results")

            current_params = bt_params_tuple(
                bt_initial_days, bt_period_days, bt_horizon_days,
                freq, regressor_cols,
                changepoint_prior_scale, seasonality_prior_scale, manual_changepoints
            )

            recompute = (
                st.session_state.bt_cv_df is None
                or st.session_state.bt_perf_df is None
                or st.session_state.bt_params != current_params
            )

            colR1, _ = st.columns([1, 3])
            with colR1:
                if st.button("Recompute backtest"):
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
                    with st.spinner("Running backtest cross-validation (this can be slow)..."):
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

                        # Normalize 'horizon' across versions
                        cv_df, perf_df = normalize_backtest_frames(cv_df, perf_df)

                        st.session_state.bt_cv_df = cv_df
                        st.session_state.bt_perf_df = perf_df
                        st.session_state.bt_params = current_params

            # Load from cache (and normalize again just in case)
            cv_df = st.session_state.bt_cv_df
            perf_df = st.session_state.bt_perf_df
            cv_df, perf_df = normalize_backtest_frames(cv_df, perf_df)
            st.session_state.bt_cv_df = cv_df
            st.session_state.bt_perf_df = perf_df

            if cv_df is None or perf_df is None or cv_df.empty or perf_df.empty:
                st.info("Backtest results are not available yet (not enough data or computation skipped).")
            else:
                with st.expander("Cross-validation predictions (sample)"):
                    st.dataframe(cv_df.head(50))

                with st.expander("Performance metrics table"):
                    st.dataframe(perf_df)

                # Metric charts (no dropdown selector)
                metric_candidates = ["mape", "mae", "rmse", "mdape", "smape", "coverage"]
                available_metrics = [m for m in metric_candidates if m in perf_df.columns]
                if available_metrics:
                    st.markdown("**Metric trends vs horizon**")
                    tabs = st.tabs([m.upper() for m in available_metrics])
                    for tab, m in zip(tabs, available_metrics):
                        with tab:
                            fig = px.line(
                                perf_df, x="horizon", y=m,
                                labels={"horizon": "Horizon", m: m.upper()},
                                title=f"{m.upper()} vs Horizon"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # Interpretable views
                render_interpretable_backtest_views(cv_df)

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

            if run_backtest and cv_df is not None:
                csv_cv = io.StringIO()
                cv_df.to_csv(csv_cv, index=False)
                zf.writestr("backtest_cv_predictions.csv", csv_cv.getvalue())

            if run_backtest and perf_df is not None:
                csv_perf = io.StringIO()
                perf_df.to_csv(csv_perf, index=False)
                zf.writestr("backtest_performance_metrics.csv", csv_perf.getvalue())

        buffer.seek(0)
        st.success("Done. Download your results below.")
        st.download_button(
            "Download Results",
            data=buffer,
            file_name="forecast_results.zip",
            mime="application/zip",
        )

    # Don’t rerun the whole forecast on every widget rerun
    st.session_state.run_forecast = False

except Exception as e:
    st.error(f"Failed to process the uploaded file: {e}")
