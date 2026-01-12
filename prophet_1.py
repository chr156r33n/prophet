import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
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
# Sidebar config (all controls live here)
# ----------------------------
st.sidebar.header("Forecast Settings")

# Keep defaults stable across reruns
if "run_forecast" not in st.session_state:
    st.session_state.run_forecast = False

if "run_backtest_now" not in st.session_state:
    st.session_state.run_backtest_now = False

# Put widgets in a form so changing them doesn't immediately "do stuff"
with st.sidebar.form("config_form", clear_on_submit=False):
    st.subheader("Model")
    changepoint_prior_scale = st.slider(
        "Changepoint Prior Scale",
        min_value=0.0, max_value=1.0, value=0.05, step=0.01
    )
    seasonality_prior_scale = st.slider(
        "Seasonality Prior Scale",
        min_value=1.0, max_value=20.0, value=10.0, step=0.5
    )
    manual_changepoints = st.text_area(
        "Manual Changepoints (comma-separated dates, e.g., 2024-01-01,2024-06-01)",
        ""
    )

    st.subheader("Data frequency & horizon")
    freq_choice = st.selectbox(
        "Frequency",
        ["Infer Automatically", "Daily", "Weekly", "Monthly", "Yearly"],
        index=0
    )
    forecast_periods = st.number_input(
        "Future periods to forecast (used only if your file has no blank future metric rows)",
        min_value=0, value=0, step=1
    )

    st.subheader("Backtest (Cross-Validation)")
    run_backtest = st.checkbox("Run backtest (rolling-origin CV)", value=False)

    bt_initial_days = st.number_input("Initial training window (days)", min_value=30, value=365, step=30)
    bt_period_days = st.number_input("Period between cutoffs (days)", min_value=7, value=30, step=7)
    bt_horizon_days = st.number_input("Forecast horizon (days)", min_value=7, value=90, step=7)

    submitted = st.form_submit_button("Apply settings")

# ----------------------------
# File upload (main area)
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file (first col = date, second col = metric, optional regressors after)",
    type=["csv"]
)

# ----------------------------
# Run button in sidebar (does not cause app to "forget" controls)
# ----------------------------
st.sidebar.divider()
run_clicked = st.sidebar.button("Run Forecast", type="primary")
if run_clicked:
    st.session_state.run_forecast = True
else:
    # Only reset if user explicitly changes file or hits a (future) reset button
    pass

# ----------------------------
# Main logic
# ----------------------------
if uploaded_file:
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

        # Coerce metric, allowing blanks
        metric_coerced = coerce_numeric_allow_blanks(data[metric_col])
        non_numeric_examples = find_true_non_numeric_examples(data[metric_col], metric_coerced)
        if non_numeric_examples:
            st.error(
                f"Non-numeric values detected in metric column '{metric_col}'. "
                f"Examples: {', '.join(non_numeric_examples)}"
            )
            st.stop()
        data[metric_col] = metric_coerced

        # Coerce regressors
        for reg_col in regressor_cols:
            reg_coerced = coerce_numeric_allow_blanks(data[reg_col])
            reg_bad = find_true_non_numeric_examples(data[reg_col], reg_coerced)
            if reg_bad:
                st.error(
                    f"Non-numeric values detected in regressor '{reg_col}'. "
                    f"Examples: {', '.join(reg_bad)}"
                )
                st.stop()
            data[reg_col] = reg_coerced

        st.info(f"Date column: '{date_col}', Metric column: '{metric_col}'")
        st.info(f"Regressors: {', '.join(regressor_cols) if regressor_cols else '(none)'}")

        # Prepare Prophet frame
        df = data.rename(columns={date_col: "ds", metric_col: "y"}).copy()
        df = df.sort_values("ds").reset_index(drop=True)

        st.write("Processed Data (pre-resample):", df.head())

        # Only run heavy work when user clicks Run Forecast
        if st.session_state.run_forecast:
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

                # Resample to regular frequency
                df = df.set_index("ds").asfreq(freq).reset_index()

                # Regressors must exist in rows used for predict()
                if regressor_cols:
                    for reg_col in regressor_cols:
                        if reg_col in df.columns and df[reg_col].isna().any():
                            st.warning(
                                f"Regressor '{reg_col}' has missing values after resampling. "
                                f"Forward/back-filling so Prophet can run. (Not real future regressor data.)"
                            )
                            df[reg_col] = df[reg_col].ffill().bfill()

                st.write(f"Data resampled to {freq} frequency:")
                st.dataframe(df.head())

                # Split train vs future-missing metric rows
                train_df = df[df["y"].notna()].copy()
                missing_future_df = df[df["y"].isna()].copy()

                if train_df.empty:
                    st.error("No historical (non-null) metric values found to train on.")
                    st.stop()

                # Initialize model
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

                # Fit
                model.fit(train_df)

                # Build future:
                if not missing_future_df.empty:
                    future = missing_future_df[["ds"] + regressor_cols].copy()
                    st.info(f"Detected {len(missing_future_df)} blank metric rows. Forecasting those dates from your file.")
                else:
                    future = model.make_future_dataframe(periods=int(forecast_periods), freq=freq)
                    for reg_col in regressor_cols:
                        last_val = train_df[reg_col].dropna().iloc[-1] if train_df[reg_col].notna().any() else 0.0
                        future[reg_col] = last_val

                forecast = model.predict(future)

                # Combine history + predictions
                hist = df[["ds", "y"]].copy()
                pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                combined = pd.merge(hist, pred, on="ds", how="outer").sort_values("ds")

                combined["final"] = combined["y"].combine_first(combined["yhat"])
                combined["year"] = combined["ds"].dt.year
                combined["month"] = combined["ds"].dt.month

                matrix = combined.pivot_table(
                    index="month", columns="year", values="final", aggfunc="mean"
                ).round(2)

                if matrix.shape[1] >= 2:
                    prev, curr = matrix.columns[-2], matrix.columns[-1]
                    denom = matrix[prev].replace(0, pd.NA)
                    matrix["% Change"] = ((matrix[curr] - matrix[prev]) / denom).astype(float).round(4) * 100

                st.write("Forecast Matrix (Monthly by Year):")
                st.dataframe(matrix)

                st.write("Year-on-Year Comparison Line Chart:")
                plot_mat = matrix.copy()
                if "% Change" in plot_mat.columns:
                    plot_mat = plot_mat.drop(columns=["% Change"])
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
                # Backtest
                # -----------------------
                cv_df = None
                perf_df = None
                fig_bt = None

                if run_backtest:
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

                        st.subheader("Backtest Results")
                        st.write("Cross-validation predictions (sample):")
                        st.dataframe(cv_df.head(30))

                        st.write("Performance metrics:")
                        st.dataframe(perf_df)

                        metric_choice = st.selectbox(
                            "Backtest metric to plot",
                            [c for c in ["mape", "mae", "rmse", "mdape", "smape", "coverage"] if c in perf_df.columns],
                            index=0
                        )
                        fig_bt = px.line(
                            perf_df,
                            x="horizon",
                            y=metric_choice,
                            labels={"horizon": "Horizon", metric_choice: metric_choice.upper()},
                            title=f"{metric_choice.upper()} vs Horizon"
                        )
                        st.plotly_chart(fig_bt, use_container_width=True)

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

                    if cv_df is not None:
                        csv_cv = io.StringIO()
                        cv_df.to_csv(csv_cv, index=False)
                        zf.writestr("backtest_cv_predictions.csv", csv_cv.getvalue())

                    if perf_df is not None:
                        csv_perf = io.StringIO()
                        perf_df.to_csv(csv_perf, index=False)
                        zf.writestr("backtest_performance_metrics.csv", csv_perf.getvalue())

                    if fig_bt is not None:
                        zf.writestr("backtest_metric_plot.html", to_html(fig_bt, full_html=True, include_plotlyjs="cdn"))

                buffer.seek(0)

                st.success("Done. Download your results below.")
                st.download_button(
                    "Download Results",
                    data=buffer,
                    file_name="forecast_results.zip",
                    mime="application/zip",
                )

            # Optional: auto-reset the run flag so it doesn't re-run heavy compute on every rerun
            st.session_state.run_forecast = False
        else:
            st.info("Configure options in the sidebar, then click **Run Forecast**.")

    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e}")
else:
    st.info("Upload a CSV to begin.")
