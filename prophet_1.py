import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
from plotly.io import to_html
import zipfile
import io
import plotly.express as px

# Streamlit app configuration
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
    • **Significance:** Ensures that your dates parse correctly and that numeric columns contain real numbers.  
    • **Impact:** Invalid dates or stray commas can lead to empty/NaN values that skew model training and break visualizations.  

    **Step 3. Configure forecasting options**  
    • **Changepoint Prior Scale:** Controls how sensitive the model is to trend shifts.  
      - **Low value** → smoother trend, fewer sudden changes captured.  
      - **High value** → model can overfit noise, seeing too many “breaks.”  
    • **Seasonality Prior Scale:** Governs how strongly the model enforces seasonal patterns.  
      - **Low value** → seasonality is muted, good for noisy data.  
      - **High value** → seasonality dominates, may over-emphasize repeating cycles.  
    • **Manual Changepoints:** Let you force in dates where you know the trend shifted (e.g., promotions, pandemics).  

    **Step 4. Specify data frequency**  
    • **Significance:** Prophet assumes evenly spaced data.  
    • **Impact:** Misaligned or irregular intervals lead to misleading forecasts.  

    **Step 5. Run the forecast**  
    • Forecast + decomposition + YoY matrix.  

    **Step 6. (Optional) Backtest**  
    • Runs rolling-origin cross-validation to score the model out-of-sample (stops “tuning until it looks right”).  
    """)

uploaded_file = st.file_uploader(
    "Upload your CSV file (with 'date', 'metric', and optional regressors)",
    type=["csv"]
)

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())

        if len(data.columns) < 2:
            st.error("The file must have at least two columns: 'date' and 'metric'.")
            st.stop()

        date_col = data.columns[0]
        metric_col = data.columns[1]
        regressor_cols = list(data.columns[2:])

        # Parse + clean
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data[metric_col] = _safe_numeric(data[metric_col])
        for reg_col in regressor_cols:
            data[reg_col] = _safe_numeric(data[reg_col])

        if data[date_col].isnull().any():
            st.error(f"Invalid dates detected in column '{date_col}'. Ensure all dates are valid.")
            st.stop()

        if data[metric_col].isnull().any():
            st.error(f"Non-numeric values detected in metric column '{metric_col}'. Fix those first.")
            st.stop()

        st.info(f"Date column: '{date_col}', Metric column: '{metric_col}'")
        st.info(f"Regressors: {', '.join(regressor_cols) if regressor_cols else '(none)'}")

        # Model controls
        changepoint_prior_scale = st.slider(
            "Changepoint Prior Scale", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        seasonality_prior_scale = st.slider(
            "Seasonality Prior Scale", 1.0, 20.0, 10.0, 0.5
        )
        manual_changepoints = st.text_area(
            "Manual Changepoints (comma-separated dates, e.g., 2024-01-01,2024-06-01)",
            ""
        )

        # Frequency selection (IMPORTANT: outside the button so it doesn't reset on rerun)
        freq_choice = st.selectbox(
            "Frequency",
            ["Infer Automatically", "Daily", "Weekly", "Monthly", "Yearly"],
            index=0
        )

        # Backtest controls
        st.subheader("Backtest (Cross-Validation)")
        run_backtest = st.checkbox("Run backtest (rolling-origin CV)", value=False)
        colA, colB, colC = st.columns(3)
        with colA:
            bt_initial_days = st.number_input("Initial training window (days)", min_value=30, value=365, step=30)
        with colB:
            bt_period_days = st.number_input("Period between cutoffs (days)", min_value=7, value=30, step=7)
        with colC:
            bt_horizon_days = st.number_input("Forecast horizon (days)", min_value=7, value=90, step=7)

        # Prepare data for Prophet
        df = data.rename(columns={date_col: "ds", metric_col: "y"}).copy()
        df = df.sort_values("ds")

        st.write("Processed Data for Forecasting:", df.head())

        if st.button("Run Forecast"):
            with st.spinner("Running the forecast..."):

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

                # Resample to regular frequency (Prophet likes evenly spaced data)
                df = df.set_index("ds").asfreq(freq).reset_index()

                # Handle missing values introduced by resampling:
                # - y: keep as NaN (Prophet can handle gaps in y)
                # - regressors: MUST be filled (Prophet cannot forecast with NaN regressors)
                for reg_col in regressor_cols:
                    if reg_col in df.columns:
                        df[reg_col] = df[reg_col].ffill().bfill()

                st.write(f"Data resampled to {freq} frequency:")
                st.dataframe(df.head())

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

                # Fit model
                model.fit(df)

                # In-sample prediction (periods=0 like your original)
                future = model.make_future_dataframe(periods=0, freq=freq)
                for reg_col in regressor_cols:
                    future[reg_col] = df[reg_col].values

                forecast = model.predict(future)

                # Combine + YoY matrix
                hist = df[["ds", "y"]]
                pred = forecast[["ds", "yhat"]]
                combined = pd.merge(hist, pred, on="ds", how="outer")
                combined["final"] = combined["y"].combine_first(combined["yhat"])
                combined["year"] = combined["ds"].dt.year
                combined["month"] = combined["ds"].dt.month

                matrix = combined.pivot_table(
                    index="month", columns="year", values="final", aggfunc="mean"
                ).round(2)
                if matrix.shape[1] >= 2:
                    prev, curr = matrix.columns[-2], matrix.columns[-1]
                    matrix["% Change"] = ((matrix[curr] - matrix[prev]) / matrix[prev]).round(4) * 100

                st.write("Forecast Matrix (Monthly by Year):")
                st.dataframe(matrix)

                st.write("Year-on-Year Comparison Line Chart:")
                yoy = matrix.iloc[:, :-1].reset_index() if "% Change" in matrix.columns else matrix.reset_index()
                yoy = yoy.melt(id_vars="month", var_name="year", value_name="value")
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
                # Backtest (Cross-Validation)
                # -----------------------
                cv_df = None
                perf_df = None
                fig_bt = None

                if run_backtest:
                    # Prophet CV expects strings like "365 days"
                    initial = f"{int(bt_initial_days)} days"
                    period = f"{int(bt_period_days)} days"
                    horizon = f"{int(bt_horizon_days)} days"

                    # Basic feasibility check
                    total_days = (df["ds"].max() - df["ds"].min()).days
                    if total_days < (bt_initial_days + bt_horizon_days + bt_period_days):
                        st.warning(
                            "Not enough history for the chosen backtest windows. "
                            "Reduce initial/horizon/period or upload more data."
                        )
                    else:
                        with st.spinner("Running backtest cross-validation (this can be slow)..."):
                            # cross_validation refits internally. Keep parallel None for Streamlit sanity.
                            cv_df = cross_validation(
                                model,
                                initial=initial,
                                period=period,
                                horizon=horizon
                            )
                            perf_df = performance_metrics(cv_df)

                        st.subheader("Backtest Results")
                        st.write("Cross-validation predictions (sample):")
                        st.dataframe(cv_df.head(25))

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
                    # CSVs
                    csv_all = io.StringIO()
                    combined.to_csv(csv_all, index=False)
                    zf.writestr("combined_data.csv", csv_all.getvalue())

                    csv_mat = io.StringIO()
                    matrix.to_csv(csv_mat, index=True)
                    zf.writestr("forecast_matrix.csv", csv_mat.getvalue())

                    # Charts (HTML)
                    zf.writestr("yoy_comparison.html", to_html(fig_yoy, full_html=True, include_plotlyjs="cdn"))
                    zf.writestr("forecast.html", to_html(fig_forecast, full_html=True, include_plotlyjs="cdn"))
                    zf.writestr("components.html", to_html(fig_decomp, full_html=True, include_plotlyjs="cdn"))

                    # Backtest exports (if run)
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

                st.success("Forecasting complete! Download your results below.")
                st.download_button(
                    "Download Results",
                    data=buffer,
                    file_name="forecast_results.zip",
                    mime="application/zip",
                )

    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e}")
