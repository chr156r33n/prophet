import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
from plotly.io import write_image
import zipfile
import io
import plotly.express as px

# Streamlit app configuration
st.set_page_config(page_title="Prophet Forecasting App", layout="wide")

# Title
st.title("Prophet Forecasting App")
st.markdown("This App takes Meta's Prophet Forecasting model and lets you apply it to SEO data to establish forecasts based on historic data. By (Chris Green)[https://www.chris-green.net/]")

with st.expander("How to use this app"):
    st.markdown("""
    **Step 1. Upload your CSV file**  
    • **Significance:** The quality of your input data drives the accuracy of any forecast.  
    • **Impact:** Missing dates or misformatted numbers here will cascade into bad model fits or errors later.  

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
    • **Impact:** Misaligned or irregular intervals lead to misleading forecasts—e.g., inferring weekly seasonality on daily data will be wrong.  
    • **Inference vs. Manual:** Auto-inference works most of the time, but if your data has gaps or custom intervals, pick your own.  

    **Step 5. Run the forecast**  
    • **Significance:** This triggers model training and prediction.  
    • **Impact:** You’ll see:  
      1. **Forecast Plot:** Visual check that your model “makes sense.”  
      2. **Decomposition Plot:** Breaks down trend, weekly & yearly seasonality—useful for diagnosing over/underfitting.  
      3. **Year-on-Year Matrix:** Numerical comparison by month/year, helping you spot anomalies or growth rates.  

    **Step 6. Download your results**  
    • **Significance:** Packages your raw forecasts, summary matrices, and plots for sharing or deeper analysis.  
    • **Impact:** Keeps everyone on the same page—no copy-paste errors, and you have a ZIP of exactly what you saw.  
    """)

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file (with 'date', 'metric', and optional regressors)", type=["csv"])

if uploaded_file:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())

        # Validate date column
        if len(data.columns) < 2:
            st.error("The file must have at least two columns: 'date' and 'metric'.")
        else:
            date_col = data.columns[0]
            metric_col = data.columns[1]
            regressor_cols = data.columns[2:]

            # Remove commas and parse dates
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            data[metric_col] = data[metric_col].replace(",", "", regex=True).astype(float)
            for reg_col in regressor_cols:
                data[reg_col] = data[reg_col].replace(",", "", regex=True).astype(float)

            if data[date_col].isnull().any():
                st.error(f"Invalid dates detected in column '{date_col}'. Ensure all dates are valid.")
            else:
                st.info(f"Date column: '{date_col}', Metric column: '{metric_col}'")
                st.info(f"Regressors: {', '.join(regressor_cols)} (if any)")

                # Forecast configuration
                changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.0, 1.0, 0.5, 0.1)
                seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0, 0.5)
                manual_changepoints = st.text_area(
                    "Manual Changepoints (comma-separated dates, e.g., 2024-01-01,2024-06-01)", ""
                )

                # Prepare data for Prophet
                data = data.rename(columns={date_col: "ds", metric_col: "y"})
                st.write("Processed Data for Forecasting:", data.head())

                if st.button("Run Forecast"):
                    with st.spinner("Running the forecast..."):
                        # Initialize model
                        model = Prophet(
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale,
                        )
                        for reg_col in regressor_cols:
                            model.add_regressor(reg_col)
                        if manual_changepoints.strip():
                            cps = [pd.to_datetime(d.strip()) for d in manual_changepoints.split(",")]
                            model.changepoints = cps
                            st.write("Using manual changepoints:", cps)

                        # Frequency selection
                        freq = st.selectbox(
                            "Frequency", ["Infer Automatically", "Daily", "Weekly", "Monthly", "Yearly"], index=0
                        )
                        if freq == "Infer Automatically":
                            inferred = pd.infer_freq(data["ds"])
                            if inferred:
                                freq = inferred
                                st.info(f"Inferred frequency: {freq}")
                            else:
                                freq = "D"
                                st.warning("Could not infer frequency. Defaulting to 'Daily'.")
                        else:
                            freq_map = {"Daily":"D","Weekly":"W","Monthly":"M","Yearly":"Y"}
                            freq = freq_map[freq]
                        data = data.set_index("ds").asfreq(freq).reset_index()
                        st.write(f"Data resampled to {freq} frequency:")
                        st.dataframe(data.head())

                        # Fit & predict
                        model.fit(data)
                        future = model.make_future_dataframe(periods=0)
                        for reg_col in regressor_cols:
                            future[reg_col] = data[reg_col]
                        forecast = model.predict(future)

                        # Combine and format
                        hist = data[["ds","y"]]
                        pred = forecast[["ds","yhat"]]
                        combined = pd.merge(hist,pred,on="ds",how="outer")
                        combined["final"] = combined["y"].combine_first(combined["yhat"])
                        combined["year"] = combined["ds"].dt.year
                        combined["month"] = combined["ds"].dt.month
                        matrix = combined.pivot_table(
                            index="month", columns="year", values="final", aggfunc="mean"
                        ).round(2)
                        if matrix.shape[1] >= 2:
                            prev, curr = matrix.columns[-2], matrix.columns[-1]
                            matrix["% Change"] = ((matrix[curr]-matrix[prev]) / matrix[prev]).round(4)*100

                        # Display results
                        st.write("Forecast Matrix (Monthly by Year):")
                        st.dataframe(matrix)
                        st.write("Year-on-Year Comparison Line Chart:")
                        yoy = matrix.iloc[:,:-1].reset_index().melt(
                            id_vars="month", var_name="year", value_name="value"
                        )
                        fig_yoy = px.line(yoy, x="month", y="value", color="year",
                                          labels={"month":"Month","value":"Metric"})
                        st.plotly_chart(fig_yoy)

                        st.write("Forecast Plot:")
                        fig_forecast = plot_plotly(model, forecast)
                        st.plotly_chart(fig_forecast)

                        st.write("Decomposition Plot:")
                        fig_decomp = plot_components_plotly(model, forecast)
                        st.plotly_chart(fig_decomp)

                        # Prepare download ZIP
                        buffer = io.BytesIO()
                        with zipfile.ZipFile(buffer, "w") as zf:
                            csv_all = io.StringIO(); combined.to_csv(csv_all, index=False)
                            zf.writestr("combined_data.csv", csv_all.getvalue())
                            csv_mat = io.StringIO(); matrix.to_csv(csv_mat, index=True)
                            zf.writestr("forecast_matrix.csv", csv_mat.getvalue())
                            b1 = io.BytesIO(); write_image(fig_yoy, b1, format="png")
                            zf.writestr("yoy_comparison.png", b1.getvalue())
                            for name, fig in [("forecast", fig_forecast), ("components", fig_decomp)]:
                                b2 = io.BytesIO(); write_image(fig, b2, format="png")
                                zf.writestr(f"{name}.png", b2.getvalue())
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
