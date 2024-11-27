import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
import zipfile
import io
import os
import matplotlib.pyplot as plt

# Set Streamlit configuration
st.set_page_config(page_title="Prophet Forecasting App", layout="wide")

# Title
st.title("Forecasting with Prophet")

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head())

    # Ensure required columns
    if len(data.columns) < 2:
        st.error("The file must have at least two columns: 'date' and 'metric'.")
    else:
        date_col = data.columns[0]
        metric_col = data.columns[1]
        regressor_cols = data.columns[2:]
        
        st.info(f"Using '{date_col}' as the date column and '{metric_col}' as the metric column.")
        st.info(f"Found {len(regressor_cols)} regressors: {', '.join(regressor_cols)}")

        # User options for forecast tuning
        changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.0, 1.0, 0.5, 0.1)
        seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0, 0.5)
        manual_changepoints = st.text_area("Manual Changepoints (comma-separated dates, e.g., 2024-01-01,2024-06-01)", "")

        # Prepare data for Prophet
        data = data.rename(columns={date_col: "ds", metric_col: "y"})
        for reg_col in regressor_cols:
            data[reg_col] = data[reg_col]
        
        st.write("Processed Data:", data.head())

        # Run Forecast
        if st.button("Run Forecast"):
            with st.spinner("Running the forecast..."):
                try:
                    # Initialize Prophet model
                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale
                    )

                    # Add regressors
                    for reg_col in regressor_cols:
                        model.add_regressor(reg_col)

                    # Add manual changepoints if provided
                    if manual_changepoints.strip():
                        changepoints = [cp.strip() for cp in manual_changepoints.split(",")]
                        model.changepoints = pd.to_datetime(changepoints)
                        st.write("Using manual changepoints:", changepoints)

                    # Fit the model
                    model.fit(data)

                    # Forecast
                    future = model.make_future_dataframe(periods=0)
                    for reg_col in regressor_cols:
                        future[reg_col] = data[reg_col]
                    forecast = model.predict(future)

                    # Plot results
                    st.write("Forecast:")
                    forecast_fig = plot_plotly(model, forecast)
                    st.plotly_chart(forecast_fig)

                    st.write("Decomposition Plots:")
                    components_fig = plot_components_plotly(model, forecast)
                    st.plotly_chart(components_fig)

                    # Export results
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w") as zf:
                        # Save forecast to CSV
                        forecast_csv = io.StringIO()
                        forecast.to_csv(forecast_csv, index=False)
                        zf.writestr("forecast.csv", forecast_csv.getvalue())

                        # Save plots
                        for plot_name, plot_fig in zip(["forecast", "components"], [forecast_fig, components_fig]):
                            img_buffer = io.BytesIO()
                            plt.figure(plot_fig)
                            plt.savefig(img_buffer, format="png")
                            plt.close()
                            zf.writestr(f"{plot_name}.png", img_buffer.getvalue())
                    
                    st.success("Forecasting complete!")
                    st.download_button("Download Results", buffer.getvalue(), "forecast_results.zip")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
