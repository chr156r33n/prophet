import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
from plotly.io import write_image
import zipfile
import io

# Streamlit app configuration
st.set_page_config(page_title="Prophet Forecasting App", layout="wide")

# Title
st.title("Prophet Forecasting App")

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file (with 'date', 'metric', and optional regressors)", type=["csv"])

if uploaded_file:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())

        # Ensure correct structure
        if len(data.columns) < 2:
            st.error("The file must have at least two columns: 'date' and 'metric'.")
        else:
            # Identify columns
            date_col = data.columns[0]
            metric_col = data.columns[1]
            regressor_cols = data.columns[2:]

            st.info(f"Date column: '{date_col}', Metric column: '{metric_col}'")
            st.info(f"Regressors: {', '.join(regressor_cols)} (if any)")

            # User input for forecast options
            changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.0, 1.0, 0.5, 0.1)
            seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0, 0.5)
            manual_changepoints = st.text_area(
                "Manual Changepoints (comma-separated dates, e.g., 2024-01-01,2024-06-01)", ""
            )

            # Prepare data
            data = data.rename(columns={date_col: "ds", metric_col: "y"})
            for reg_col in regressor_cols:
                data[reg_col] = data[reg_col]

            st.write("Processed Data for Forecasting:", data.head())

            # Run Forecast Button
            if st.button("Run Forecast"):
                with st.spinner("Running the forecast..."):
                    try:
                        # Initialize Prophet model
                        model = Prophet(
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale,
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

                        # Create future dataframe
                        future = model.make_future_dataframe(periods=0)
                        for reg_col in regressor_cols:
                            future[reg_col] = data[reg_col]
                        forecast = model.predict(future)

                        # Plot forecast
                        st.write("Forecast Plot:")
                        forecast_fig = plot_plotly(model, forecast)
                        st.plotly_chart(forecast_fig)

                        # Plot components
                        st.write("Decomposition Plot:")
                        components_fig = plot_components_plotly(model, forecast)
                        st.plotly_chart(components_fig)

                        # Prepare download
                        st.write("Preparing download...")
                        buffer = io.BytesIO()
                        with zipfile.ZipFile(buffer, "w") as zf:
                            # Save forecast as CSV
                            forecast_csv = io.StringIO()
                            forecast.to_csv(forecast_csv, index=False)
                            zf.writestr("forecast.csv", forecast_csv.getvalue())

                            # Save plots as PNG
                            for plot_name, plot_fig in zip(
                                ["forecast", "components"], [forecast_fig, components_fig]
                            ):
                                img_buffer = io.BytesIO()
                                write_image(plot_fig, img_buffer, format="png")
                                zf.writestr(f"{plot_name}.png", img_buffer.getvalue())

                        buffer.seek(0)
                        st.success("Forecasting complete! Download your results below.")
                        st.download_button(
                            "Download Results",
                            data=buffer,
                            file_name="forecast_results.zip",
                            mime="application/zip",
                        )
                    except Exception as e:
                        st.error(f"An error occurred during the forecast: {e}")
    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e}")
