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

# Helper function for date validation
def validate_dates(date_series):
    try:
        return pd.to_datetime(date_series, errors="coerce")
    except Exception as e:
        return None

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

            # Remove commas from numbers and validate date format
            data[date_col] = validate_dates(data[date_col])
            data[metric_col] = data[metric_col].replace(",", "", regex=True).astype(float)
            for reg_col in regressor_cols:
                data[reg_col] = data[reg_col].replace(",", "", regex=True).astype(float)

            if data[date_col].isnull().any():
                st.error(f"Invalid dates detected in column '{date_col}'. Ensure all dates are in a valid format.")
            else:
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

                            # Frequency selection
                            st.write("Specify the frequency of your data:")
                            freq = st.selectbox(
                                "Frequency", ["Infer Automatically", "Daily", "Weekly", "Monthly", "Yearly"], index=0
                            )
                            
                            # Infer or validate frequency
                            if freq == "Infer Automatically":
                                inferred_freq = pd.infer_freq(data["ds"])
                                if inferred_freq:
                                    freq = inferred_freq
                                    st.info(f"Inferred frequency: {freq}")
                                else:
                                    freq = "D"  # Default to daily
                                    st.warning(
                                        "Frequency could not be inferred automatically. Defaulting to 'Daily'."
                                    )
                            else:
                                freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
                                freq = freq_map[freq]
                            
                            # Ensure regular intervals if frequency is specified
                            if freq:
                                data = data.set_index("ds").asfreq(freq).reset_index()
                                st.write(f"Data resampled to {freq} frequency:")
                                st.dataframe(data.head())


                            # Fit the model
                            model.fit(data)

                            # Create future dataframe
                            future = model.make_future_dataframe(periods=0)
                            for reg_col in regressor_cols:
                                future[reg_col] = data[reg_col]
                            forecast = model.predict(future)

                            # Separate historical and forecast data
                            historical_data = data[["ds", "y"]]
                            forecast_data = forecast[["ds", "yhat"]]
                            combined_data = pd.merge(historical_data, forecast_data, on="ds", how="outer")
                            combined_data["final"] = combined_data["y"].combine_first(combined_data["yhat"])

                            # Format forecast matrix
                            combined_data["year"] = combined_data["ds"].dt.year
                            combined_data["month"] = combined_data["ds"].dt.month
                            forecast_matrix = combined_data.pivot_table(
                                index="month", columns="year", values="final", aggfunc="mean"
                            ).round(2)

                            # Add percentage change columns for forecast dates
                            if len(forecast_matrix.columns) >= 2:
                                last_year = forecast_matrix.columns[-2]
                                forecast_year = forecast_matrix.columns[-1]
                                forecast_matrix["% Change"] = (
                                    (forecast_matrix[forecast_year] - forecast_matrix[last_year])
                                    / forecast_matrix[last_year]
                                ).round(4) * 100

                            # Display forecast matrix
                            st.write("Forecast Matrix (Monthly by Year):")
                            st.dataframe(forecast_matrix)

                            # Plot year-on-year comparison
                            st.write("Year-on-Year Comparison Line Chart:")
                            yoy_data = forecast_matrix.iloc[:, :-1].reset_index().melt(
                                id_vars="month", var_name="year", value_name="value"
                            )
                            yoy_fig = px.line(
                                yoy_data,
                                x="month",
                                y="value",
                                color="year",
                                title="Year-on-Year Forecast Comparison",
                                labels={"month": "Month", "value": "Metric"},
                            )
                            st.plotly_chart(yoy_fig)

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
                                combined_csv = io.StringIO()
                                combined_data.to_csv(combined_csv, index=False)
                                zf.writestr("combined_data.csv", combined_csv.getvalue())

                                # Save forecast matrix as CSV
                                matrix_csv = io.StringIO()
                                forecast_matrix.to_csv(matrix_csv, index=True)
                                zf.writestr("forecast_matrix.csv", matrix_csv.getvalue())

                                # Save year-on-year comparison plot
                                yoy_buffer = io.BytesIO()
                                write_image(yoy_fig, yoy_buffer, format="png")
                                zf.writestr("yoy_comparison.png", yoy_buffer.getvalue())

                                # Save forecast and decomposition plots as PNG
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
