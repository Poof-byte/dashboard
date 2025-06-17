import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# --- Configuration ---
st.set_page_config(page_title="Water Quality Index Analysis Dashboard", page_icon="📊", layout="centered")

# Initialize session state for active section if not already set
if 'current_section' not in st.session_state:
    st.session_state.current_section = "Welcome"


# --- Helper Function for Model Performance Display ---
def display_model_performance(model_name, model_data, data_type="Water Quality"):
    """
    Displays the performance metrics table, a bar graph, and a simulated actual vs. predicted graph for a given model.

    Args:
        model_name (str): The name of the model (e.g., "LSTM", "CNN", "Hybrid CNN-LSTM").
        model_data (dict): Dictionary containing "Parameter", "MSE", "MAE", "R²" data.
        data_type (str): Type of data for which performance is displayed (e.g., "Water Quality", "Meteorological", "Volcanic Activity").
    """
    st.subheader(f"📊 Deep Learning Model Performance Metrics - {model_name} ({data_type})")
    st.write(f"Performance metrics for {model_name} models applied to {data_type.lower()} parameter prediction:")

    df_model = pd.DataFrame(model_data)
    st.dataframe(df_model, use_container_width=True)

    st.info(
        "Lower MSE and MAE values indicate higher accuracy, while an R² value closer to 1 (or positive) signifies a better fit of the model to the data.")

    st.markdown("---")

    # --- Bar Graph for Metrics ---
    st.subheader(f"{model_name} Model ({data_type}): Performance Metrics Visualization")

    df_metrics_melted = df_model.melt(id_vars=["Parameter"], var_name="Metric", value_name="Value")

    fig_metrics = px.bar(df_metrics_melted, x="Parameter", y="Value", color="Metric",
                         barmode="group",
                         title=f"{model_name} Model ({data_type}): MSE, MAE, and R² per Parameter",
                         labels={"Value": "Metric Value", "Parameter": f"{data_type} Parameter"})

    st.plotly_chart(fig_metrics, use_container_width=True)
    st.warning(
        "Note: R² values can be negative and may be on a different scale than MSE/MAE. For best visual comparison, consider separate charts for R² or normalizing metrics if scales vary widely.")

    st.markdown("---")

    # --- Simulated Actual vs Predicted Graph ---
    st.subheader(f"{model_name} Model ({data_type}): Simulated Actual vs. Predicted Values")
    st.write("This graph illustrates how well the model's predictions might align with actual observations.")

    # Generate simulated data for Actual vs Predicted
    np.random.seed(hash(model_name + data_type) % (
                2 ** 32 - 1))  # Use model name and data type for reproducible but different seeds
    num_samples = 100
    actual_values = np.random.rand(num_samples) * 10 + 5  # Example range 5-15

    # Adjust noise based on a simplistic R² interpretation for visual demonstration
    r_squared_avg = df_model['R²'].mean()
    if r_squared_avg < 0:  # If R^2 is negative, higher scatter
        noise_scale = 1.0 - r_squared_avg  # Larger negative R^2 means larger noise_scale
    else:  # If R^2 is positive, smaller scatter
        noise_scale = 1.0 - r_squared_avg  # Closer to 1, smaller noise_scale. Min noise at R2=1
    noise_scale = max(0.1, min(5.0, noise_scale * 0.5))  # Clamp to reasonable range for visual effect

    predicted_values = actual_values + np.random.randn(num_samples) * noise_scale
    df_actual_predicted = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values})

    # Determine plot ranges dynamically to fit all points and the perfect prediction line
    min_val_plot = min(df_actual_predicted['Actual'].min(), df_actual_predicted['Predicted'].min())
    max_val_plot = max(df_actual_predicted['Actual'].max(), df_actual_predicted['Predicted'].max())

    try:
        fig_actual_predicted = px.scatter(df_actual_predicted, x="Actual", y="Predicted",
                                          title=f"{model_name} Model ({data_type}): Simulated Actual vs. Predicted Values",
                                          labels={"Actual": "Actual Value", "Predicted": "Predicted Value"},
                                          color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_actual_predicted.add_trace(go.Scatter(x=[min_val_plot, max_val_plot],
                                                  y=[min_val_plot, max_val_plot],
                                                  mode='lines',
                                                  line=dict(color='red', dash='dash'), name='Perfect Prediction'))
        fig_actual_predicted.update_layout(xaxis_range=[min_val_plot, max_val_plot],
                                           yaxis_range=[min_val_plot, max_val_plot],
                                           xaxis_title="Actual Value",
                                           yaxis_title="Predicted Value")
        st.plotly_chart(fig_actual_predicted, use_container_width=True)
        st.info("A perfect model would have all points lie on the red diagonal line.")
    except Exception as e:
        st.error(f"Error loading {model_name} Actual vs. Predicted visualization for {data_type}: {e}")

    st.markdown("---")


# --- WQI Calculation Function ---
def calculate_wqi(row):
    """
    Calculates a simplified Water Quality Index (WQI) for a given row of water quality data.
    Scores are assigned based on predefined ranges for each parameter, with 100 being best.
    """
    scores = {}

    # Average Temperature (assuming all three temp columns are present and numeric)
    # Check if all temperature components are available before averaging
    if pd.notna(row['Surface Water Temp.']) and pd.notna(row['Middle Water Temp.']) and pd.notna(
            row['Bottom Water Temp.']):
        avg_temp = (row['Surface Water Temp.'] + row['Middle Water Temp.'] + row['Bottom Water Temp.']) / 3
        if 20 <= avg_temp <= 30:
            scores['Temperature_Score'] = 100
        elif (15 <= avg_temp < 20) or (30 < avg_temp <= 35):
            scores['Temperature_Score'] = 70
        else:
            scores['Temperature_Score'] = 30
    else:
        scores['Temperature_Score'] = np.nan  # Mark as NaN if any temp data is missing

    # pH Level
    if pd.notna(row['pH Level']):
        if 7.0 <= row['pH Level'] <= 8.5:
            scores['pH_Score'] = 100
        elif (6.5 <= row['pH Level'] < 7.0) or (8.5 < row['pH Level'] <= 9.0):
            scores['pH_Score'] = 70
        else:
            scores['pH_Score'] = 30
    else:
        scores['pH_Score'] = np.nan

    # Ammonia (n)
    if pd.notna(row['Ammonia (n)']):
        if row['Ammonia (n)'] < 0.1:
            scores['Ammonia_Score'] = 100
        elif 0.1 <= row['Ammonia (n)'] <= 0.5:
            scores['Ammonia_Score'] = 70
        else:
            scores['Ammonia_Score'] = 30
    else:
        scores['Ammonia_Score'] = np.nan

    # Nitrate-N
    if pd.notna(row['Nitrate-N']):
        if row['Nitrate-N'] < 1.0:
            scores['Nitrate_Score'] = 100
        elif 1.0 <= row['Nitrate-N'] <= 10.0:
            scores['Nitrate_Score'] = 70
        else:
            scores['Nitrate_Score'] = 30
    else:
        scores['Nitrate_Score'] = np.nan

    # Phosphate
    if pd.notna(row['Phosphate']):
        if row['Phosphate'] < 0.05:
            scores['Phosphate_Score'] = 100
        elif 0.05 <= row['Phosphate'] <= 0.1:
            scores['Phosphate_Score'] = 70
        else:
            scores['Phosphate_Score'] = 30
    else:
        scores['Phosphate_Score'] = np.nan

    # Dissolved Oxygen (DO)
    if pd.notna(row['Dissolved O']):
        if row['Dissolved O'] > 7.0:
            scores['DO_Score'] = 100
        elif 5.0 <= row['Dissolved O'] <= 7.0:
            scores['DO_Score'] = 70
        else:
            scores['DO_Score'] = 30
    else:
        scores['DO_Score'] = np.nan

    # Calculate WQI as the average of available scores. Exclude NaN scores.
    valid_scores = [score for score in scores.values() if pd.notna(score)]
    if valid_scores:
        return np.mean(valid_scores)
    return np.nan  # Return NaN if no valid scores could be calculated


# --- Generic Data Loading and Processing Function ---
def load_and_process_data(csv_string, column_renames=None, numeric_cols=None, missing_value_placeholder=None,
                          calculate_wqi_flag=False):
    """
    Loads data from a CSV string, processes it, and aggregates trends.

    Args:
        csv_string (str): The raw CSV data as a string.
        column_renames (dict, optional): A dictionary to rename columns. Defaults to None.
        numeric_cols (list, optional): List of columns to convert to numeric. Defaults to None.
        missing_value_placeholder (any, optional): Value to replace with NaN before dropping. Defaults to None.
        calculate_wqi_flag (bool): If True, calculates WQI for water quality data.

    Returns:
        tuple: A tuple containing (df, df_weekly, df_monthly, df_yearly, numeric_cols_used).
               Returns empty DataFrames and empty list if an error occurs.
    """
    df = pd.DataFrame()
    df_weekly = pd.DataFrame()
    df_monthly = pd.DataFrame()
    df_yearly = pd.DataFrame()
    numeric_cols_used = []

    try:
        # Explicitly handle common missing value strings, including 'None'
        df = pd.read_csv(io.StringIO(csv_string), sep='\t', na_values=['None', 'NaN', 'null', ''])

        # Create 'Date' column
        df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01', errors='coerce')
        df = df.drop(columns=['YEAR', 'MONTH'])

        # Standardize and rename columns if mapping is provided
        if column_renames:
            df.columns = df.columns.str.strip().str.replace(r' \(°C\)', '', regex=True).str.replace(r' \(MG/L\)', '',
                                                                                                    regex=True)
            df = df.rename(columns=column_renames)

        # Replace placeholder missing values if specified
        if missing_value_placeholder is not None:
            df.replace(missing_value_placeholder, np.nan, inplace=True)

        # Convert specified columns to numeric
        if numeric_cols:
            numeric_cols_used = [col for col in numeric_cols if col in df.columns]
            for col in numeric_cols_used:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where essential numeric columns are NaN, but only AFTER potential WQI calculation if applicable
            # For water quality, we need all params for WQI
            if calculate_wqi_flag:
                # Calculate WQI first on the raw (but type-converted) data
                df['Water Quality Index (WQI)'] = df.apply(calculate_wqi, axis=1)
                numeric_cols_used.append('Water Quality Index (WQI)')  # Add WQI to the list of numeric columns used
                df.dropna(subset=numeric_cols_used, how='any',
                          inplace=True)  # Now drop rows with NaNs in any relevant column including WQI

            else:  # For other data types, just drop NaNs based on the original numeric_cols
                df.dropna(subset=numeric_cols_used, how='any', inplace=True)

        # Aggregate Data for Trends
        # Only aggregate if the DataFrame is not empty after dropping NaNs
        if not df.empty:
            df_weekly = df.groupby(pd.Grouper(key='Date', freq='W'))[numeric_cols_used].mean().reset_index()
            df_weekly['Period'] = df_weekly['Date'].dt.strftime('%Y-%m-%d')
            if len(df_weekly) > 50:  # Limit to latest 50 weekly entries for display efficiency
                df_weekly = df_weekly.tail(50)
            df_weekly = df_weekly.round(2)

            df_monthly = df.groupby(pd.Grouper(key='Date', freq='M'))[numeric_cols_used].mean().reset_index()
            df_monthly['Period'] = df_monthly['Date'].dt.strftime('%Y-%m')
            df_monthly = df_monthly.round(2)

            df_yearly = df.groupby(pd.Grouper(key='Date', freq='Y'))[numeric_cols_used].mean().reset_index()
            df_yearly['Period'] = df_yearly['Date'].dt.strftime('%Y')
            df_yearly = df_yearly.round(2)
        else:
            # If df is empty after dropping NaNs, ensure trend DataFrames are also empty
            df_weekly = pd.DataFrame(columns=['Period'] + numeric_cols_used)
            df_monthly = pd.DataFrame(columns=['Period'] + numeric_cols_used)
            df_yearly = pd.DataFrame(columns=['Period'] + numeric_cols_used)


    except Exception as e:
        st.error(f"Error processing data: {e}")
        # Return empty dataframes on error
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    return df, df_weekly, df_monthly, df_yearly, numeric_cols_used


# --- Raw Data Strings ---
water_quality_csv_data = """YEAR	MONTH	SURFACE WATER TEMP (°C)	MIDDLE WATER TEMP (°C)	BOTTOM WATER TEMP (°C)	PH LEVEL	AMMONIA (MG/L)	NITRATE-N OR NITRITE-N (MG/L)	PHOSPHATE (MG/L)	DISSOLVED OXYGEN (MG/L)
2013	4	30.92	27.4	26.86	8.06	0.02	0.01	NaN	6.22
2013	8	29.67	NaN	NaN	8.5	0.14	0.14	NaN	6.38
2013	9	30.88	NaN	NaN	8.35	0.18	0.01	NaN	5.03
2014	1	25.44	25.21	25.37	8.06	0.48	NaN	NaN	4.48
2014	2	27.04	25.79	25.67	8.21	0.26	0.01	NaN	10.37
2014	3	26.96	26	25.94	8.32	0.31	0.01	NaN	8.03
2014	4	29.71	26.88	26.08	8.05	0.18	0.01	NaN	6.45
2014	5	31.12	26.5	26.29	8.08	0.23	0.02	NaN	7.09
2014	6	30.79	27.3	25.82	8.92	0.37	0.02	NaN	7.6
2014	7	30.91	27.09	26.27	8.75	0.37	0.01	NaN	7.05
2014	11	28.6	27.4	26.83	8.23	0.27	0.12	NaN	5.05
2014	12	28.0	27.78	27.12	8.19	0.3	0.04	NaN	8.3
2015	1	27.79	26.42	26.17	7.82	0.18	0.04	NaN	5.34
2015	2	26.69	26.29	26.08	7.96	0.31	0.13	NaN	8.42
2015	3	27.21	26.45	26.2	8.17	0.13	0.17	NaN	5.28
2015	4	28.75	27.38	26.92	8.44	0.26	0.14	NaN	5.58
2015	5	30.9	28.16	27.31	8.48	0.29	0.05	NaN	6.66
2015	6	32.12	28.44	27.72	8.69	0.23	0.06	NaN	5.93
2015	7	30.77	27.83	28.28	8.69	0	0.01	NaN	5.39
2015	8	30.24	28.52	27.51	8.48	0.38	0	NaN	5.14
2015	9	31.59	28.92	27.93	8.84	0.07	0	NaN	6
2015	10	29.38	28.35	27.62	8.7	0.32	0.07	NaN	5.9
2015	11	28.88	28.09	27.34	7.22	0.28	0.07	NaN	5.48
2015	12	28.57	27.72	27.12	8.4	0.18	0.06	NaN	5.71
2016	1	27.19	26.57	26.27	8.08	0.27	0.05	NaN	4.02
2016	2	25.84	25.82	25.65	8.02	0.19	0.08	NaN	4.94
2016	3	28.12	26.98	26.79	8.52	0.23	0.12	NaN	4.71
2016	4	29.14	27.62	27.22	8.82	0.34	0.14	NaN	7.85
2016	6	30.7	28.15	27.6	8.37	0.2	0	NaN	4.67
2016	8	29.27	28.37	27.69	8.68	0.38	0.01	NaN	6.18
2016	9	30.05	28.52	27.48	8.87	0.07	0	NaN	7.13
2016	10	29.82	28.55	27.62	8.58	0.24	0.07	NaN	4.75
2016	11	28.21	27.43	26.82	7.58	0.35	0.07	NaN	4.78
2016	12	28.08	28.02	27.55	7.45	0.46	0.07	NaN	7.08
2017	1	26.85	26.75	26.78	7.09	0.19	0.14	NaN	5.04
2017	2	26.15	26.16	26.11	7.97	0.27	0.15	NaN	5.03
2017	3	27.23	26.82	26.87	7.89	0.22	0.2	NaN	4.3
2017	4	28.98	27.28	26.98	8.17	0.17	0.17	NaN	6.96
2017	5	30.32	27.98	27.57	8	0.25	0.1	NaN	6.69
2017	6	32.06	28.14	27.51	8.03	0.74	0.16	NaN	5.66
2017	7	31.63	27.76	27.24	8.26	0.42	0.14	NaN	3.6
2017	8	30.28	28.32	27.53	8.64	0.36	0.09	NaN	4.78
2017	9	30.82	29.02	28.22	8.32	0	0.05	NaN	4.78
2017	10	30.53	29.05	28.36	8.56	0.03	0	NaN	3.93
2017	11	29.72	28.48	27.88	8.48	0.41	0.05	NaN	4.76
2017	12	28.95	28.45	28.01	8.32	0.69	0.05	NaN	5.25
2018	1	27.24	26.98	26.9	8.07	0.56	0.12	2.29	3.98
2018	2	27.07	26.88	26.69	7.84	0.43	0.23	2.39	4.08
2018	3	27.99	27.42	27.07	8.17	0.37	0.09	2.44	4.84
2018	4	28.46	27.34	26.97	8.28	0	0.11	2.36	4.97
2018	6	31.49	29.14	28	8.31	0.37	0.09	2.57	4.71
2018	7	30.33	28.42	27.88	8.3	0.28	0.05	2.52	5.26
2018	8	29.82	28.4	27.76	8.34	0.37	0.06	2.46	5.98
2018	9	29.17	28.36	28.01	8.27	0.4	0.07	2.39	6.36
2018	10	30.38	28.3	27.88	8.0	0.35	0.07	2.34	6.39
2018	11	29.24	28.41	27.54	7.89	0.33	0.07	2.49	4.67
2018	12	28.07	27.93	27.5	7.85	0.16	0.13	2.48	4.57
2019	1	26.7	26.61	26.53	8.17	0.16	0.24	2.42	3.57
2019	2	26.72	26.7	26.59	7.82	0.02	0.25	2.36	3.36
2019	3	27.57	26.74	26.53	7.92	0.14	0.19	2.4	5.03
2019	4	29.22	27.62	26.83	8.05	0.18	0.23	2.49	5.32
2019	5	31.87	28.27	27.18	8.14	0.17	0.17	2.55	4.29
2019	6	31.76	28.6	27.11	8.23	0.39	0.11	2.57	3.86
2019	7	31.51	29.11	27.74	8.03	0.28	0.07	2.6	4.83
2019	8	28.79	28.73	27.9	8.18	0.17	0.13	2.55	4.87
2019	9	29.16	28.68	27.46	7.97	0.29	0.02	2.5	3.88
2019	10	29.9	28.89	27.44	8.02	0.29	0.02	2.47	3.94
2019	11	29.37	28.74	27.81	8.03	0.24	0.02	2.29	4.04
2019	12	28.09	27.82	27.68	7.82	0.36	0.18	2.63	3.2
2020	2	26.41	26.4	26.29	7.23	0.15	0.15	2.53	2.73
2020	3	28.98	27.28	26.93	7.53	0.07	0.08	2.49	5.96
2020	4	29.89	28.32	27.2	8.87	0.27	0.14	2.69	5.5
22020	6	31.13	29.2	27.16	8.1	0.19	0.08	2.49	4.74
2020	7	31.81	29.43	27.53	8.16	0.27	0.04	3.0	3.58
2020	9	31.34	29.71	28.21	8.07	0.12	0.03	2.4	5.23
2020	10	31.41	30.11	28.6	8.33	0.21	0.04	2.44	4.94
2020	11	28.36	28.12	27.51	7.91	0.29	0.08	2.56	4.97
2020	12	27.96	27.82	27.44	7.65	0.26	0.07	2.42	5.28
2021	6	31.16	29.29	27.91	8.07	0.17	5.67	2.48	6
2021	7	30.32	28.76	27.8	8.07	0.11	0.2	2.75	4.89
2021	8	29.54	29.02	27.81	8.3	0.32	0.05	2.57	5.71
2021	9	29.6	28.97	28.09	8.21	0.47	0.11	2.68	4.98
2022	1	26.81	26.66	26.6	7.92	0.16	0.24	2.67	3.81
2022	2	27.28	26.9	26.73	8.15	0.17	0.25	2.72	5.28
2022	3	28.59	27.24	27.17	8.14	0.22	0.17	2.6	5.89
2022	6	32.1	29.13	28.13	8.14	0.37	0.16	2.56	5.29
2022	7	31.02	29.23	28.04	8.36	0.41	0.1	2.58	3.86
2022	8	30.77	29.42	28.01	9.16	0.21	0.09	2.34	4.52
2022	9	27.19	29.39	27.9	8.5	0.47	0.12	2.44	5.29
2022	10	28.46	28.32	27.88	8.16	0.54	0.17	2.51	5.03
2022	11	28.87	28.67	27.9	8.14	0.59	0.12	2.46	4.53
2022	12	28.64	28.48	27.94	8.0	0.66	0.14	2.33	5.37
"""

meteorological_csv_data = """YEAR	MONTH	RAINFALL	TMAX	TMIN	RH	WIND_SPEED	WIND_DIRECTION
2013	1	46.2	29.2	22.7	79	2	40
2013	2	122	30.5	23.2	78	2	70
2013	3	20.5	31.9	23.4	77	2	40
2013	4	42.4	34.4	23.8	75	1	40
2013	5	230.3	33.2	24.9	81	1	40
2013	6	386.1	32.4	24.8	85	1	240
2013	7	285.9	31.6	24.2	85	1	40
2013	8	524.7	30.7	24.2	86	2	240
2013	9	815.8	30.1	24.1	89	1	220
2013	10	121.1	30.7	23.2	83	1	40
2013	11	225.2	30.4	24.3	83	2	40
2013	12	49.1	30.4	24	81	2	40
2014	1	0.5	28.4	20.9	74	3	40
2014	2	0	30.7	20.6	76	2	40
2014	3	28.1	32	22.1	77	2	40
2014	4	22.4	34.1	22.8	76	1	40
2014	5	154.6	35	24.5	80	1	40
2014	6	78.9	33.1	24.7	83	1	240
2014	7	406.7	31.1	24.2	87	2	360
2014	8	212.5	31.9	23.8	84	2	220
2014	9	321.5	31.4	23.6	85	2	40
2014	10	223.5	31.4	23.4	86	2	40
2014	11	43	31.3	23.2	81	2	40
2014	12	326.8	29.3	23.6	83	3	40
2015	1	50.8	28.6	21.6	80	2	40
2015	2	5	30.4	21.1	77	2	40
2015	3	15.2	31.7	21.5	75	2	40
2015	4	0	34	23.4	73	2	40
2015	5	11.7	34.7	24.1	76	1	40
2015	6	159.7	34	24.7	81	1	220
2015	7	309.1	32	24.1	83	2	220
2015	8	265.3	32.3	23.9	85	2	220
2015	9	243.8	32.6	24.2	85	1	40
2015	10	230.2	32.2	23.4	84	2	40
2015	11	43.4	31.8	24.3	82	2	40
2015	12	292.5	30.6	23	83	2	40
2016	1	18.7	31	22.3	82	2	40
2016	2	45.3	30.7	22.5	79	3	40
2016	3	10.2	33.5	22.1	79	2	40
2016	4	8.2	35.8	24.2	75	1	40
2016	5	101.4	35.9	25	78	1	40
2016	6	95.6	34.5	24.6	85	1	220
2016	7	225.5	33.7	24.1	86	1	40
2016	8	375.4	31.8	25	86	2	220
2016	9	167	32.6	24.1	86	1	220
2016	10	312.5	32	23.8	87	1	220
2016	11	162.1	31.5	23.5	85	2	40
2016	12	187.3	31	23.8	84	2	40
2017	1	87.1	29.8	22.9	83	3	40
2017	2	20.2	30.4	21.4	80	3	40
2017	3	19.4	32.3	22.4	77	2	40
2017	4	17.8	34.6	23.8	77	2	40
2017	5	126.6	35.1	25	80	1	220
2017	6	115.8	34.8	24.2	83	1	220
2017	7	254	32.3	24.4	85	1	220
2017	8	375	33	24.1	86	2	220
2017	9	585	32.9	24.1	87	1	220
2017	10	241.7	31.9	24	88	1	40
2017	11	189.7	31.9	24.5	85	2	40
2017	12	183.7	30.3	23.6	84	2	40
2018	1	40.1	30	22.8	82	3	40
2018	2	2.4	31.6	22.9	80	2	40
2018	3	4	32.3	22.8	77	2	40
2018	4	6.02	34.6	23.9	75	2	40
2018	5	131.5	34.9	24.6	79	1	70
2018	6	488.4	32.3	24.1	85	1	70
2018	7	596.2	31.2	24.1	87	2	70
2018	8	160.8	31.7	24.6	84	2	230
2018	9	255.6	32	23.8	86	1	220
2018	10	127.7	32.8	23.3	82	1	40
2018	11	20.3	31.9	23.8	81	2	40
2018	12	286.2	30.1	23.7	84	2	40
2019	1	17.3	29.9	22.3	80	3	40
2019	2	9.1	31	20.7	77	2	40
2019	3	0	33.1	21.9	74	2	40
2019	4	0.2	35	23.6	72	2	40
2019	5	139.2	34.7	24.6	82	1	210
2019	6	150.2	34.9	25.3	81	2	40
2019	7	243.9	32.7	24.1	86	2	240
2019	8	323	31.3	24.9	86	2	240
2019	9	234.6	31.6	24.3	87	2	240
2019	10	38.1	32.9	23.5	82	2	20
2019	11	145.2	31.8	23.8	84	2	40
2019	12	194.1	30.7	23.5	83	2	40
2020	2	28.2	30.3	22.3	78	3	40
2020	3	4.4	33.5	23.1	77	2	40
2020	4	3	34.4	24.4	74	2	40
2020	5	114	35.3	25.1	79	2	40
2020	6	124.6	34	24.7	83	1	220
2020	7	255	33.3	24.2	86	1	40
2020	8	173.9	32.5	24.1	86	1	40
2020	9	101.7	33.1	24.3	86	1	220
2020	10	481.3	31.5	24.1	87	2	40
2020	11	424	31.1	23.7	84	2	40
2020	12	149.9	30.6	24.1	85	2	40
2021	1	51.6	29.5	23.4	82	3	40
2021	2	35.1	30.5	22.6	81	2	40
2021	3	43.2	32.5	23.7	80	2	40
2021	4	11.4	33.4	23	77	2	40
2021	5	56.6	35	25.1	76	2	40
2021	6	174.6	33.1	24.3	84	2	360
2021	7	632	31.5	24.9	85	3	220
2021	8	168.4	31.9	24.5	84	2	360
2021	9	295.7	31.9	23.9	85	2	360
2021	10	161	31.4	24.5	83	2	360
2021	11	51.3	31.8	24.3	80	2	40
2021	12	91.4	29.7	23.2	78	3	40
2022	1	15.8	30.1	22.5	78	3	40
2022	2	35.4	30.4	23.5	78	3	40
2022	3	40.8	33.3	24	75	2	40
2022	4	117.6	32.1	24.1	78	2	40
2022	5	284	33.6	24	79	1	360
2022	6	253.5	32.9	24.1	83	1	360
2022	7	199.7	32.4	24.4	84	1	360
2022	8	275.5	31.6	23.9	86	1	360
2022	9	325	31.5	24.1	85	2	360
2022	10	475.7	31.6	24.3	84	2	360
2022	11	79	31.6	24.4	82	2	40
2022	12	116.5	29.8	23.5	82	2	40
2023	1	203.3	28.6	23.2	80	3	40
2023	2	22.7	30.2	23.1	77	3	40
2023	3	0.8	31.8	22.5	70	2	40
2023	4	101.4	33.3	23.7	75	1	40
2023	5	89.8	33.8	25.1	78	2	40
2023	6	252.9	32.8	24.6	82	1	220
2023	7	394.1	32.3	24.7	82	2	220
2023	8	143.7	32.2	25.5	81	3	240
2023	9	163.4	32.5	24.6	84	2	240
2023	10	89.9	31.5	24.5	82	2	40
2023	11	166.6	31.2	24.5	81	2	40
2023	12	127.9	31	24	81	2	40
"""

so2_flux_csv_data = """YEAR	MONTH	SO2_Flux
2020	1	1281.93
2020	2	92.89
2020	9	50
2021	3	865.41
2021	4	1558.68
2021	5	2961.57
2021	6	4631.54
2021	7	6148.63
2021	8	5973.09
2021	9	7785.37
2021	10	9550.62
2021	11	8704.39
2021	12	7130.75
2022	1	10213.23
2022	2	5973.67
2022	3	9833.5
2022	4	3167.5
2022	5	996.27
2022	6	1715.71
2022	7	3018.67
2022	8	5230.65
2022	9	4949.89
2022	10	3213
2022	11	1603.25
2022	12	2981.78
2023	1	5307.12
2023	2	4345.33
2023	3	3279.83
2023	4	2941.62
2023	5	3555.38
2023	6	5611.08
2023	7	4913.11
2023	8	2628
2023	9	2991.4
2023	10	6146.55
2023	11	8144
2023	12	8832.88
2024	1	11845.11
2024	2	9449.9
2024	3	9988.77
2024	4	5521.2
2024	5	4360.33
2024	6	4845.38
2024	7	4259.86
2024	8	4153.82
2024	9	4447
2024	10	1803.2
2024	11	5283
2024	12	3469.75
"""

co2_flux_csv_data = """YEAR	MONTH	CO2_FLUX
2013	3	870
2013	7	1700
2013	10	540
2014	1	500
2014	2	680
2014	6	700
2014	7	1800
2014	11	2290
2015	1	1980
2015	4	1580
2015	7	2300
2015	10	1550
2016	1	1380
2016	7	465
2017	1	460
2017	4	390
2017	7	598
2017	11	446
2018	3	984
2018	8	251.11
2019	3	311
2019	8	1302.57
2019	10	1147.15
2019	11	1364.36
"""

# --- Load and Process Data for Each Category ---

# Water Quality Data
water_quality_renames = {
    'SURFACE WATER TEMP': 'Surface Water Temp.',
    'MIDDLE WATER TEMP': 'Middle Water Temp.',
    'BOTTOM WATER TEMP': 'Bottom Water Temp.',
    'PH LEVEL': 'pH Level',
    'AMMONIA': 'Ammonia (n)',
    'NITRATE-N OR NITRITE-N': 'Nitrate-N',
    'PHOSPHATE': 'Phosphate',
    'DISSOLVED OXYGEN': 'Dissolved O'
}
numeric_cols_water_quality = [
    'Surface Water Temp.', 'Middle Water Temp.', 'Bottom Water Temp.',
    'pH Level', 'Ammonia (n)', 'Nitrate-N', 'Phosphate', 'Dissolved O'
]
df_water_quality_raw, df_water_quality_weekly, df_water_quality_monthly, df_water_quality_yearly, numeric_cols_water_quality_trends = \
    load_and_process_data(water_quality_csv_data, water_quality_renames, numeric_cols_water_quality,
                          calculate_wqi_flag=True)

# Meteorological Data
meteorological_renames = None  # No special renaming needed beyond stripping
numeric_cols_meteorological = [
    'RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION'
]
df_meteorological_raw, df_meteorological_weekly, df_meteorological_monthly, df_meteorological_yearly, numeric_cols_meteorological_trends = \
    load_and_process_data(meteorological_csv_data, meteorological_renames, numeric_cols_meteorological,
                          missing_value_placeholder=-999)

# Volcanic Activity Data
# Load SO2 and CO2 separately
df_so2_raw, _, _, _, _ = load_and_process_data(so2_flux_csv_data, numeric_cols=['SO2_Flux'])
df_co2_raw, _, _, _, _ = load_and_process_data(co2_flux_csv_data, numeric_cols=['CO2_FLUX'])

# Merge them on 'Date' to combine for trends
# Use an outer merge to keep all dates present in either SO2 or CO2 data
df_volcanic_activity_merged = pd.merge(df_so2_raw[['Date', 'SO2_Flux']], df_co2_raw[['Date', 'CO2_FLUX']], on='Date',
                                       how='outer')

# Re-process the merged dataframe for trends
volcanic_activity_renames = None  # No renames as columns are already fine
numeric_cols_volcanic_activity_trends = ['SO2_Flux', 'CO2_FLUX']  # Both columns are now available

# Apply aggregation on the merged dataframe
# Check if df_volcanic_activity_merged is not empty before grouping
if not df_volcanic_activity_merged.empty and numeric_cols_volcanic_activity_trends:
    df_volcanic_activity_weekly = df_volcanic_activity_merged.groupby(pd.Grouper(key='Date', freq='W'))[
        numeric_cols_volcanic_activity_trends].mean().reset_index()
    df_volcanic_activity_weekly['Period'] = df_volcanic_activity_weekly['Date'].dt.strftime('%Y-%m-%d')
    if len(df_volcanic_activity_weekly) > 50:
        df_volcanic_activity_weekly = df_volcanic_activity_weekly.tail(50)
    df_volcanic_activity_weekly = df_volcanic_activity_weekly.round(2)

    df_volcanic_activity_monthly = df_volcanic_activity_merged.groupby(pd.Grouper(key='Date', freq='M'))[
        numeric_cols_volcanic_activity_trends].mean().reset_index()
    df_volcanic_activity_monthly['Period'] = df_volcanic_activity_monthly['Date'].dt.strftime('%Y-%m')
    df_volcanic_activity_monthly = df_volcanic_activity_monthly.round(2)

    df_volcanic_activity_yearly = df_volcanic_activity_merged.groupby(pd.Grouper(key='Date', freq='Y'))[
        numeric_cols_volcanic_activity_trends].mean().reset_index()
    df_volcanic_activity_yearly['Period'] = df_volcanic_activity_yearly['Date'].dt.strftime('%Y')
    df_volcanic_activity_yearly = df_volcanic_activity_yearly.round(2)
else:
    # If merged dataframe is empty or no numeric cols, initialize empty trend dataframes
    df_volcanic_activity_weekly = pd.DataFrame(columns=['Period'] + numeric_cols_volcanic_activity_trends)
    df_volcanic_activity_monthly = pd.DataFrame(columns=['Period'] + numeric_cols_volcanic_activity_trends)
    df_volcanic_activity_yearly = pd.DataFrame(columns=['Period'] + numeric_cols_volcanic_activity_trends)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")

# Add a menu button for the Introduction/Welcome Screen
if st.sidebar.button("Introduction"):
    st.session_state.current_section = "Welcome"

# Section for Model Metric Performances
st.sidebar.subheader("Model Metric Performances")

# Buttons for each model's performance
if st.sidebar.button("LSTM"):
    st.session_state.current_section = "LSTM Performance"
if st.sidebar.button("CNN"):
    st.session_state.current_section = "CNN Performance"
if st.sidebar.button("Hybrid CNN-LSTM"):
    st.session_state.current_section = "Hybrid CNN-LSTM Performance"

st.sidebar.markdown("---")  # Separator

# New section for Data Trends
st.sidebar.subheader("Data Trends")

trend_options = ['Select Period', 'Weekly', 'Monthly', 'Yearly']

# Water Quality Trends Dropdown
water_quality_trend_selection = st.sidebar.selectbox(
    "Water Quality Trends:",
    options=trend_options,
    key="sidebar_water_quality_trends"
)
if water_quality_trend_selection != "Select Period":
    st.session_state.current_section = f"Water Quality Trends - {water_quality_trend_selection}"

# Meteorological Trends Dropdown
meteorological_trend_selection = st.sidebar.selectbox(
    "Meteorological Trends:",
    options=trend_options,
    key="sidebar_meteorological_trends"
)
if meteorological_trend_selection != "Select Period":
    st.session_state.current_section = f"Meteorological Trends - {meteorological_trend_selection}"

# Volcanic Activity Trends Dropdown
volcanic_activity_trend_selection = st.sidebar.selectbox(
    "Volcanic Activity Trends:",
    options=trend_options,
    key="sidebar_volcanic_activity_trends"
)
if volcanic_activity_trend_selection != "Select Period":
    st.session_state.current_section = f"Volcanic Activity Trends - {volcanic_activity_trend_selection}"

# --- Main Content Area ---
st.title("📊 Water Quality Index Analysis Dashboard")
st.markdown("---")

# Conditional rendering based on current_section in session state
if st.session_state.current_section == "Welcome":
    st.write(
        "Welcome to the Water Quality Index Analysis Dashboard. This interactive tool provides insights into water quality parameters, meteorological data, and volcanic activity metrics. You can also explore the performance of different deep learning models used for prediction.")
    st.info(
        "Use the sidebar to navigate through different sections, including model performance metrics and various data trends (weekly, monthly, yearly).")

elif st.session_state.current_section == "LSTM Performance":
    st.header("Deep Learning Model Performance - LSTM")
    lstm_water_quality_data = {
        "Parameter": ["Ammonia", "Bottom Water Temp.", "Middle Water Temp.", "Surface Water Temp.", "Dissolved Oxygen",
                      "Nitrate", "pH Level", "Phosphate"],
        "MSE": [0.11830, 0.65115, 1.00949, 0.69685, 0.06093, 2.1903, 0.19347, 0.22463],
        "MAE": [0.09872, 0.53507, 0.84357, 0.42207, 0.04968, 1.56651, 0.15204, 0.15202],
        "R²": [-2.76204, -0.96678, -0.24864, -0.24373, -8.87357, -0.74374, 0.29938, 0.0033]
    }
    display_model_performance("LSTM", lstm_water_quality_data, "Water Quality")

    lstm_meteorological_data = {
        "Parameter": ["Rainfall", "Relative Humidity", "Max Temperature", "Minimum Temperature", "Wind Direction",
                      "Wind Speed"],
        "MSE": [7482.5500, 37.8890, 6.8140, 7.8374, 21489.9642, 50.2397],
        "MAE": [65.36984, 4.17260, 2.12671, 1.72942, 126.47095, 5.14132],
        "R²": [0.47031, -1.87164, -3.38781, -15.88326, -0.07401, -116.55547]
    }
    display_model_performance("LSTM", lstm_meteorological_data, "Meteorological")

    lstm_volcanic_activity_data = {
        "Parameter": ["CO₂", "SO₂"],
        "MSE": [387161.5542, 1618533.3276],
        "MAE": [588.08010, 899.45384],
        "R²": [-1.16058, -0.65138]
    }
    display_model_performance("LSTM", lstm_volcanic_activity_data, "Volcanic Activity")


elif st.session_state.current_section == "CNN Performance":
    st.header("Deep Learning Model Performance - CNN")
    cnn_water_quality_data = {
        "Parameter": ["Ammonia", "Bottom Water Temp.", "Middle Water Temp.", "Surface Water Temp.", "Dissolved Oxygen",
                      "Nitrate", "pH Level", "Phosphate"],
        "MSE": [1.39594, 1.40864, 1.03885, 0.73711, 0.73711, 3.81135, 2.49556, 0.65751],
        "MAE": [1.35772, 1.32216, 1.02245, 0.54551, 4.86078, 3.37753, 2.48291, 0.61166],
        "R²": [-2.83056, -8.2045, -0.32234, -0.39157, -6.3013, -4.27999, -115.57361, -7.53973]
    }
    display_model_performance("CNN", cnn_water_quality_data, "Water Quality")

    cnn_meteorological_data = {
        "Parameter": ["Rainfall", "Relative Humidity", "Max Temperature", "Minimum Temperature", "Wind Direction",
                      "  Wind Speed"],
        "MSE": [7761.7486, 10.5074, 13.3732, 0.8315, 16884.6820, 4.2438],
        "MAE": [67.31469, 2.42539, 3.02555, 0.63255, 107.33447, 1.90034],
        "R²": [0.45055, 0.20353, -7.61236, -0.79087, 0.15545, -8.92233]
    }
    display_model_performance("CNN", cnn_meteorological_data, "Meteorological")


elif st.session_state.current_section == "Hybrid CNN-LSTM Performance":
    st.header("Deep Learning Model Performance - Hybrid CNN-LSTM")
    hybrid_water_quality_data = {
        "Parameter": ["Ammonia", "Bottom Water Temp.", "Middle Water Temp.", "Surface Water Temp.", "Dissolved Oxygen",
                      "Nitrate", "pH Level", "Phosphate"],
        "MSE": [0.10341, 1.03128, 0.77751, 0.60096, 0.02999, 1.43801, 0.18122, 0.20981],
        "MAE": [0.09339, 0.92348, 0.63248, 0.41008, 0.02339, 1.41256, 0.14368, 0.1575],
        "R²": [-1.87446, -3.93352, 0.25929, 0.07502, -1.39146, 0.24838, 0.38527, 0.13048]
    }
    display_model_performance("Hybrid CNN-LSTM", hybrid_water_quality_data, "Water Quality")

    hybrid_meteorological_data = {
        "Parameter": ["Rainfall", "Relative Humidity", "Max Temperature", "Minimum Temperature", "Wind Direction",
                      "  Wind Speed"],
        "MSE": [6134.6893, 7.5743, 3.1053, 0.6286, 15458.7536, 0.8482],
        "MAE": [57.50735, 2.25265, 1.38521, 0.59754, 96.45390, 0.72400],
        "R²": [0.56573, 0.42606, -1.00029, -0.35387, 0.22742, -0.98397]
    }
    display_model_performance("Hybrid CNN-LSTM", hybrid_meteorological_data, "Meteorological")


# New Trend Sections based on sidebar dropdown selections
elif st.session_state.current_section == "Water Quality Trends - Weekly":
    st.header("📈 Water Quality Trends - Weekly")
    if not df_water_quality_weekly.empty:
        st.write(
            "Displaying weekly averages for water quality parameters, including the calculated Water Quality Index (WQI):")
        st.dataframe(df_water_quality_weekly[['Period'] + numeric_cols_water_quality_trends], use_container_width=True)

        # Plot all water quality trends on a single graph
        fig = px.line(df_water_quality_weekly, x='Period', y=numeric_cols_water_quality_trends,
                      title='Weekly Trends of Water Quality Parameters',
                      labels={'Period': 'Week', 'value': 'Value'})
        fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The Water Quality Index (WQI) is a single number that reflects the overall water quality. A higher WQI generally indicates better water quality.")
    else:
        st.info(
            "No weekly water quality trend data available. Please ensure the CSV data is properly formatted and contains no NaN values in relevant columns.")

elif st.session_state.current_section == "Water Quality Trends - Monthly":
    st.header("📈 Water Quality Trends - Monthly")
    if not df_water_quality_monthly.empty:
        st.write(
            "Displaying monthly averages for water quality parameters, including the calculated Water Quality Index (WQI):")
        st.dataframe(df_water_quality_monthly[['Period'] + numeric_cols_water_quality_trends], use_container_width=True)

        # Plot all water quality trends on a single graph
        fig = px.line(df_water_quality_monthly, x='Period', y=numeric_cols_water_quality_trends,
                      title='Monthly Trends of Water Quality Parameters',
                      labels={'Period': 'Month', 'value': 'Value'})
        fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The Water Quality Index (WQI) is a single number that reflects the overall water quality. A higher WQI generally indicates better water quality.")
    else:
        st.info(
            "No monthly water quality trend data available. Please ensure the CSV data is properly formatted and contains no NaN values in relevant columns.")

elif st.session_state.current_section == "Water Quality Trends - Yearly":
    st.header("📈 Water Quality Trends - Yearly")
    if not df_water_quality_yearly.empty:
        st.write(
            "Displaying yearly averages for water quality parameters, including the calculated Water Quality Index (WQI):")
        st.dataframe(df_water_quality_yearly[['Period'] + numeric_cols_water_quality_trends], use_container_width=True)

        # Plot all water quality trends on a single graph
        fig = px.line(df_water_quality_yearly, x='Period', y=numeric_cols_water_quality_trends,
                      title='Yearly Trends of Water Quality Parameters',
                      labels={'Period': 'Year', 'value': 'Value'})
        fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The Water Quality Index (WQI) is a single number that reflects the overall water quality. A higher WQI generally indicates better water quality.")
    else:
        st.info(
            "No yearly water quality trend data available. Please ensure the CSV data is properly formatted and contains no NaN values in relevant columns.")

elif st.session_state.current_section.startswith("Meteorological Trends - "):
    period = st.session_state.current_section.split(" - ")[1]
    st.header(f"☁️ Meteorological Trends - {period}")
    st.write(f"Displaying {period.lower()} averages for meteorological parameters:")

    if period == "Weekly":
        if not df_meteorological_weekly.empty:
            st.dataframe(df_meteorological_weekly[['Period'] + numeric_cols_meteorological_trends],
                         use_container_width=True)
            # Plot all meteorological trends on a single graph
            fig = px.line(df_meteorological_weekly, x='Period', y=numeric_cols_meteorological_trends,
                          title='Weekly Trends of Meteorological Parameters',
                          labels={'Period': 'Week', 'value': 'Value'})
            fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No weekly meteorological trend data available. Please ensure the CSV data is properly formatted and contains no -999 or NaN values in relevant columns.")
    elif period == "Monthly":
        if not df_meteorological_monthly.empty:
            st.dataframe(df_meteorological_monthly[['Period'] + numeric_cols_meteorological_trends],
                         use_container_width=True)
            # Plot all meteorological trends on a single graph
            fig = px.line(df_meteorological_monthly, x='Period', y=numeric_cols_meteorological_trends,
                          title='Monthly Trends of Meteorological Parameters',
                          labels={'Period': 'Month', 'value': 'Value'})
            fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No monthly meteorological trend data available. Please ensure the CSV data is properly formatted and contains no -999 or NaN values in relevant columns.")
    elif period == "Yearly":
        if not df_meteorological_yearly.empty:
            st.dataframe(df_meteorological_yearly[['Period'] + numeric_cols_meteorological_trends],
                         use_container_width=True)
            # Plot all meteorological trends on a single graph
            fig = px.line(df_meteorological_yearly, x='Period', y=numeric_cols_meteorological_trends,
                          title='Yearly Trends of Meteorological Parameters',
                          labels={'Period': 'Year', 'value': 'Value'})
            fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No yearly meteorological trend data available. Please ensure the CSV data is properly formatted and contains no -999 or NaN values in relevant columns.")

elif st.session_state.current_section.startswith("Volcanic Activity Trends - "):
    period = st.session_state.current_section.split(" - ")[1]
    st.header(f"🌋 Volcanic Activity Trends - {period}")
    st.write(f"Displaying {period.lower()} averages for volcanic activity parameters (SO2 Flux and CO2 Flux):")

    if period == "Weekly":
        if not df_volcanic_activity_weekly.empty:
            st.dataframe(df_volcanic_activity_weekly[['Period'] + numeric_cols_volcanic_activity_trends],
                         use_container_width=True)
            # Plot all volcanic activity trends on a single graph
            fig = px.line(df_volcanic_activity_weekly, x='Period', y=numeric_cols_volcanic_activity_trends,
                          title='Weekly Trends of Volcanic Activity Parameters',
                          labels={'Period': 'Week', 'value': 'Value'})
            fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No weekly volcanic activity trend data available. Please ensure the CSV data is properly formatted and contains no NaN values in relevant columns.")
    elif period == "Monthly":
        if not df_volcanic_activity_monthly.empty:
            st.dataframe(df_volcanic_activity_monthly[['Period'] + numeric_cols_volcanic_activity_trends],
                         use_container_width=True)
            # Plot all volcanic activity trends on a single graph
            fig = px.line(df_volcanic_activity_monthly, x='Period', y=numeric_cols_volcanic_activity_trends,
                          title='Monthly Trends of Volcanic Activity Parameters',
                          labels={'Period': 'Month', 'value': 'Value'})
            fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No monthly volcanic activity trend data available. Please ensure the CSV data is properly formatted and contains no NaN values in relevant columns.")
    elif period == "Yearly":
        if not df_volcanic_activity_yearly.empty:
            st.dataframe(df_volcanic_activity_yearly[['Period'] + numeric_cols_volcanic_activity_trends],
                         use_container_width=True)
            # Plot all volcanic activity trends on a single graph
            fig = px.line(df_volcanic_activity_yearly, x='Period', y=numeric_cols_volcanic_activity_trends,
                          title='Yearly Trends of Volcanic Activity Parameters',
                          labels={'Period': 'Year', 'value': 'Value'})
            fig.update_layout(xaxis_type='category')  # Ensure x-axis is treated as categories
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No yearly volcanic activity trend data available. Please ensure the CSV data is properly formatted and contains no NaN values in relevant columns.")

# Default Welcome section
else:
    st.write(
        "Welcome to the Water Quality Index Analysis Dashboard. Use the sidebar to navigate through different sections.")
    st.info(
        "Select a model from the 'Model Metric Performances' section or a trend period from the 'Data Trends' section in the sidebar.")
