import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Streamlit configuration with custom theme
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")

# Apply custom theme styling
st.markdown("""
    <style>
        html, body, [class*='css'] {
            background-color: #fffafa;
            color: #020310;
        }
        .stButton>button {
            background-color: #1a2ad2 !important;
            color: #fffafa !important;
        }
        .stDataFrame, .stMarkdown, .stSelectbox, .stMetric {
            color: #020310 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Timestamp
st.title(" Environmental Prediction Dashboard")
st.markdown(f"**Date & Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# --- Input Variables for Each Section ---

# Water Quality Parameters and Simulated Data
water_quality_params = ["Ammonia", "Bottom Water Temp.", "Middle Water Temp.", "Surface Water Temp.",
                        "Dissolved Oxygen", "Nitrate", "pH Level", "Phosphate"]

# Simulated Water Quality True & Predicted Values
y_true_wq = np.random.rand(len(water_quality_params)) * 10
y_pred_lstm_wq = y_true_wq + np.random.normal(0, 0.5, len(water_quality_params))
y_pred_cnn_wq = y_true_wq + np.random.normal(0, 0.7, len(water_quality_params))
y_pred_hybrid_wq = y_true_wq + np.random.normal(0, 0.3, len(water_quality_params))

# Water Quality Metrics Data (Manually Input)
wq_lstm_metrics = {
    "Parameter": water_quality_params,
    "MSE": [0.11830, 0.65115, 1.00949, 0.69685, 0.06093, 2.1903, 0.19347, 0.22463],
    "MAE": [0.09872, 0.53507, 0.84357, 0.42207, 0.04968, 1.56651, 0.15204, 0.15202],
    "R²": [-2.76204, -0.96678, -0.24864, -0.24373, -8.87357, -0.74374, 0.29938, 0.0033]
}
wq_cnn_metrics = {
    "Parameter": water_quality_params,
    "MSE": [1.39594, 1.40864, 1.03885, 0.73711, 0.73711, 3.81135, 2.49556, 0.65751],
    "MAE": [1.35772, 1.32216, 1.02245, 0.54551, 4.86078, 3.37753, 2.48291, 0.61166],
    "R²": [-2.83056, -8.2045, -0.32234, -0.39157, -6.3013, -4.27999, -115.57361, -7.53973]
}
wq_hybrid_metrics = {
    "Parameter": water_quality_params,
    "MSE": [0.10341, 1.03128, 0.77751, 0.60096, 0.02999, 1.43801, 0.18122, 0.20981],
    "MAE": [0.09339, 0.92348, 0.63248, 0.41008, 0.02339, 1.41256, 0.14368, 0.1575],
    "R²": [-1.87446, -3.93352, 0.25929, 0.07502, -1.39146, 0.24838, 0.38527, 0.13048]
}

# Meteorological Parameters and Simulated Data
meteo_params = ["Rainfall", "Relative Humidity", "Max Temperature", "Minimum Temperature", "Wind Direction",
                "Wind Speed"]

# Simulated Meteorological True & Predicted Values
y_true_meteo = {
    "Rainfall": np.random.uniform(0, 50),
    "Relative Humidity": np.random.uniform(50, 100),
    "Max Temperature": np.random.uniform(25, 35),
    "Minimum Temperature": np.random.uniform(18, 25),
    "Wind Direction": np.random.uniform(0, 360),
    "Wind Speed": np.random.uniform(0, 20)
}
y_pred_lstm_meteo = {param: y_true_meteo[param] + np.random.normal(0, np.random.uniform(0.5, 5)) for param in
                     meteo_params}
y_pred_cnn_meteo = {param: y_true_meteo[param] + np.random.normal(0, np.random.uniform(0.7, 7)) for param in
                    meteo_params}
y_pred_hybrid_meteo = {param: y_true_meteo[param] + np.random.normal(0, np.random.uniform(0.3, 3)) for param in
                       meteo_params}

meteo_predictions = {
    "LSTM": y_pred_lstm_meteo,
    "CNN": y_pred_cnn_meteo,
    "Hybrid": y_pred_hybrid_meteo
}

# Meteorological Model Metrics (manually input)
meteo_metrics_data = {
    "Model": ["LSTM"] * 6 + ["CNN"] * 6 + ["Hybrid"] * 6,
    "Parameter": meteo_params * 3,
    "MSE": [
        7482.55, 37.8890, 6.8140, 7.8374, 21489.9642, 50.2397,
        7761.7486, 10.5074, 13.3732, 0.8315, 16884.6820, 4.2438,
        6134.6893, 7.5743, 3.1053, 0.6286, 15458.7536, 0.8482
    ],
    "MAE": [
        65.36984, 4.17260, 2.12671, 1.72942, 126.47095, 5.14132,
        67.31469, 2.42539, 3.02555, 0.63255, 107.33447, 1.90034,
        57.50735, 2.25265, 1.38521, 0.59754, 96.45390, 0.72400
    ],
    "R²": [
        0.47031, -1.87164, -3.38781, -15.88326, -0.07401, -116.55547,
        0.45055, 0.20353, -7.61236, -0.79087, 0.15545, -8.92233,
        0.56573, 0.42606, -1.00029, -0.35387, 0.22742, -0.98397
    ]
}


# Volcanic Activity Parameters and Simulated Data
volcanic_params = ["CO₂", "SO₂"]

# Simulated Volcanic Activity True & Predicted Values
y_true_volcanic = {
    "CO₂": np.random.uniform(400, 1500),  # Simulating elevated CO2
    "SO₂": np.random.uniform(0, 200)  # Simulating SO2 in ppb
}

y_pred_lstm_volcanic = {param: y_true_volcanic[param] + np.random.normal(0, np.random.uniform(10, 50)) for param in
                        volcanic_params}
y_pred_cnn_volcanic = {param: y_true_volcanic[param] + np.random.normal(0, np.random.uniform(15, 70)) for param in
                       volcanic_params}
y_pred_hybrid_volcanic = {param: y_true_volcanic[param] + np.random.normal(0, np.random.uniform(5, 30)) for param in
                          volcanic_params}

volcanic_predictions = {
    "LSTM": y_pred_lstm_volcanic,
    "CNN": y_pred_cnn_volcanic,
    "Hybrid": y_pred_hybrid_volcanic
}

# Volcanic Activity Metrics Table (Manually Input)
volcanic_metrics_data = {
    "Parameter": volcanic_params,
    "MSE": [387161.5542, 1618533.3276],
    "MAE": [588.08010, 899.45384],
    "R²": [-1.16058, -0.65138]
}


# --- Functions to retrieve metrics (now using the separated variables) ---

def get_wq_metrics():
    df_lstm = pd.DataFrame(wq_lstm_metrics)
    df_lstm["Model"] = "LSTM"
    df_cnn = pd.DataFrame(wq_cnn_metrics)
    df_cnn["Model"] = "CNN"
    df_hybrid = pd.DataFrame(wq_hybrid_metrics)
    df_hybrid["Model"] = "Hybrid"
    combined_df = pd.concat([df_lstm, df_cnn, df_hybrid])
    return combined_df

def get_meteo_metrics():
    return pd.DataFrame(meteo_metrics_data)

def get_volcanic_metrics():
    df_volcanic = pd.DataFrame(volcanic_metrics_data)
    df_volcanic["Model"] = "LSTM" # Assuming only LSTM metrics are provided for volcanic for simplicity
    return df_volcanic

# --- Water Quality Index (WQI) Calculation ---
# Simplified WQI for demonstration. In a real application, you'd use established WQI methodologies.
# This example considers DO, pH, Ammonia, Nitrate, and Phosphate.
# For simplicity, we'll assume ideal ranges and assign penalty scores outside these ranges.
# Lower WQI score indicates better water quality.

def calculate_wqi(ammonia, do, nitrate, ph, phosphate):
    # Ideal ranges (example values - these should be based on actual water quality standards)
    ideal_do = (5.0, 9.0)  # mg/L
    ideal_ph = (6.5, 8.5)
    ideal_ammonia = (0.0, 0.1)  # mg/L
    ideal_nitrate = (0.0, 10.0) # mg/L
    ideal_phosphate = (0.0, 0.1) # mg/L

    # Weights for each parameter (example weights)
    # These weights should reflect the relative importance of each parameter to overall quality
    w_ammonia = 0.25
    w_do = 0.30
    w_nitrate = 0.15
    w_ph = 0.20
    w_phosphate = 0.10

    # Calculate sub-indices (simplified penalty score: higher penalty for worse quality)
    # A score of 0 indicates ideal, higher scores indicate deviation/pollution
    
    # Dissolved Oxygen (DO) - too low or too high is bad
    if do < ideal_do[0]:
        si_do = (ideal_do[0] - do) * 10 # Larger deviation, higher penalty
    elif do > ideal_do[1]:
        si_do = (do - ideal_do[1]) * 5 # Slightly less penalty for too high, adjust as needed
    else:
        si_do = 0

    # pH Level - too low or too high is bad
    if ph < ideal_ph[0]:
        si_ph = (ideal_ph[0] - ph) * 15 # pH is very sensitive
    elif ph > ideal_ph[1]:
        si_ph = (ph - ideal_ph[1]) * 15
    else:
        si_ph = 0

    # Ammonia - higher is worse
    if ammonia > ideal_ammonia[1]:
        si_ammonia = (ammonia - ideal_ammonia[1]) * 50 # Ammonia is very critical
    else:
        si_ammonia = 0
    
    # Nitrate - higher is worse
    if nitrate > ideal_nitrate[1]:
        si_nitrate = (nitrate - ideal_nitrate[1]) * 10
    else:
        si_nitrate = 0

    # Phosphate - higher is worse
    if phosphate > ideal_phosphate[1]:
        si_phosphate = (phosphate - ideal_phosphate[1]) * 30 # Phosphate can lead to eutrophication
    else:
        si_phosphate = 0

    # Combine sub-indices with weights
    wqi = (si_ammonia * w_ammonia +
           si_do * w_do +
           si_nitrate * w_nitrate +
           si_ph * w_ph +
           si_phosphate * w_phosphate)
    
    # Normalize WQI to a scale (e.g., 0-100 or 0-10) - this is a simplified normalization
    # For a penalty-based WQI, a higher number means worse quality.
    # We can cap it or scale it to make it more intuitive if needed.
    # Let's say, 0-100, where 0 is perfect and 100+ is very bad.
    wqi = min(wqi, 100.0) # Cap at 100 for display purposes if it goes too high with this simplified model

    return wqi

def get_wqi_category(wqi_score):
    if wqi_score <= 10:
        return "Excellent"
    elif wqi_score <= 25:
        return "Good"
    elif wqi_score <= 50:
        return "Fair"
    elif wqi_score <= 75:
        return "Poor"
    else:
        return "Very Poor"

# Tabs for navigation
tab = st.radio("Select Data Type",
                ["Water Quality Parameters", "Meteorological Parameters", "Volcanic Activity Parameters"])

# --- Water Quality Section ---
if tab == "Water Quality Parameters":
    st.subheader("Model Performance Metrics (Water Quality)")
    st.dataframe(get_wq_metrics(), use_container_width=True)

    # Note: Bar chart below still uses a single selected model for comparison
    st.subheader("Parameter Comparison (Water Quality)")
    models = ["LSTM", "CNN", "Hybrid"]
    preds = [y_pred_lstm_wq, y_pred_cnn_wq, y_pred_hybrid_wq]
    selected_model = st.selectbox("Select Model", models)
    selected_pred = preds[models.index(selected_model)]

    # Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name="True Values", x=water_quality_params, y=y_true_wq, marker_color='#33658A'))
    fig.add_trace(
        go.Bar(name=f"Predicted by {selected_model}", x=water_quality_params, y=selected_pred, marker_color='#F26430'))
    fig.update_layout(
        barmode='group',
        xaxis_title="Parameter",
        yaxis_title="Value",
        plot_bgcolor='#fffafa',
        paper_bgcolor='#fffafa',
        font=dict(color="#020310")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Line Chart - UPDATED to show all models
    st.subheader("Trend Line Comparison (Water Quality)")

    # Prepare data for line chart including all models
    line_data_wq = {
        "Parameter": water_quality_params * 4,  # True, LSTM, CNN, Hybrid
        "Value": list(y_true_wq) + list(y_pred_lstm_wq) + list(y_pred_cnn_wq) + list(y_pred_hybrid_wq),
        "Type": ["True Values"] * len(water_quality_params) + \
                ["LSTM"] * len(water_quality_params) + \
                ["CNN"] * len(water_quality_params) + \
                ["Hybrid"] * len(water_quality_params)
    }
    line_df_wq = pd.DataFrame(line_data_wq)

    line_fig_wq = px.line(line_df_wq,
                          x="Parameter",
                          y="Value",
                          color="Type",
                          markers=True,
                          color_discrete_sequence=["#33658A", "#F26430", "#3E92CC", "#6D597A"])  # Added more colors
    line_fig_wq.update_layout(plot_bgcolor='#fffafa', paper_bgcolor='#fffafa', font=dict(color="#020310"))
    st.plotly_chart(line_fig_wq, use_container_width=True)

    # Metric display (still based on single selected model)
    st.subheader("Live Parameter Meters")
    cols = st.columns(4)
    for i in range(len(water_quality_params)):
        with cols[i % 4]:
            st.metric(label=water_quality_params[i], value=f"{selected_pred[i]:.2f}")

    # --- Water Quality Index Display ---
    st.markdown("---")
    st.subheader("Overall Water Quality Index (WQI)")

    # Get the index for each relevant parameter in the water_quality_params list
    # This assumes the order of water_quality_params is consistent
    try:
        ammonia_idx = water_quality_params.index("Ammonia")
        do_idx = water_quality_params.index("Dissolved Oxygen")
        nitrate_idx = water_quality_params.index("Nitrate")
        ph_idx = water_quality_params.index("pH Level")
        phosphate_idx = water_quality_params.index("Phosphate")

        # Calculate WQI for True Values
        true_wqi = calculate_wqi(
            ammonia=y_true_wq[ammonia_idx],
            do=y_true_wq[do_idx],
            nitrate=y_true_wq[nitrate_idx],
            ph=y_true_wq[ph_idx],
            phosphate=y_true_wq[phosphate_idx]
        )
        true_wqi_category = get_wqi_category(true_wqi)

        # Calculate WQI for the selected model's predictions
        predicted_wqi = calculate_wqi(
            ammonia=selected_pred[ammonia_idx],
            do=selected_pred[do_idx],
            nitrate=selected_pred[nitrate_idx],
            ph=selected_pred[ph_idx],
            phosphate=selected_pred[phosphate_idx]
        )
        predicted_wqi_category = get_wqi_category(predicted_wqi)

        # Display WQI metrics
        wqi_cols = st.columns(2)
        with wqi_cols[0]:
            st.metric(label="Current WQI (True Values)", value=f"{true_wqi:.2f}", help="Lower score is better water quality.")
            st.info(f"**Category:** {true_wqi_category}")
        with wqi_cols[1]:
            st.metric(label=f"Predicted WQI by {selected_model}", value=f"{predicted_wqi:.2f}", help="Lower score is better water quality.")
            st.info(f"**Category:** {predicted_wqi_category}")

        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
            <p style="font-size: 14px; color: #555;">
                <strong>WQI Categories (Example):</strong><br>
                0-10: Excellent<br>
                11-25: Good<br>
                26-50: Fair<br>
                51-75: Poor<br>
                76-100+: Very Poor<br>
                <br>
                <em>Note: This WQI is a simplified example. For practical applications, use established WQI methodologies and standards.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)


    except ValueError as e:
        st.error(f"Error calculating WQI: One or more required water quality parameters (Ammonia, Dissolved Oxygen, Nitrate, pH Level, Phosphate) are missing from `water_quality_params`. Please ensure they are included and spelled correctly. Details: {e}")

# --- Meteorological Section ---
elif tab == "Meteorological Parameters":
    st.subheader("Model Performance Metrics (Meteorological Parameters)")
    meteo_df = get_meteo_metrics()
    st.dataframe(meteo_df, use_container_width=True)

    # Select model for plotting (Bar Chart and Live Meters will still use this)
    model_selected_meteo = st.selectbox("Select Model for Meteorological Analysis", ["LSTM", "CNN", "Hybrid"])

    # Bar Chart for True vs Predicted Meteorological Values
    st.subheader("Parameter Comparison (Meteorological)")

    meteo_bar_data = pd.DataFrame({
        "Parameter": meteo_params,
        "True Values": [y_true_meteo[param] for param in meteo_params],
        f"Predicted by {model_selected_meteo}": [meteo_predictions[model_selected_meteo][param] for param in
                                                 meteo_params]
    })

    fig_meteo_bar = go.Figure()
    fig_meteo_bar.add_trace(go.Bar(name="True Values", x=meteo_bar_data["Parameter"], y=meteo_bar_data["True Values"],
                                   marker_color='#33658A'))
    fig_meteo_bar.add_trace(go.Bar(name=f"Predicted by {model_selected_meteo}", x=meteo_bar_data["Parameter"],
                                   y=meteo_bar_data[f"Predicted by {model_selected_meteo}"], marker_color='#F26430'))
    fig_meteo_bar.update_layout(
        barmode='group',
        xaxis_title="Parameter",
        yaxis_title="Value",
        plot_bgcolor='#fffafa',
        paper_bgcolor='#fffafa',
        font=dict(color="#020310")
    )
    st.plotly_chart(fig_meteo_bar, use_container_width=True)

    # Line Chart - UPDATED to show all models
    st.subheader("Trend Line Comparison (Meteorological)")

    # Prepare data for line chart including all models
    line_data_meteo = {
        "Parameter": meteo_params * 4,
        "Value": [y_true_meteo[param] for param in meteo_params] + \
                 [meteo_predictions["LSTM"][param] for param in meteo_params] + \
                 [meteo_predictions["CNN"][param] for param in meteo_params] + \
                 [meteo_predictions["Hybrid"][param] for param in meteo_params],
        "Type": ["True Values"] * len(meteo_params) + \
                ["LSTM"] * len(meteo_params) + \
                ["CNN"] * len(meteo_params) + \
                ["Hybrid"] * len(meteo_params)
    }
    line_df_meteo = pd.DataFrame(line_data_meteo)

    line_fig_meteo = px.line(line_df_meteo,
                             x="Parameter",
                             y="Value",
                             color="Type",
                             markers=True,
                             color_discrete_sequence=["#33658A", "#F26430", "#3E92CC", "#6D597A"])  # Added more colors
    line_fig_meteo.update_layout(plot_bgcolor='#fffafa', paper_bgcolor='#fffafa', font=dict(color="#020310"))
    st.plotly_chart(line_fig_meteo, use_container_width=True)

    # Live Parameter Meters for Meteorological Data
    st.subheader("Live Meteorological Parameter Meters")
    cols_meteo = st.columns(3)  # Adjust column count as needed
    selected_pred_meteo = meteo_predictions[model_selected_meteo]

    for i, param in enumerate(meteo_params):
        with cols_meteo[i % 3]:
            st.metric(label=param, value=f"{selected_pred_meteo[param]:.2f}")

# --- Volcanic Activity Section ---
else:  # tab == "Volcanic Activity Parameters"
    st.subheader("Model Performance Metrics (Volcanic Activity Parameters)")
    st.dataframe(get_volcanic_metrics(), use_container_width=True)

    # Moved selectbox to be immediately after metrics table for consistency
    volcanic_models = ["LSTM", "CNN", "Hybrid"]
    selected_volcanic_model = st.selectbox("Select Model for Volcanic Analysis", volcanic_models,
                                           index=volcanic_models.index("LSTM"))
    selected_pred_volcanic = volcanic_predictions[selected_volcanic_model]

    st.subheader("Parameter Comparison (Volcanic Activity)")

    # Bar Chart for Volcanic Activity
    fig_volcanic_bar = go.Figure()
    fig_volcanic_bar.add_trace(go.Bar(name="True Values", x=volcanic_params,
                                     y=[y_true_volcanic[param] for param in volcanic_params],
                                     marker_color='#33658A'))
    fig_volcanic_bar.add_trace(go.Bar(name=f"Predicted by {selected_volcanic_model}", x=volcanic_params,
                                     y=[selected_pred_volcanic[param] for param in volcanic_params],
                                     marker_color='#F26430'))
    fig_volcanic_bar.update_layout(
        barmode='group',
        xaxis_title="Air Quality Parameter",
        yaxis_title="Value",
        plot_bgcolor='#fffafa',
        paper_bgcolor='#fffafa',
        font=dict(color="#020310")
    )
    st.plotly_chart(fig_volcanic_bar, use_container_width=True)

    # Line Chart - UPDATED to show all models
    st.subheader("Trend Line Comparison (Volcanic Activity)")

    # Prepare data for line chart including all models
    line_data_volcanic = {
        "Parameter": volcanic_params * 4,
        "Value": [y_true_volcanic[param] for param in volcanic_params] + \
                 [volcanic_predictions["LSTM"][param] for param in volcanic_params] + \
                 [volcanic_predictions["CNN"][param] for param in volcanic_params] + \
                 [volcanic_predictions["Hybrid"][param] for param in volcanic_params],
        "Type": ["True Values"] * len(volcanic_params) + \
                ["LSTM"] * len(volcanic_params) + \
                ["CNN"] * len(volcanic_params) + \
                ["Hybrid"] * len(volcanic_params)
    }
    line_df_volcanic = pd.DataFrame(line_data_volcanic)

    line_fig_volcanic = px.line(line_df_volcanic,
                                 x="Parameter",
                                 y="Value",
                                 color="Type",
                                 markers=True,
                                 color_discrete_sequence=["#33658A", "#F26430", "#3E92CC",
                                                         "#6D597A"])  # Added more colors
    line_fig_volcanic.update_layout(plot_bgcolor='#fffafa', paper_bgcolor='#fffafa', font=dict(color="#020310"))
    st.plotly_chart(line_fig_volcanic, use_container_width=True)

    # Live Parameter Meters for Volcanic Activity
    st.subheader("Live Volcanic Activity Parameter Meters")
    cols_volcanic = st.columns(2)  # Two columns for CO2 and SO2
    for i, param in enumerate(volcanic_params):
        with cols_volcanic[i % 2]:
            st.metric(label=param, value=f"{selected_pred_volcanic[param]:.2f}")

# Footer
st.markdown("---")
st.caption("Dashboard updated in real-time. Replace simulated or static values with model outputs for production.")
