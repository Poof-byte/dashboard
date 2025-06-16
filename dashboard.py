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
st.title("🌊 Environmental Prediction Dashboard")
st.markdown(f"**Date & Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# Parameters
water_quality_params = ["Ammonia", "Bottom Water Temp.", "Middle Water Temp.", "Surface Water Temp.",
                        "Dissolved Oxygen", "Nitrate", "pH Level", "Phosphate"]
meteo_params = ["Rainfall", "Relative Humidity", "Max Temperature", "Minimum Temperature", "Wind Direction",
                "Wind Speed"]
volcanic_params = ["CO₂", "SO₂"]  # New Volcanic Parameters

# Simulated Water Quality True & Predicted Values (for plotting)
y_true_wq = np.random.rand(len(water_quality_params)) * 10
y_pred_lstm_wq = y_true_wq + np.random.normal(0, 0.5, len(water_quality_params))
y_pred_cnn_wq = y_true_wq + np.random.normal(0, 0.7, len(water_quality_params))
y_pred_hybrid_wq = y_true_wq + np.random.normal(0, 0.3, len(water_quality_params))

# Simulated Meteorological True & Predicted Values (for plotting)
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

# Simulated Volcanic Activity True & Predicted Values (for plotting)
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


# Meteorological Model Metrics (manually input)
def create_meteo_df():
    data = {
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
    return pd.DataFrame(data)


# Water Quality Metrics Table
def get_wq_metrics():
    # LSTM Metrics
    lstm_data = {
        "Parameter": water_quality_params,
        "MSE": [0.11830, 0.65115, 1.00949, 0.69685, 0.06093, 2.1903, 0.19347, 0.22463],
        "MAE": [0.09872, 0.53507, 0.84357, 0.42207, 0.04968, 1.56651, 0.15204, 0.15202],
        "R²": [-2.76204, -0.96678, -0.24864, -0.24373, -8.87357, -0.74374, 0.29938, 0.0033]
    }
    df_lstm = pd.DataFrame(lstm_data)
    df_lstm["Model"] = "LSTM"

    # CNN Metrics
    cnn_data = {
        "Parameter": water_quality_params,
        "MSE": [1.39594, 1.40864, 1.03885, 0.73711, 0.73711, 3.81135, 2.49556, 0.65751],
        "MAE": [1.35772, 1.32216, 1.02245, 0.54551, 4.86078, 3.37753, 2.48291, 0.61166],
        "R²": [-2.83056, -8.2045, -0.32234, -0.39157, -6.3013, -4.27999, -115.57361, -7.53973]
    }
    df_cnn = pd.DataFrame(cnn_data)
    df_cnn["Model"] = "CNN"

    # Hybrid Metrics (from the third table provided in previous turns)
    hybrid_data = {
        "Parameter": water_quality_params,
        "MSE": [0.10341, 1.03128, 0.77751, 0.60096, 0.02999, 1.43801, 0.18122, 0.20981],
        "MAE": [0.09339, 0.92348, 0.63248, 0.41008, 0.02339, 1.41256, 0.14368, 0.1575],
        "R²": [-1.87446, -3.93352, 0.25929, 0.07502, -1.39146, 0.24838, 0.38527, 0.13048]
    }
    df_hybrid = pd.DataFrame(hybrid_data)
    df_hybrid["Model"] = "Hybrid"

    combined_df = pd.concat([df_lstm, df_cnn, df_hybrid])
    # No longer grouping by model to show individual parameter metrics per model
    return combined_df


# Volcanic Activity Metrics Table - NEW function using provided static data
def get_volcanic_metrics():
    volcanic_metrics_data = {
        "Parameter": volcanic_params,
        "MSE": [387161.5542, 1618533.3276],
        "MAE": [588.08010, 899.45384],
        "R²": [-1.16058, -0.65138]
    }
    df_volcanic = pd.DataFrame(volcanic_metrics_data)
    df_volcanic["Model"] = "LSTM"
    return df_volcanic


# Tabs for navigation
tab = st.radio("Select Data Type",
               ["Water Quality Parameters", "Meteorological Parameters", "Volcanic Activity Parameters"])

# --- Water Quality Section ---
if tab == "Water Quality Parameters":
    st.subheader("🔢 Model Performance Metrics (Water Quality)")
    st.dataframe(get_wq_metrics(), use_container_width=True)

    # Note: Bar chart below still uses a single selected model for comparison
    st.subheader(":bar_chart: Parameter Comparison (Water Quality)")
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
    st.subheader(":chart_with_upwards_trend: Trend Line Comparison (Water Quality)")

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
    st.subheader(":level_slider: Live Parameter Meters")
    cols = st.columns(4)
    for i in range(len(water_quality_params)):
        with cols[i % 4]:
            st.metric(label=water_quality_params[i], value=f"{selected_pred[i]:.2f}")

# --- Meteorological Section ---
elif tab == "Meteorological Parameters":
    st.subheader("🌦️ Model Performance Metrics (Meteorological Parameters)")
    meteo_df = create_meteo_df()
    st.dataframe(meteo_df, use_container_width=True)

    # Select model for plotting (Bar Chart and Live Meters will still use this)
    model_selected_meteo = st.selectbox("Select Model for Meteorological Analysis", ["LSTM", "CNN", "Hybrid"])

    # Bar Chart for True vs Predicted Meteorological Values
    st.subheader(":bar_chart: Parameter Comparison (Meteorological)")

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
    st.subheader(":chart_with_upwards_trend: Trend Line Comparison (Meteorological)")

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
    st.subheader(":level_slider: Live Meteorological Parameter Meters")
    cols_meteo = st.columns(3)  # Adjust column count as needed
    selected_pred_meteo = meteo_predictions[model_selected_meteo]

    for i, param in enumerate(meteo_params):
        with cols_meteo[i % 3]:
            st.metric(label=param, value=f"{selected_pred_meteo[param]:.2f}")

# --- Volcanic Activity Section ---
else:  # tab == "Volcanic Activity Parameters"
    st.subheader("🌋 Model Performance Metrics (Volcanic Activity Parameters)")
    st.dataframe(get_volcanic_metrics(), use_container_width=True)

    # Moved selectbox to be immediately after metrics table for consistency
    volcanic_models = ["LSTM", "CNN", "Hybrid"]
    selected_volcanic_model = st.selectbox("Select Model for Volcanic Analysis", volcanic_models,
                                           index=volcanic_models.index("LSTM"))
    selected_pred_volcanic = volcanic_predictions[selected_volcanic_model]

    st.subheader(":bar_chart: Parameter Comparison (Volcanic Activity)")

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
    st.subheader(":chart_with_upwards_trend: Trend Line Comparison (Volcanic Activity)")

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
    st.subheader(":level_slider: Live Volcanic Activity Parameter Meters")
    cols_volcanic = st.columns(2)  # Two columns for CO2 and SO2
    for i, param in enumerate(volcanic_params):
        with cols_volcanic[i % 2]:
            st.metric(label=param, value=f"{selected_pred_volcanic[param]:.2f}")

# Footer
st.markdown("---")
st.caption("Dashboard updated in real-time. Replace simulated or static values with model outputs for production.")