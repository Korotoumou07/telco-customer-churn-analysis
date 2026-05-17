import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# COLORS
# =========================

DARK = "#2B2D42"
WHITE = "#FFFFFF"
TURQUOISE = "#2EC4B6"
LIGHT_BG = "#F3F7F8"
ACCENT = "#FF9F1C"

# =========================
# CSS DESIGN
# =========================

st.markdown(f"""
<style>
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2rem;
}}

header {{
    visibility: hidden;
}}
.stApp {{
    background-color: {WHITE};
}}

.hero {{
    margin-top: 0px !important;
    padding: 16px;
    border-radius: 20px;
    background: linear-gradient(135deg, {DARK} 0%, {TURQUOISE} 100%);
    color: white;
    margin-bottom: 16px;
    box-shadow: 0px 8px 24px rgba(43, 45, 66, 0.22);
}}

.hero h1 {{
    font-size: 32px;
    font-weight: 900;
    color: white;
    margin-bottom: 6px;
}}

.hero p {{
    font-size: 15px;
    color: white;
    margin-bottom: 0px;
}}

.section-title {{
    font-size: 21px;
    color: {DARK};
    font-weight: 900;
    margin-top: 24px;
    margin-bottom: 10px;
}}

.compact-title {{
    font-size: 18px;
    color: {DARK};
    font-weight: 900;
    margin-bottom: 10px;
}}

.kpi-card {{
    padding: 16px;
    border-radius: 16px;
    background-color: {WHITE};
    box-shadow: 0px 6px 18px rgba(43, 45, 66, 0.10);
    border-left: 6px solid {TURQUOISE};
    min-height: 92px;
}}

.kpi-card-orange {{
    padding: 16px;
    border-radius: 16px;
    background-color: {WHITE};
    box-shadow: 0px 6px 18px rgba(43, 45, 66, 0.10);
    border-left: 6px solid {ACCENT};
    min-height: 92px;
}}

.kpi-title {{
    font-size: 12px;
    color: {DARK};
    font-weight: 700;
    margin-bottom: 6px;
}}

.kpi-value {{
    font-size: 24px;
    color: {DARK};
    font-weight: 900;
}}

.kpi-value-small {{
    font-size: 19px;
    color: {DARK};
    font-weight: 900;
}}

.info-box {{
    padding: 18px;
    border-radius: 16px;
    background-color: {LIGHT_BG};
    border-left: 6px solid {TURQUOISE};
    color: {DARK};
    font-weight: 600;
    line-height: 1.5;
}}

.sidebar-box {{
    padding: 16px;
    border-radius: 16px;
    background: linear-gradient(135deg, {DARK} 0%, {TURQUOISE} 100%);
    color: white;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 14px;
    box-shadow: 0px 8px 22px rgba(0, 0, 0, 0.18);
}}

.sidebar-box strong {{
    color: white;
    font-size: 14px;
}}

.stSidebar {{
    background-color: {DARK};
}}

.stSidebar h1, .stSidebar h2, .stSidebar h3,
.stSidebar p, .stSidebar label, .stSidebar span {{
    color: white !important;
}}

.stDownloadButton button {{
    background-color: {TURQUOISE};
    color: white;
    border-radius: 14px;
    padding: 10px 24px;
    font-weight: 800;
    border: none;
}}

.stDownloadButton button:hover {{
    background-color: {DARK};
    color: white;
}}

[data-testid="stDataFrame"] {{
    border-radius: 16px;
    overflow: hidden;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL FILES
# =========================

model = joblib.load("model/final_churn_model.pkl")
features = joblib.load("model/model_features.pkl")

# =========================
# HERO SECTION
# =========================

st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Prediction App</h1>
    <p>
    Predict customer churn risk using Machine Learning and identify high-risk customers
    to support customer retention strategies.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ App Settings")
st.sidebar.write("Upload your transformed customer dataset.")

threshold = st.sidebar.slider(
    "Churn Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transformed CSV",
    type=["csv"]
)

st.sidebar.markdown("""
<div class="sidebar-box">
    <strong>Business Objective</strong><br>
    This application estimates the probability that a customer will churn.
    It helps identify at-risk customers and supports proactive retention actions.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-box">
    <strong>Model Performance</strong><br>
    Accuracy: 74.73%<br>
    Churn Recall: 77.27%<br>
    ROC AUC: 82.85%<br>
    Optimized Threshold: 0.45
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN APP
# =========================

if uploaded_file is None:

    st.markdown("""
    <div class="info-box">
        👈 Upload your transformed CSV file from the sidebar to start predictions.
    </div>
    """, unsafe_allow_html=True)

else:

    df = pd.read_csv(uploaded_file)

    # =========================
    # DATASET PREVIEW IN SIDEBAR
    # =========================

    with st.sidebar.expander("📁 Dataset Preview"):
        st.write("Preview of the uploaded transformed dataset:")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.write(
            f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        )

    # =========================
    # CHECK REQUIRED FEATURES
    # =========================

    missing_cols = [
        col for col in features
        if col not in df.columns
    ]

    if missing_cols:
        st.warning(
            "Some required columns are missing. They will be automatically filled with 0."
        )
        st.write(missing_cols)

    # =========================
    # SELECT FEATURES
    # =========================

    X = df.reindex(
        columns=features,
        fill_value=0
    )

    # =========================
    # FINAL CLEANING
    # =========================

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    # =========================
    # PREDICTIONS
    # =========================

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    df["Churn_Prediction"] = predictions
    df["Churn_Probability"] = probabilities

    df["Risk_Level"] = pd.cut(
        df["Churn_Probability"],
        bins=[0, 0.40, threshold, 1],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True
    )

    # =========================
    # KPI VALUES
    # =========================

    total_customers = len(df)
    predicted_churners = int(df["Churn_Prediction"].sum())
    avg_probability = df["Churn_Probability"].mean() * 100

    churn_rate = (
        predicted_churners / total_customers * 100
        if total_customers > 0
        else 0
    )

    # =========================
    # COMPACT KPI + MODEL INFO
    # =========================

    left_block, right_block = st.columns([2.2, 1.8])

    with left_block:
        st.markdown(
            '<div class="compact-title">📌 Key Performance Indicators</div>',
            unsafe_allow_html=True
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Customers</div>
                <div class="kpi-value">{total_customers}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Predicted Churners</div>
                <div class="kpi-value">{predicted_churners}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="kpi-card-orange">
                <div class="kpi-title">Churn Rate</div>
                <div class="kpi-value">{churn_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Probability</div>
                <div class="kpi-value">{avg_probability:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with right_block:
        st.markdown(
            '<div class="compact-title">🤖 Model Information</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-title">Final Model</div>
                <div class="kpi-value-small">Voting</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="kpi-card-orange">
                <div class="kpi-title">Threshold</div>
                <div class="kpi-value">{threshold:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Features</div>
                <div class="kpi-value">{len(features)}</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CHARTS
    # =========================

    st.markdown(
        '<div class="section-title">📈 Risk Analysis</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        risk_count = (
            df["Risk_Level"]
            .value_counts()
            .reset_index()
        )

        risk_count.columns = [
            "Risk Level",
            "Count"
        ]

        fig_risk = px.pie(
            risk_count,
            names="Risk Level",
            values="Count",
            hole=0.45,
            title="Customer Risk Distribution",
            color="Risk Level",
            color_discrete_map={
                "Low Risk": TURQUOISE,
                "Medium Risk": ACCENT,
                "High Risk": DARK
            }
        )

        fig_risk.update_layout(
            title_font_color=DARK,
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    with chart_col2:

        fig_hist = px.histogram(
            df,
            x="Churn_Probability",
            nbins=20,
            title="Churn Probability Distribution",
            color="Risk_Level",
            color_discrete_map={
                "Low Risk": TURQUOISE,
                "Medium Risk": ACCENT,
                "High Risk": DARK
            }
        )

        fig_hist.update_layout(
            title_font_color=DARK,
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE,
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    # =========================
    # HIGH RISK CUSTOMERS
    # =========================

    st.markdown(
        '<div class="section-title">🚨 High Risk Customers</div>',
        unsafe_allow_html=True
    )

    high_risk = (
        df[df["Risk_Level"] == "High Risk"]
        .sort_values(
            by="Churn_Probability",
            ascending=False
        )
    )

    st.dataframe(
        high_risk,
        use_container_width=True
    )

    # =========================
    # PREDICTION RESULTS
    # =========================

    st.markdown(
        '<div class="section-title">🧾 Prediction Results</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD RESULTS
    # =========================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv,
        file_name="churn_predictions_results.csv",
        mime="text/csv"
    )