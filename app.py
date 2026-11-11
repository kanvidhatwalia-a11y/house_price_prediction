import joblib
import pandas as pd
import streamlit as st

# Page configuration - Wide layout modern dashboard look deta hai
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished metric cards & styling
st.markdown("""
    <style>
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained 10-feature model safely
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Selected features (same order as Jupyter)
features = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'FullBath',
    'YearBuilt',
    'YearRemodAdd',
    'GarageYrBlt',
    'YrSold',
    'GarageArea'
]

# Reset function
def reset_inputs():
    st.session_state["overall"] = 5
    st.session_state["area"] = 1500
    st.session_state["garage"] = 2
    st.session_state["basement"] = 1000
    st.session_state["bath"] = 2
    st.session_state["year_built"] = 2000
    st.session_state["year_remod"] = 2000
    st.session_state["garage_year"] = 2000
    st.session_state["year_sold"] = 2008
    st.session_state["garage_area"] = 500

# Header
st.title("🏡 House Price Prediction Dashboard")
st.caption("Accurate real estate valuations powered by Machine Learning")
st.write("")

# 2-Column Split: Left = Inputs, Right = Results & Project Info
col_form, col_display = st.columns([1.3, 1], gap="large")

with col_form:
    st.subheader("📋 Property Specifications")
    
    # Overall Quality Slider (Primary Feature)
    overall = st.slider(
        "⭐ Overall Quality (1 = Poor, 10 = Luxury)",
        min_value=1,
        max_value=10,
        value=5,
        key="overall"
    )

    # Clean Tabbed Interface for the remaining 9 features
    tab_dims, tab_specs, tab_years = st.tabs(["📐 Dimensions & Areas", "🛁 Specs & Garage", "📅 Timeline"])

    with tab_dims:
        c1, c2 = st.columns(2)
        with c1:
            area = st.number_input(
                "Ground Living Area (sq ft)",
                min_value=100,
                max_value=10000,
                value=1500,
                step=50,
                key="area"
            )
        with c2:
            basement = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0,
                max_value=5000,
                value=1000,
                step=50,
                key="basement"
            )
        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=1500,
            value=500,
            step=25,
            key="garage_area"
        )

    with tab_specs:
        c3, c4 = st.columns(2)
        with c3:
            bath = st.slider(
                "Full Bathrooms",
                min_value=0,
                max_value=5,
                value=2,
                key="bath"
            )
        with c4:
            garage = st.slider(
                "Garage Capacity (Cars)",
                min_value=0,
                max_value=5,
                value=2,
                key="garage"
            )

    with tab_years:
        c5, c6 = st.columns(2)
        with c5:
            year_built = st.number_input(
                "Year Built",
                min_value=1872,
                max_value=2010,
                value=2000,
                key="year_built"
            )
            garage_year = st.number_input(
                "Garage Year Built",
                min_value=1900,
                max_value=2010,
                value=2000,
                key="garage_year"
            )
        with c6:
            year_remod = st.number_input(
                "Year Remodeled",
                min_value=1950,
                max_value=2010,
                value=2000,
                key="year_remod"
            )
            year_sold = st.number_input(
                "Year Sold",
                min_value=2006,
                max_value=2010,
                value=2008,
                key="year_sold"
            )

    # Action Buttons Side by Side
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        predict_clicked = st.button("🔮 Calculate Valuation", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🔄 Reset", on_click=reset_inputs, use_container_width=True)

with col_display:
    st.subheader("📊 Valuation Output")
    
    if predict_clicked:
        input_data = pd.DataFrame(
            [[
                overall, area, garage, basement, bath,
                year_built, year_remod, garage_year, year_sold, garage_area
            ]],
            columns=features
        )
        
        with st.spinner("Processing valuation..."):
            pred = model.predict(input_data)[0]
        
        st.success("Valuation estimated successfully!")
        st.metric(label="Estimated Market Value", value=f"${pred:,.2f}")
        st.balloons()
        
        st.info(
            f"💡 **Quick Summary:** Valuation is based on a **{area} sq ft** property "
            f"with quality rating **{overall}/10** and **{bath} full bathrooms**."
        )
    else:
        st.info("👈 Fill out the specifications and click **Calculate Valuation** to see the result.")

    # Project Details Section
    with st.expander("📌 Model & Methodology"):
        st.markdown("""
        * **Algorithm:** Random Forest Regressor
        * **Target Variable:** SalePrice
        * **Metric:** $R^2$ Score
        * **Features Processed:** 10 core real-estate attributes
        """)

    with st.expander("💡 Input Guide"):
        st.markdown("""
        * Adjust dimensions under the **Dimensions** tab.
        * Quality score (1–10) heavily influences property tier.
        * Use **Reset** to restore default baseline values anytime.
        """)

# Clean Footer
st.divider()
st.caption("House Price Prediction Engine | Built with Streamlit & Scikit-Learn")