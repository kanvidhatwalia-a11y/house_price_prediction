import joblib
import pandas as pd
import streamlit as st


# Load trained model
model = joblib.load("house_price_model.pkl")


# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)


# Title
st.title("House Price Prediction")

st.subheader("🏠 Enter House Details")

st.info(
    "💡 Enter the house details below and click "
    "Predict to get an estimated price."
)

st.caption(
    "Adjust the values below to estimate the house price."
)


# Overall Quality
overall = st.slider(
    "⭐ Overall Quality",
    min_value=1,
    max_value=10,
    value=5,
    key="overall",
    help="Rate the overall quality of the house from 1 to 10.",
)


# First row
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "📐 Ground Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=50,
        key="area",
        help=(
            "Enter the above-ground living area "
            "of the house in square feet."
        ),
    )

with col2:
    garage = st.slider(
        "🚗 Garage Capacity (Cars)",
        min_value=0,
        max_value=5,
        value=2,
        key="garage",
        help=(
            "Select the number of cars that can "
            "fit in the garage."
        ),
    )


# Second row
col3, col4 = st.columns(2)

with col3:
    basement = st.number_input(
        "🏠 Total Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
        key="basement",
        help=(
            "Enter the total basement area "
            "in square feet."
        ),
    )

with col4:
    bath = st.slider(
        "🛁 Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        key="bath",
        help=(
            "Select the number of full bathrooms "
            "in the house."
        ),
    )


# Reset function
def reset_inputs():
    st.session_state["overall"] = 5
    st.session_state["area"] = 1500
    st.session_state["garage"] = 2
    st.session_state["basement"] = 1000
    st.session_state["bath"] = 2


# Reset button
st.button(
    "🔄 Reset",
    on_click=reset_inputs,
    use_container_width=True,
)


# Prediction button
if st.button(
    "🔮 Predict House Price",
    use_container_width=True,
):

    # Create input DataFrame
    data = pd.DataFrame(
        {
            "OverallQual": [overall],
            "GrLivArea": [area],
            "GarageCars": [garage],
            "TotalBsmtSF": [basement],
            "FullBath": [bath],
        }
    )

    # Make prediction
    prediction = model.predict(data)

    # Display prediction
    st.success("🏠 Predicted House Price")

    st.markdown(
        f"## ₹{prediction[0]:,.2f}"
    )

    st.divider()


# About project
with st.expander("📌 About this Project"):
    st.write(
        """
        This House Price Prediction app uses a
        Machine Learning model to estimate house prices.

        The prediction is based on:

        • Overall Quality
        • Ground Living Area
        • Garage Capacity
        • Total Basement Area
        • Full Bathrooms
        """
    )


# Model information
with st.expander("🤖 Model Information"):
    st.write(
        """
        **Algorithm:** Random Forest Regressor

        **Problem Type:** House Price Regression

        **Evaluation Metric:** R² Score

        The model predicts the estimated house price
        based on the selected house features.
        """
    )


# How it works
with st.expander("💡 How does it work?"):
    st.write(
        """
        1. Enter the house details.
        2. Click on Predict House Price.
        3. The trained Machine Learning model
           processes the inputs.
        4. The estimated house price is displayed below.
        """
    )


# Footer
st.divider()

st.caption(
    "🏠 House Price Prediction | Built with Python, "
    "Machine Learning & Streamlit"
)
