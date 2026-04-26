import streamlit as st
import joblib
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer AI", layout="wide")

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}
.big-result {
    font-size: 32px;
    font-weight: bold;
    color: #22c55e;
}
.story-text {
    font-size: 20px;
    color: #e2e8f0;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("customer_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- TITLE ----------------
st.markdown('<div class="big-title">📊 Customer Segmentation AI</div>', unsafe_allow_html=True)
st.write("Understand your customer in seconds")

# ---------------- INPUT ----------------
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 80, 30)
income = st.sidebar.slider("Income", 10000, 100000, 50000)
spending = st.sidebar.slider("Spending", 0, 2000, 500)
web_purchases = st.sidebar.slider("Web Purchases", 0, 10, 3)
store_purchases = st.sidebar.slider("Store Purchases", 0, 10, 5)
web_visits = st.sidebar.slider("Web Visits", 0, 10, 5)
recency = st.sidebar.slider("Recency", 0, 100, 30)

# ---------------- PREDICT ----------------
if st.sidebar.button("Analyze Customer"):

    input_data = np.array([[age, income, spending, web_purchases,
                            store_purchases, web_visits, recency]])

    scaled = scaler.transform(input_data)
    cluster = model.predict(scaled)[0]

    # ---------------- LABELS ----------------
    labels = {
        0: "💎 High-Value Customer",
        1: "👀 Window Shopper",
        2: "🧓 Low Engagement",
        3: "⚠️ Churn Risk",
        4: "📱 Digital Buyer",
        5: "🏬 Premium Offline Buyer"
    }

    stories = {
        0: "This customer spends a lot and is highly valuable. They are likely loyal and contribute significantly to revenue.",
        1: "This customer browses frequently but rarely buys. They need motivation to convert.",
        2: "This customer shows low activity and engagement. They may not interact much with digital platforms.",
        3: "This customer hasn't interacted recently. They are at risk of leaving.",
        4: "This customer actively shops online and engages digitally.",
        5: "This is a wealthy customer who prefers in-store experiences."
    }

    actions = {
        0: "Offer loyalty rewards and exclusive deals.",
        1: "Provide discounts and targeted ads.",
        2: "Engage through personalized communication.",
        3: "Run win-back campaigns.",
        4: "Push app notifications and digital offers.",
        5: "Enhance in-store experience."
    }

    # ---------------- RESULT ----------------
    st.markdown(f'<div class="big-result">{labels[cluster]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="story-text">{stories[cluster]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- KEY HIGHLIGHTS ----------------
    st.subheader("📊 Key Highlights")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="card"><h3>💰 Income</h3><h2>₹{income}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card"><h3>🛒 Spending</h3><h2>{spending}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card"><h3>📱 Web Visits</h3><h2>{web_visits}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- ACTION ----------------
    st.subheader("🎯 Recommended Action")
    st.success(actions[cluster])

else:
    st.info("👈 Enter details and click Analyze Customer")