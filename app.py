import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Influencer Detection AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.title {
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:#38bdf8;
}
.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}
.metric-box{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 4px 10px rgba(0,0,0,0.2);
}
.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#06b6d4,#3b82f6);
    color:white;
    border:none;
    padding:14px;
    font-size:18px;
    border-radius:12px;
    font-weight:bold;
}
.stButton>button:hover{
    background:linear-gradient(90deg,#0284c7,#2563eb);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">📊 Influencer Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced AI Based Social Media Profile Analyzer</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ About")
st.sidebar.info("""
This AI model predicts whether a user is:

✅ Real Influencer  
📈 Growing Influencer  
❌ Not Influencer  

Enter social media profile data to analyze.
""")

# -----------------------------
# Input Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Basic Stats")
    followers = st.number_input("Followers", min_value=1, step=100)
    following = st.number_input("Following", min_value=0, step=10)
    posts = st.number_input("Total Posts", min_value=0, step=1)
    avg_posts_per_day = st.slider("Avg Posts Per Day", 0.0, 20.0, 1.0)

    account_age = st.slider("Account Age (Months)", 1, 120, 12)

with col2:
    st.subheader("📈 Engagement Stats")
    avg_views = st.number_input("Avg Views / Post", min_value=0, step=100)
    avg_likes = st.number_input("Avg Likes / Post", min_value=0, step=10)
    avg_comments = st.number_input("Avg Comments / Post", min_value=0, step=1)
    avg_shares = st.number_input("Avg Shares / Post", min_value=0, step=1)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🚀 Analyze Profile"):

    engagement_rate = (avg_likes + avg_comments + avg_shares) / followers
    views_ratio = avg_views / followers

    data = [[
        followers,
        following,
        posts,
        avg_posts_per_day,
        avg_views,
        avg_likes,
        avg_comments,
        avg_shares,
        account_age,
        engagement_rate,
        views_ratio
    ]]

    pred = model.predict(data)
    result = le.inverse_transform(pred)[0]

    if result in ["Real Influencer", "Growing Influencer"]:
        status = "Influencer ✅"
        color = "green"
    else:
        status = "Not Influencer ❌"
        color = "red"

    # -----------------------------
    # Results
    # -----------------------------
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    c1, c2, c3 = st.columns(3)

    c1.metric("User Type", result)
    c2.metric("Status", status)
    c3.metric("Engagement Rate", f"{engagement_rate:.2%}")

    st.metric("Views Ratio", f"{views_ratio:.2f}")

    # Data Summary Table
    st.subheader("📄 Input Summary")

    df = pd.DataFrame({
        "Metric": [
            "Followers", "Following", "Posts", "Avg Posts/Day",
            "Views", "Likes", "Comments", "Shares", "Account Age"
        ],
        "Value": [
            followers, following, posts, avg_posts_per_day,
            avg_views, avg_likes, avg_comments, avg_shares, account_age
        ]
    })

    st.dataframe(df, use_container_width=True)

    # Success Message
    if "Influencer" in status:
        st.success("This profile shows strong influencer signals.")
    else:
        st.warning("This profile currently has low influencer signals.")
