import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

# Page Config
st.set_page_config(page_title="Influencer Detection", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: #0f172a;
}

h1 {
    color: white;
    text-align: center;
    font-size: 42px;
    margin-bottom: 5px;
}

.subtext {
    color: #94a3b8;
    text-align: center;
    margin-bottom: 30px;
}

div[data-baseweb="input"] input {
    background: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    padding: 10px !important;
}

div[data-baseweb="input"] input:focus {
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 0 8px rgba(59,130,246,0.4);
}

label {
    color: white !important;
    font-weight: 500;
}

.stButton>button {
    width: 100%;
    background: #3b82f6;
    color: white;
    border: none;
    padding: 12px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
}

.stButton>button:hover {
    background: #2563eb;
}

.result-box {
    background: #1e293b;
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin-top: 20px;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>Influencer Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Analyze social media profile performance</p>", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    followers = st.number_input("Followers", min_value=1)
    following = st.number_input("Following", min_value=0)
    posts = st.number_input("Posts", min_value=0)
    avg_posts_per_day = st.number_input("Average Posts Per Day", min_value=0.0)

with col2:
    avg_views = st.number_input("Average Views Per Post", min_value=0)
    avg_likes = st.number_input("Average Likes Per Post", min_value=0)
    avg_comments = st.number_input("Average Comments Per Post", min_value=0)
    avg_shares = st.number_input("Average Shares Per Post", min_value=0)

account_age = st.number_input("Account Age (Months)", min_value=1)

# Predict
if st.button("Predict Result"):

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
        status = "Influencer"
    else:
        status = "Not Influencer"

    st.markdown(f"""
    <div class="result-box">
        <h3>Prediction Result</h3>
        <p><strong>User Type:</strong> {result}</p>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Engagement Rate:</strong> {round(engagement_rate,4)}</p>
        <p><strong>Views Ratio:</strong> {round(views_ratio,4)}</p>
    </div>
    """, unsafe_allow_html=True)
