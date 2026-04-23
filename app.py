import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

# Page Config
st.set_page_config(page_title="Influencer Detection", layout="centered")

# Black & White CSS
st.markdown("""
<style>
.stApp {
    background-color: black;
    color: white;
}

h1 {
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

label {
    color: white !important;
    font-weight: 500;
}

div[data-baseweb="input"] input {
    background-color: #111111 !important;
    color: white !important;
    border: 1px solid white !important;
    border-radius: 6px !important;
}

textarea {
    background-color: #111111 !important;
    color: white !important;
    border: 1px solid white !important;
}

.stButton>button {
    width: 100%;
    background-color: white;
    color: black;
    border: none;
    padding: 10px;
    border-radius: 6px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #dddddd;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Influencer Detection System")

# Inputs
followers = st.number_input("Followers", min_value=1)
following = st.number_input("Following", min_value=0)
posts = st.number_input("Posts", min_value=0)
avg_posts_per_day = st.number_input("Average Posts Per Day", min_value=0.0)

avg_views = st.number_input("Average Views Per Post", min_value=0)
avg_likes = st.number_input("Average Likes Per Post", min_value=0)
avg_comments = st.number_input("Average Comments Per Post", min_value=0)
avg_shares = st.number_input("Average Shares Per Post", min_value=0)

account_age = st.number_input("Account Age (Months)", min_value=1)

# Predict
if st.button("Predict"):

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

    st.success(f"User Type: {result}")
    st.write(f"Status: {status}")
    st.write(f"Engagement Rate: {round(engagement_rate,4)}")
    st.write(f"Views Ratio: {round(views_ratio,4)}")
