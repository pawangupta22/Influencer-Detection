import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

st.set_page_config(page_title="Influencer Detection")

st.title("Influencer Detection System")

st.write("Enter profile details")

# Inputs
followers = st.number_input("Followers", min_value=1)
following = st.number_input("Following", min_value=0)
posts = st.number_input("Posts", min_value=0)
avg_posts_per_day = st.number_input("Avg Posts Per Day", min_value=0.0)

avg_views = st.number_input("Avg Views Per Post", min_value=0)
avg_likes = st.number_input("Avg Likes Per Post", min_value=0)
avg_comments = st.number_input("Avg Comments Per Post", min_value=0)
avg_shares = st.number_input("Avg Shares Per Post", min_value=0)

account_age = st.number_input("Account Age (months)", min_value=1)

# Prediction
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

    # Influencer or Not
    if result in ["Real Influencer", "Growing Influencer"]:
        status = "Influencer"
    else:
        status = "Not Influencer"

    # Output
    st.subheader("Result")
    st.write("User Type:", result)
    st.write("Status:", status)
    st.write("Engagement Rate:", round(engagement_rate, 4))
    st.write("Views Ratio:", round(views_ratio, 4))
