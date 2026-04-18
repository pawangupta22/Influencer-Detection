import streamlit as st
import joblib

model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

st.title("Influencer Classification System")

st.write("Enter user details")

# Inputs
followers = st.number_input("Followers", min_value=1)
following = st.number_input("Following", min_value=0)
posts = st.number_input("Posts", min_value=0)
views = st.number_input("Views", min_value=0)

likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
shares = st.number_input("Shares", min_value=0)

account_age_days = st.number_input("Account Age (days)", min_value=1)

# Prediction
if st.button("Predict"):

    # average engagement
    avg_engagement = (likes + comments + shares) / 3

    # engagement rate
    engagement_rate = (likes + comments + shares) / followers

    data = [[
        followers, following, posts, views,
        avg_engagement, shares, account_age_days, engagement_rate
    ]]

    pred = model.predict(data)
    result = le.inverse_transform(pred)

    st.write("Average Engagement:", int(avg_engagement))
    st.write("Engagement Rate:", round(engagement_rate, 4))
    st.write("User Type:", result[0])