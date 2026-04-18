import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

st.set_page_config(page_title="Influencer Dashboard")

st.title("Influencer Classification System")

st.write("Enter user details")

# Inputs
followers = st.number_input("Followers", min_value=1)
views = st.number_input("Average Views", min_value=0)
likes = st.number_input("Average Likes", min_value=0)
comments = st.number_input("Average Comments", min_value=0)

# Prediction
if st.button("Predict"):

    engagement_rate = (likes + comments) / followers

    data = [[followers, views, likes, comments, engagement_rate]]

    pred = model.predict(data)
    result = le.inverse_transform(pred)

    st.write("Engagement Rate:", round(engagement_rate, 4))
    st.write("User Type:", result[0])
