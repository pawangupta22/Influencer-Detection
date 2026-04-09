import streamlit as st
import pickle
import numpy as np

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
le_platform = pickle.load(open("platform.pkl", "rb"))
le_post = pickle.load(open("post.pkl", "rb"))

st.set_page_config(page_title="Influencer Detection", layout="centered")

st.title("📊 Influencer Detection System")
st.write("Enter your average post performance (last 5–10 posts)")

# INPUTS
platform = st.selectbox("Platform", le_platform.classes_)
post_type = st.selectbox("Post Type", le_post.classes_)

post_length = st.number_input("Post Length (seconds)", min_value=1)

views = st.number_input("Average Views per Post", min_value=0)
likes = st.number_input("Average Likes per Post", min_value=0)
comments = st.number_input("Average Comments per Post", min_value=0)
shares = st.number_input("Average Shares per Post", min_value=0)

# PREDICT
if st.button("Predict Influencer Level"):

    # Encode
    p = le_platform.transform([platform])[0]
    pt = le_post.transform([post_type])[0]

    # Engagement Rate
    if views > 0:
        engagement_rate = ((likes + comments + shares) / views) * 100
    else:
        engagement_rate = 0

    # Prediction
    input_data = np.array([[p, pt, post_length, views, likes, comments, shares]])
    result = model.predict(input_data)[0]

    # OUTPUT
    st.success(f"🎯 Influencer Level: {result}")
    st.info(f"📈 Engagement Rate: {round(engagement_rate, 2)}%")

    # Suggestions
    if engagement_rate < 1:
        st.warning("Low engagement → Improve content quality & hooks")
    elif engagement_rate < 5:
        st.info("Average engagement → Stay consistent")
    else:
        st.success("High engagement → You're doing great!")
    