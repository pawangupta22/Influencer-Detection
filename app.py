import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Influencer Detection", layout="centered")

# Title
st.title("Influencer Detection System")


# CENTER INPUT FORM
st.subheader("Enter Influencer Details")

followers = st.number_input("Followers", min_value=1)
views = st.number_input("Average Views", min_value=0)
likes = st.number_input("Average Likes", min_value=0)
comments = st.number_input("Average Comments", min_value=0)
months_old = st.number_input("Account Age (months)", min_value=0)

st.markdown("")

# Prediction Button (centered feel)
if st.button("Analyze Influencer"):

    # Calculations
    view_rate = views / followers if followers > 0 else 0
    engagement_rate = (likes + comments) / followers if followers > 0 else 0

    features = np.array([[followers, views, likes, comments,
                          view_rate, engagement_rate, months_old]])

    prediction = model.predict(features)[0]

    st.markdown("---")

    # RESULT DISPLAY
    st.subheader("Result")

    if "Genuine" in prediction:
        st.success(f"{prediction}")
    elif "Fake" in prediction:
        st.error(f" {prediction}")
    elif "New" in prediction:
        st.info(f"{prediction}")
    else:
        st.warning(f" {prediction}")

    # Metrics
    st.markdown("### Metrics")

    st.write(f"**View Rate:** {round(view_rate, 4)}")
    st.write(f"**Engagement Rate:** {round(engagement_rate, 4)}")
