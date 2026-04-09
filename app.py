import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Influencer Detector", layout="centered")

st.title("Influencer Detection")
st.write("Check if an account is Real or Suspicious")

# =========================
# INPUTS (MATCH TRAINING)
# =========================

followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)

follower_following_ratio = st.number_input("Follower/Following Ratio", min_value=0.0)

account_age_days = st.number_input("Account Age (Days)", min_value=0)

posts = st.number_input("Total Posts", min_value=0)
posts_per_day = st.number_input("Posts per Day", min_value=0.0)

follow_unfollow_rate = st.slider("Follow-Unfollow Rate (0–1)", 0.0, 1.0, 0.1)

spam_comments_rate = st.slider("Spam Comments Rate (0–1)", 0.0, 1.0, 0.1)
generic_comment_rate = st.slider("Generic Comment Rate (0–1)", 0.0, 1.0, 0.1)

suspicious_links_in_bio = st.selectbox("Suspicious Links in Bio", [0, 1])

# =========================
# PREDICT
# =========================

if st.button("Check Authenticity"):

    input_data = np.array([[
        followers,
        following,
        follower_following_ratio,
        account_age_days,
        posts,
        posts_per_day,
        follow_unfollow_rate,
        spam_comments_rate,
        generic_comment_rate,
        suspicious_links_in_bio
    ]])

    result = model.predict(input_data)[0]

    # =========================
    # OUTPUT
    # =========================

    if result == 1:
        st.success("Real Influencer")
        st.balloons()
    else:
        st.error("Suspicious / Fake Account")

    # Insights
    st.subheader("Insights")

    if spam_comments_rate > 0.5:
        st.warning("High spam activity detected")

    if follow_unfollow_rate > 0.5:
        st.warning("Suspicious follow-unfollow behavior")

    if posts_per_day > 5:
        st.info("Unusual posting frequency")