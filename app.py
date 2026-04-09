import streamlit as st
import pickle
import numpy as np

# Load
model_inf = pickle.load(open("model_influencer.pkl","rb"))
model_sus = pickle.load(open("model_suspicious.pkl","rb"))
le_cat = pickle.load(open("cat.pkl","rb"))
le_country = pickle.load(open("country.pkl","rb"))

st.title("Influencer Detection")

# =========================
# INPUT
# =========================
category = st.selectbox("Category", le_cat.classes_)
country = st.selectbox("Country", le_country.classes_)

followers = st.number_input("Followers", min_value=1)
views = st.number_input("Avg Views", min_value=0)
likes = st.number_input("Avg Likes", min_value=0)
comments = st.number_input("Avg Comments", min_value=0)

# =========================
# PREDICT
# =========================
if st.button("Analyze"):

    cat = le_cat.transform([category])[0]
    ctr = le_country.transform([country])[0]

    data = np.array([[cat, ctr, followers, views, likes, comments]])

    inf = model_inf.predict(data)[0]
    sus = model_sus.predict(data)[0]

    # Metrics
    if followers > 0:
        engagement = ((likes + comments) / followers) * 100
        performance = (views / followers) * 100
    else:
        engagement = 0
        performance = 0

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Results")

    st.write(f"Engagement Rate: {round(engagement,2)}%")
    st.write(f"Performance Rate: {round(performance,2)}%")

    if inf == 1:
        st.success("Influencer Account")
    else:
        st.warning("Not an Influencer")

    if sus == 1:
        st.error("Suspicious Audience Detected")
    else:
        st.success("Healthy Audience")

    # Insights
    st.subheader("Insights")

    if engagement < 1:
        st.warning("Low engagement compared to followers")

    if performance < 5:
        st.info("Content reach is low")

    if engagement > 5:
        st.success("Strong audience engagement ")