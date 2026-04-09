import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Influencer Detection", layout="wide")

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.title("Input Influencer Data")

followers = st.sidebar.number_input("Followers", min_value=1)
views = st.sidebar.number_input("Avg Views", min_value=0)
likes = st.sidebar.number_input("Avg Likes", min_value=0)
comments = st.sidebar.number_input("Avg Comments", min_value=0)
months_old = st.sidebar.number_input("Account Age (months)", min_value=0)

st.sidebar.markdown("---")

# -------------------------
# MAIN TITLE
# -------------------------
st.title("Influencer Detection System")
st.markdown("### Smart influencer classification using ML")

# -------------------------
# CALCULATIONS
# -------------------------
view_rate = views / followers if followers > 0 else 0
engagement_rate = (likes + comments) / followers if followers > 0 else 0

# -------------------------
# PREDICTION
# -------------------------
if st.sidebar.button("Analyze"):

    features = np.array([[followers, views, likes, comments,
                          view_rate, engagement_rate, months_old]])

    prediction = model.predict(features)[0]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Followers", followers)
    col2.metric("View Rate", round(view_rate, 4))
    col3.metric("Engagement Rate", round(engagement_rate, 4))

    st.markdown("---")

    # Result
    if "Genuine" in prediction:
        st.success(f" {prediction}")
    elif "Fake" in prediction:
        st.error(f" {prediction}")
    elif "New" in prediction:
        st.info(f" {prediction}")
    else:
        st.warning(f" {prediction}")

    st.markdown("---")

    # Chart
    st.subheader(" Performance Overview")

    chart_data = pd.DataFrame({
        "Metric": ["Followers", "Views", "Likes", "Comments"],
        "Value": [followers, views, likes, comments]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_data["Metric"], chart_data["Value"])
    st.pyplot(fig)

# -------------------------
# CSV UPLOAD
# -------------------------
st.markdown("## Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.write("Preview:")
    st.dataframe(data.head())

    if st.button("Analyze Dataset"):

        data.fillna(0, inplace=True)

        data['view_rate'] = data['views (avg.)'] / data['followers']
        data['engagement_rate'] = (data['likes (avg.)'] + data['comments (avg.)']) / data['followers']

        data['months_old'] = data.get('months_old', 12)

        features = data[['followers','views (avg.)','likes (avg.)','comments (avg.)',
                         'view_rate','engagement_rate','months_old']]

        data['prediction'] = model.predict(features)

        st.write("Results:")
        st.dataframe(data)