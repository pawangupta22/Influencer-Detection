# ==============================
# Influencer Detection Dashboard
# ==============================

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Influencer Detector", layout="centered")

st.title("Influencer Detection Dashboard")
st.write("Enter details to check if a person is an influencer")

# ==============================
# Load & Train Model
# ==============================

@st.cache_data
def load_model():
    df = pd.read_csv("influncer_dection.csv")

    # --- Data Cleaning ---
    def convert(x):
        x = str(x).replace(',', '').strip()
        if 'M' in x:
            return float(x.replace('M','')) * 1000000
        elif 'K' in x:
            return float(x.replace('K','')) * 1000
        else:
            try:
                return float(x)
            except:
                return 0

    df['followers'] = df['followers'].apply(convert)
    df['views'] = df['views (avg.)'].apply(convert)
    df['likes'] = df['likes (avg.)'].apply(convert)
    df['comments'] = df['comments (avg.)'].apply(convert)

    # Remove unwanted columns
    df.drop(['views (avg.)','likes (avg.)','comments (avg.)','category','country'], axis=1, inplace=True)

    # Feature Engineering
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['followers']
    df.fillna(0, inplace=True)

    # Target
    df['is_influencer'] = df.apply(
        lambda row: 1 if row['followers'] >= 100000 and row['engagement_rate'] > 0.05 else 0,
        axis=1
    )

    X = df[['followers','views','likes','comments','engagement_rate']]
    y = df['is_influencer']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

model = load_model()

# ==============================
# User Input Section
# ==============================

st.subheader("Enter Data")

followers = st.number_input("Followers", min_value=1)
views = st.number_input("Average Views", min_value=0)
likes = st.number_input("Average Likes", min_value=0)
comments = st.number_input("Average Comments", min_value=0)

# ==============================
# Prediction
# ==============================

if st.button("Check Influencer"):

    engagement_rate = (likes + comments) / followers

    data = [[followers, views, likes, comments, engagement_rate]]

    result = model.predict(data)

    st.write(f"Engagement Rate: {engagement_rate:.4f}")

    if result[0] == 1:
        st.success("This person is an Influencer")
    else:
        st.error("This person is NOT an Influencer")

# ==============================
# Footer
# ==============================

st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")
