import streamlit as st
import requests

# The URL of your MLflow model container (we will configure this in Docker Compose)
API_URL = "http://model_backend:8080/invocations"

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to see if it is positive or negative.")

# Text input area
user_review = st.text_area("Review text:", height=150, placeholder="e.g., The cinematography was amazing, but the plot was so boring...")

if st.button("Analyze Sentiment"):
    if user_review.strip():
        with st.spinner("Analyzing..."):
            # Format the payload exactly as MLflow PyFunc expects
            payload = {
                "dataframe_split": {
                    "columns": ["text"],
                    "data": [[user_review]]
                }
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result["predictions"][0]["sentiment"]
                    
                    if sentiment == "positive":
                        st.success(f"**Positive!** 🎉")
                    else:
                        st.error(f"**Negative.** 📉")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the model backend. Is the container running?")
    else:
        st.warning("Please enter a review first.")