import os
import requests
import streamlit as st

from src.logger import logger

# Fetch environment variables with fallbacks
BACKEND_BASE_URL = os.environ.get("BACKEND_URL", "http://model_backend:8080")
API_URL = f"{BACKEND_BASE_URL}/invocations"

def main() -> None:
    """
    Renders the Streamlit frontend and handles API interaction with the FastAPI backend.
    """
    # Bind UI logs for observability
    log = logger.bind(service="streamlit_frontend")
    
    st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.write("Enter a movie review below to see if it is positive or negative.")

    user_review = st.text_area(
        "Review text:", 
        height=150, 
        placeholder="e.g., The cinematography was amazing, but the plot was so boring..."
    )

    if st.button("Analyze Sentiment"):
        if user_review.strip():
            with st.spinner("Analyzing..."):
                payload = {
                    "dataframe_split": {
                        "columns": ["text"],
                        "data": [[user_review]]
                    }
                }
                
                try:
                    log.info("Sending payload to backend", api_url=API_URL)
                    response = requests.post(API_URL, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        sentiment = result["predictions"][0]["sentiment"]
                        
                        log.info("Received valid response", sentiment=sentiment)
                        if sentiment.lower() == "positive":
                            st.success("**Positive!** 🎉")
                        else:
                            st.error("**Negative.** 📉")
                    else:
                        error_msg = f"API Error: {response.status_code} - {response.text}"
                        log.error("Backend returned an error", status_code=response.status_code, error=response.text)
                        st.error(error_msg)
                        
                except requests.exceptions.ConnectionError as e:
                    log.error("Failed to connect to backend", error=str(e))
                    st.error(f"Failed to connect to the backend at {API_URL}. Is the container running?")
        else:
            st.warning("Please enter a review first.")

if __name__ == "__main__":
    main()