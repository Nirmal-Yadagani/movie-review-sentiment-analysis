import os
import requests
import streamlit as st
import structlog

# Configure Structlog to output JSON
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger(service="streamlit_frontend")

BACKEND_BASE_URL = os.environ.get("BACKEND_URL", "http://model_backend:8080")
API_URL = f"{BACKEND_BASE_URL}/invocations"

def main() -> None:
    st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")
    st.title("🎬 Movie Review Sentiment Analyzer")

    user_review = st.text_area("Review text:", height=150)

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
                    # Structlog: Pass variables as kwargs, not f-strings!
                    logger.info("backend_request", api_url=API_URL, payload_size=len(user_review))
                    response = requests.post(API_URL, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        sentiment = result["predictions"][0]["sentiment"]
                        
                        logger.info("backend_response_success", sentiment=sentiment)
                        
                        if sentiment.lower() == "positive":
                            st.success("**Positive!** 🎉")
                        else:
                            st.error("**Negative.** 📉")
                    else:
                        logger.error("backend_response_error", status_code=response.status_code, error_detail=response.text)
                        st.error(f"API Error: {response.status_code}")
                        
                except requests.exceptions.ConnectionError as e:
                    logger.error("backend_connection_failed", error=str(e))
                    st.error("Failed to connect to the backend. Is the container running?")
        else:
            st.warning("Please enter a review first.")

if __name__ == "__main__":
    main()