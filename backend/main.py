import os
import sys
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator

from src.logger import logger

# Global cache to hold the model instance
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Loads the ML model into memory before accepting HTTP requests
    and handles graceful shutdown.
    """
    log = logger.bind(service="fastapi_backend")
    model_path = os.getenv("MODEL_PATH", "/app/model")
    
    try:
        log.info("Starting up API and loading MLflow model", model_path=model_path)
        model_cache["model"] = mlflow.pyfunc.load_model(model_path)
        log.info("Model loaded successfully and ready for inference")
        yield
    except Exception as e:
        log.error("Failed to load model during startup", error=str(e))
        # Yield anyway to allow the app to fail gracefully or serve healthchecks
        yield 
    finally:
        log.info("Shutting down API, cleaning up resources")
        model_cache.clear()

# Initialize the API with the lifespan manager
app = FastAPI(title="Movie Sentiment Model API", lifespan=lifespan)

# Instrument the API to expose the /metrics endpoint for Prometheus observability
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

class DataframeSplit(BaseModel):
    """Pydantic schema representing the dataframe split format required by MLflow."""
    columns: List[str]
    data: List[List[str]]

class InferenceRequest(BaseModel):
    """Pydantic schema for the incoming inference payload."""
    dataframe_split: DataframeSplit


@app.get("/ping")
def ping() -> Dict[str, str]:
    """Healthcheck endpoint used by Kubernetes or Docker to verify the container is alive."""
    return {"status": "healthy"}


@app.post("/invocations")
def predict(request: InferenceRequest) -> Dict[str, List[Dict[str, str]]]:
    """
    Main inference endpoint. Extracts text from the MLflow-formatted request,
    runs the sentiment model, and returns the formatted prediction.
    """
    log = logger.bind(endpoint="/invocations")
    
    if "model" not in model_cache:
        log.error("Inference attempted but model is not loaded in memory")
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # Extract the text
        input_text = request.dataframe_split.data[0][0]
        log.info("Received inference request", text_length=len(input_text))
        
        # Run the model
        model = model_cache["model"]
        predictions = model.predict([input_text])
        
        # Format the output to exactly match what the frontend expects
        if hasattr(predictions, "to_dict"):
            formatted_preds = predictions.to_dict(orient="records")
        elif hasattr(predictions, "tolist") or isinstance(predictions, list):
            preds_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions
            formatted_preds = [{"sentiment": p} for p in preds_list]
        else:
            formatted_preds = [{"sentiment": str(predictions)}]
            
        log.info("Inference successful", prediction=formatted_preds[0])
        return {"predictions": formatted_preds}
        
    except Exception as e:
        log.error("Inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))