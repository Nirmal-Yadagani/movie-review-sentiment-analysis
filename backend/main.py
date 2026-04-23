import os
import sys
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

# Configure Structlog for the backend
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger(service="fastapi_backend")

model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "/app/model")
    try:
        logger.info("startup_initiating", model_path=model_path)
        model_cache["model"] = mlflow.pyfunc.load_model(model_path)
        logger.info("model_loaded_successfully")
        yield
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        yield 
    finally:
        logger.info("shutdown_initiating")
        model_cache.clear()

app = FastAPI(title="Movie Sentiment Model API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

class DataframeSplit(BaseModel):
    columns: List[str]
    data: List[List[str]]

class InferenceRequest(BaseModel):
    dataframe_split: DataframeSplit

@app.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/invocations")
def predict(request: InferenceRequest) -> Dict[str, List[Dict[str, str]]]:
    if "model" not in model_cache:
        logger.error("inference_rejected", reason="model_unavailable")
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        input_text = request.dataframe_split.data[0][0]
        logger.info("inference_started", text_length=len(input_text))
        
        model = model_cache["model"]
        predictions = model.predict([input_text])
        
        if hasattr(predictions, "to_dict"):
            formatted_preds = predictions.to_dict(orient="records")
        elif hasattr(predictions, "tolist") or isinstance(predictions, list):
            preds_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions
            formatted_preds = [{"sentiment": p} for p in preds_list]
        else:
            formatted_preds = [{"sentiment": str(predictions)}]
            
        logger.info("inference_completed", prediction=formatted_preds[0])
        return {"predictions": formatted_preds}
        
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))