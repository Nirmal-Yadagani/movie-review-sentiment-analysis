from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize the API
app = FastAPI(title="Movie Sentiment Model API")

# 1. Load the MLflow Model from the local folder into memory
try:
    model = mlflow.pyfunc.load_model("/app/model")
except Exception as e:
    print(f"Error loading model: {e}")

# 2. Instrument the API to expose the /metrics endpoint for Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# 3. Define the exact payload structure that MLflow's PyFunc client sends
class DataframeSplit(BaseModel):
    columns: List[str]
    data: List[List[str]]

class InferenceRequest(BaseModel):
    dataframe_split: DataframeSplit

# --- Healthcheck Endpoint ---
@app.get("/ping")
def ping():
    return {"status": "healthy"}

# 4. Create the prediction endpoint
@app.post("/invocations")
def predict(request: InferenceRequest):
    try:
        # Extract the text
        input_text = request.dataframe_split.data[0][0]
        
        # Run the model
        predictions = model.predict([input_text])
        
        # --- THE FIX: Format the output to exactly match what the frontend expects ---
        
        # If the model returns a Pandas DataFrame, convert it to a list of records
        if hasattr(predictions, "to_dict"):
            formatted_preds = predictions.to_dict(orient="records")
            
        # If the model returns a Numpy array or list (e.g., ["Positive"])
        elif hasattr(predictions, "tolist") or isinstance(predictions, list):
            preds_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions
            # Wrap the raw string inside a dictionary with the "sentiment" key
            formatted_preds = [{"sentiment": p} for p in preds_list]
            
        # Fallback for single raw strings
        else:
            formatted_preds = [{"sentiment": str(predictions)}]
            
        return {"predictions": formatted_preds}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))