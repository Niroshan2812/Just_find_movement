import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.pipeline.transformer import CryptoTransformer
from app.pipeline.onchain import OnChianXGBoostModel



# initialized the past api application 

app = FastAPI(title="Cric AI Stack", version="1.0")

# initialize pytorch gloablly so it stay in mem

# import XGBoost models also 
FEATURE_DIM =5
D_model = 64

transformaer_model = CryptoTransformer(feature_dim=FEATURE_DIM, d_model=D_model)

# put model evaluation mode 
transformaer_model.eval()

# todo - pass this model path ="model.json " when have an real model 
onchain_model = OnChianXGBoostModel()



# define the JSON schema for validation using pydantic 
class SequenceInput(BaseModel):
    sequence:List[List[float]]

class OnchainInput(BaseModel):
    metrics:List[float]


@app.post("/predict/transformer")
def predict_transformer(data: SequenceInput):
    
    # Validates and converts input into a tensor.
    # Runs the Attention network to predict price movement.
    try:
        tensor_data = torch.tensor(data.sequence, dtype=torch.float32)
        if tensor_data.ndim != 2 or tensor_data.shape[1] != FEATURE_DIM:
            raise HTTPException(status_code=400, detail="Invalid feature dimensions.")
        
        batch_input = tensor_data.unsqueeze(0)
        with torch.no_grad():
            predicted_value = transformaer_model(batch_input).item()
            
        return {"status": "success", "prediction": predicted_value}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/predict/onchain")
def predict_onchain(data: OnChainInput):
    
    # Takes the JSON array of on-chain metrics from the request.
    # Passes it to the XGBoost wrapper.
    # Returns a normalized market health score.
    try:
        if len(data.metrics) != 3:
            raise HTTPException(status_code=400, detail="Expected 3 on-chain metrics.")
            
        health_score = onchain_model.predict(data.metrics)
        
        return {
            "status": "success",
            "onchain_health_score": health_score
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))