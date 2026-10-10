import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.pipeline.transformer import CryptoTransformer



# initialized the past api application 

app = FastAPI(title="Cric AI Stack", version="1.0")

# initialize pytorch gloablly so it stay in mem
FEATURE_DIM =5
D_model = 64
model = CryptoTransformer(feature_dim=FEATURE_DIM, d_model=D_model)

# put model evaluation mode 
model.eval()

# define the JSON schema for validation using pydantic 
class SequenceInput(BaseModel):
    sequence:List[List[float]]

@app.post("/predict/transformer")
def predict_transformer(data:SequenceInput):
    try:
        # convert json array to pytorch tensor
        tensor_data = torch.tensor(data.sequence, dtype=torch.float32)

        # validate tensor shape match feature dimentions 
        if  tensor_data.ndim != 2 or tensor_data.shape[1] != FEATURE_DIM:
            raise HTTPException(status_code=400, detail="iNVALID FEATURE DIMWNTIONS")

        # ADD A BATCH dimention (seq_len, features) -> 
        batch_input = tensor_data.unsqueeze(0)

        # disable gradient tracking to save and speed up inference 
        with torch.no_grad():
            predicted_value= model(batch_input).item()

        return {
            "status":"sucess",
            "prediction": predicted_value
        }

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))



    