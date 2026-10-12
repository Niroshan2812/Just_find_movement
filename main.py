import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from app.pipeline.transformer import CryptoTransformer
from app.pipeline.onchain import OnChianXGBoostModel
from app.pipeline.sentiment import CryptoSentimentEngine
from app.pipeline.execution_rl import RLExecutoinAgent



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

# load finbert model into memory
sentiment_engine = CryptoSentimentEngine()

# initialize RL aget 
rl_agent= RLExecutoinAgent()

# define the JSON schema for validation using pydantic 
class SequenceInput(BaseModel):
    sequence:List[List[float]]

class OnchainInput(BaseModel):
    metrics:List[float]

class SetimentInput(BaseModel):
    headlines:List[str]

class RLActionIput(BaseModel):
    transformer_signal: float = Field(..., description="Predicted delta from Milestone 1")
    onchain_signal: float = Field(..., description="Health score from Milestone 2 (-1.0 to 1.0)")
    sentiment_signal: float = Field(..., description="NLP score from Milestone 3 (-1.0 to 1.0)")
    current_position_pct: float = Field(0.0, ge=0.0, le=1.0, description="Current crypto allocation fraction (0.0 to 1.0)")
    cash_ratio: float = Field(1.0, ge=0.0, le=1.0, description="Available liquidity/cash fraction (0.0 to 1.0)")


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



@app.post("/predict/sentiment")
def predict_setiment(data:SetimentInput):
        # Take json array of the news Strings
        # Passes tem to finBeart for anelysis
        #Returns the aggregted numarical score

        try:
            if not data.headlines:
                raise HTTPException(status_code =400, detail = "Healine LIST CANNT BE empty")

            aggregate_sentiment = sentiment_engine.analyse_text(data.headlines)

            return {
                "status": "success",
                "finbert_sentiment_score": aggregate_sentiment,
                "articles_processed": len(data.headlines)
            }
        except Exception as err:
            raise HTTPException(status_code=500, detail=str[err])



# --- RL Execution Endpoint ---
@app.post("/execute/rl-action")
def determine_execution_action(payload: RLActionIput):
    """
    Accepts signals from prior milestones along with current portfolio state,
    and returns the RL agent's decision and trade allocation percentage.
    """
    try:
        decision = rl_agent.select_action(
            transformer_signal=payload.transformer_signal,
            onchain_signal=payload.onchain_signal,
            sentiment_signal=payload.sentiment_signal,
            current_position_pct=payload.current_position_pct,
            cash_ratio=payload.cash_ratio
        )
        return {
            "status": "success",
            "decision": decision
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))