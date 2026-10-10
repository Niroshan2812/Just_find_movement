import numpy as np
import xgboost as xgb

class OnChianXGBoostModel:
    """
    Evaluate tabuler onchain data
    """

    def __init__(self, model_path: str =None):

        # check per exist onchain is provided 
        # if true -> load that model 
        # else -> initialize a flag to use dummy weighet for testing 

        self.use_mock = False
        if model_path:
            self.model = xgb.Booster()
            self.model.load_model(model_path)

        else:
            self.use_mock =True

            # mock weight = [Funding Rate, Net Exchange Inflow, Whale Transactions]

            self.mock_weights =np.array([-0.5, -0.2, 0.7])


    def predict(self, features:list) -> float:

        # convert raw python list into numpy array
        # if real model --> True - convert to DMatrix
        # run inference and return the score 
        # if real model == false -> appy mathemetical dot product 

        np_features = np.array(features)

        if not self.use_mock:
            dmatrix = xgb.DMatrix([np_features])
            prediction = self.model.predict(dmatrix)[0]
            return float(prediction)
        else:
            # mock inference calculation
            raw_score = np.dot(np_features, self.mock_weights)
            # bound the score between -1.0 --> bearish ? 1.0 bullish
            normalized_score = float(np.tanh(raw_score))
            return normalized_score