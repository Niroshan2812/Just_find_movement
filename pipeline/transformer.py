import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Unlike LSTMs tranformers process all data at once. Positional encoding injects timestamps into the data 
    so the model knows the order od the candles. 
    """

    def __init__ (self, d_model:int, max_len:int =5000):
        super().__init__()

        # Create metrix including 0 the protential encoding 

        pe = torch.zeros(max_len, d_model)
        # get position indices like - 0 to max_len
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # calculate the frequencies using sine and cosine functions 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # save the encoding metrix in the model's memory buffer
        self.register_buffer('pe', pe.unsqueeze(0))


    def forword(self,x:torch.Tensor) ->torch.Tensor:
        # Add the positional encoding mathemetically to the incoming price data 
        return x + self.pe[:, :x.size(1)]



class CryptoTransformer(nn.Module):

    """
    The main time-series model utilizing multi-head self-Attention 
    """

    def __init__ (self, feature_dim:int, d_model:int, nhead:int=4,num_layers:int=2):
        super().__init__()

        # input feature into a latger hidden space 
        self.input_projection = nn.Linear(feature_dim, d_model)

        # Add time context
        self.pos_encoder = PositionalEncoding(d_model)

        # Define the self-Attention layer --> Calculate relationship between different time steps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead= nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True # Ensure input shape is (batch, sequence_length, features)
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final output layer to compress the attention metrix down to a sigle price prediction 
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Expand dimesions
        x = self.input_projection(x)
        # inject temp order
        x=self.pos_encoder(x)
        # Attention netwoek weighs the inportance of past candles vs recent candles 
        attention_output = self.transformer_encoder(x)
        # Isolate the output of the very last time step in the sequence 
        last_step_representation = attention_output[:,-1,:]
        # predict the target 
        prediction = self.fc_out(last_step_representation)

        return prediction

    




