import torch
import torch.nn as nn
from typing import Dict, Any


# DQN policy netowrk
class Qnetwork(nn.Module):
    """
    Evaluate market observation status and output ezpected cumulative rewards,for wach available portforlio execution action 
    """

    def __init__(self, state_dim:int=5, action_dim:int=3):
        super().__init__()
        # project state feature 
        self.fc1 = nn.Linear(state_dim, 32)
        self.relu = nn.ReLU()

        # second hidden representation layer
        self.fc2 = nn.Linear(32,32)
        #predict 1 valies for 3 actions --> hold, buy, sell
        self.fc_out = nn.Linear(32,action_dim)

    def forword(self, state:torch.Tensor) -> torch.Tensor:
        x =self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)


# RL executon Agent Wrapper
class RLExecutoinAgent:
    """
    Executon agent that take feature signle and portfolio state, evaluate the q-network policy, and compute risk-agent 
    position sizing
    """
    ACTIONS = {0:"HOLD", 1:"BUY", 2:"SELL"}

    def __init__(self,model_path:str=None):

        # Define input state dimention 
        # [transformer_signal, onchain_signal, sentiment_signal, current_position_pct, cash_ratio]
        self.state_dim = 5
        self.action_dim=3

        # initialize QNetwork 
        self.policy_net = Qnetwork(self.state_dim, self.action_dim)

        if model_path:
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        self.policy_net.eval()


    def select_action(
        self,
        transformer_signal: float,
        onchain_signal: float,
        sentiment_signal: float,
        current_position_pct: float,
        cash_ratio: float
            
    ) -> Dict[str, Any]:
        """
        Executes policy inference given market signels and current account state.
        """

        # pack the 5 state dimentions into pytorch tensor
        state_list =[
            transformer_signal,
            onchain_signal,
            sentiment_signal,
            current_position_pct,
            cash_ratio
        ]

        state_tesnor = torch.tensor(state_list,dtype=torch.float32).unsqueeze(0)


        # Disanle gradient compution and caluclate Q-vlues
        with torch.no_grad():
            q_values= self.policy_net(state_tesnor).squeeze(0)
            best_action_idx = int(torch.argmax(q_values).item())

        action_name =self.ACTIONS[best_action_idx]

        # dYNAMIC POSITION SIZING BASED on signal consensus and q_value confidence 
        #composit score given signal strength
        composite_strength = (0.45*transformer_signal)+(0.35*onchain_signal)+(0.20*sentiment_signal)

        # calculate position size adgesment (presentage of portfolio balance to allocate)
        if action_name == "BUY":
            # Scale target allocation up to available cash ratio, capped at composite strength
            target_allocation_pct = min(cash_ratio * 100.0, max(10.0, abs(composite_strength) * 100.0))
        elif action_name == "SELL":
            # Sell partial or entire holding depending on negative signal strength
            target_allocation_pct = min(current_position_pct * 100.0, max(25.0, abs(composite_strength) * 100.0))
        else:
            target_allocation_pct = 0.0

        return {
            "action": action_name,
            "allocated_size_pct": round(target_allocation_pct, 2),
            "q_values": {
                "HOLD": round(float(q_values[0]), 4),
                "BUY": round(float(q_values[1]), 4),
                "SELL": round(float(q_values[2]), 4)
            },
            "composite_strength": round(composite_strength, 4)
        }

