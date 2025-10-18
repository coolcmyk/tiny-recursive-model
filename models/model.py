# Unofficial Base Implementation of TRM
# Author: coolcmyk@github.com
# References of good implementations:
# [1] https://github.com/SamsungSAILMontreal/TinyRecursiveModels
# [2] https://github.com/lucidrains/tiny-recursive-model


# Dictionaries
# x: the input embedding tensor, shape [batch, x_dim]
# y: latent answer vector, shape [batch, y_dim] (initialized to zeros or learned initial state)
# z: other latent state, shape [batch, z_dim] (same)
# net_z: a function/module that computes new z from (x, y, z)
# net_y: a function/module that computes new y from (y, z)
# output_head: maps y -> class logits (for cross-entropy)
# q_head: maps y -> single logit estimating "is prediction correct?" (for BCEWithLogitsLoss)
# y, z = latent recursion(x, y, z, n)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class TRM(nn.Module):
    # x: the input embedding tensor, shape [batch, x_dim]
    # y: latent answer vector, shape [batch, y_dim] (initialized to zeros or learned initial state)
    # z: other latent state, shape [batch, z_dim] (same)
    # net_z: a function/module that computes new z from (x, y, z)
    # net_y: a function/module that computes new y from (y, z)
    # output_head: maps y -> class logits (for cross-entropy)
    # q_head: maps y -> single logit estimating "is prediction correct?" (for BCEWithLogitsLoss)
    def __init__(
            self, 
            x: int, #Dimensions of x, y, and z
            y: int, 
            z: int,
            hidden_dim: int,
            num_classes: int,
            learnable: bool = False,
            init_scale: float = 1e-2,
            ):
        super().__init__()

        self.x_dim = x
        self.y_dim = y
        self.z_dim = z
        
        #net_z: (x,y,z) => z
        self.net_z = nn.Sequential(
            nn.Linear(x + y + z, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z)
        )

        #net_y: (y,z) => y
        self.net_y = nn.Sequential(
            nn.Linear(y + z, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y)
        )

        self.output_head = nn.Linear(y, num_classes)
        self.q_head = nn.Linear(y, 1)
    
    def latent_recursion(self, x, y, z, n=6):
        for _ in range(n):

            inp_z = torch.cat([x, y, z], dim=-1)
            z = self.net_z(inp_z)

            inp_y = torch.cat([y,z], dim=-1)
            y = self.net_y(inp_y)
        return y, z
    
    def deep_recursion(self, x, y, z, n=6, T=3):
        for _ in range(max(0, T-1)):
            with torch.no_grad():
                y, z = self.latent_recursion(x, y, z, n=n)
        
        y, z = self.latent_recursion(x, y, z, n=n)
        y_det, z_det = y.detach(), z.detach()

        y_logits = self.output_head(y)         # shape [batch, num_classes]
        q_logits = self.q_head(y).squeeze(-1)  # shape [batch]

        return (y_det, z_det), y_logits, q_logits