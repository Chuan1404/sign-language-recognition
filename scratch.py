import torch
from src.models.SLT_model import ISLR_Transformer_Selector

model = ISLR_Transformer_Selector(num_classes=100)
features = torch.randn(2, 50, 6*27) # 6 * _NUM_NODE (27) = 162
# wait, wait, the feature size is B, T, num_nodes*3*2 ? Wait, what is _NUM_NODE and _COORD_DIM?
