import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.reformer.configuration_reformer import ReformerConfig
from transformers.models.reformer.modeling_reformer import LSHSelfAttention


class LSHAttention(LSHSelfAttention):
    def __init__(self, d_model, n_heads, dropout=0, max_seq_len=1024):
        self.d_output = d_model
        reformer_config = ReformerConfig()
        reformer_config.attn_layers = ["lsh"]
        reformer_config.is_decoder = False
        reformer_config.max_position_embeddings = max_seq_len
        reformer_config.hidden_size = d_model
        reformer_config.num_attention_heads = n_heads
        reformer_config.attention_head_size = int(d_model / n_heads)

        super().__init__(reformer_config)
        self.query_key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None, state=None):
        b, l, d = X.shape

        X = F.pad(X, (0, 0, 0, self.max_position_embeddings - l, 0, 0))
        output = super().forward(hidden_states=X, attention_mask=mask).hidden_states
        #output = output[:, :l, :]

        output = F.pad(output, (0, 0, 0, l-self.max_position_embeddings, 0, 0))
        return output, None
