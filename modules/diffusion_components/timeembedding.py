import torch
from torch import nn

"""
Time embedding for time step t. First, t is embedded using a fixed sinusoidal positional
embedding, as described in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762),
followed by a two layer MLP.
"""
class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int, pos_emb_dim: int, max_len: int = 5000):
        super().__init__()

        self.pos_emb_dim = pos_emb_dim
        self.time_emb_dim = time_emb_dim
        self.max_len = max_len

        # fixed sinusoidal positional embedding
        assert self.pos_emb_dim % 2 == 0, "Embedding dim must be a multiple of 2!"
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, self.pos_emb_dim, 2).float()
        pos_embedding = torch.zeros(self.max_len, self.pos_emb_dim)
        pos_embedding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.pos_emb_dim)))
        pos_embedding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.pos_emb_dim)))
        self.register_buffer('pos_embedding', pos_embedding, persistent=True)

        # MLP for time embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

    def forward(self, t: torch.Tensor):
        if self.pos_embedding.device != t.device:
            self.pos_embedding = self.pos_embedding.to(t.device)
        if next(self.mlp.parameters()).device != t.device:
            self.mlp = self.mlp.to(t.device)
        t_pos_emb = torch.index_select(self.pos_embedding, 0, t)
        t_emb = self.mlp(t_pos_emb)
        return t_emb