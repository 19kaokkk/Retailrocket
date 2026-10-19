"""MB-SASRec (add), adapted from the user's ML3 Big (1).py.

Architecture and initialization match SASRecBackbone/MultiBehaviorSASRec.
The training loop, client partitioning, and loss normalization live elsewhere.
"""
import torch
from torch import nn
from torch.nn import functional as F

BEHAVIOR2ID = {'view': 1, 'addtocart': 2, 'transaction': 3}


class SASRecBackbone(nn.Module):
    def __init__(self, num_items, hidden_units=50, maxlen=50, num_blocks=2,
                 num_heads=1, dropout_rate=0.5):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.num_items, self.hidden_units, self.maxlen = num_items, hidden_units, maxlen
        self.num_blocks, self.num_heads = num_blocks, num_heads
        self.head_dim = hidden_units // num_heads
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.attn_norms, self.q_layers = nn.ModuleList(), nn.ModuleList()
        self.k_layers, self.v_layers = nn.ModuleList(), nn.ModuleList()
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ffn_norms, self.ffn_linear1, self.ffn_linear2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.ffn_dropout1, self.ffn_dropout2 = nn.Dropout(dropout_rate), nn.Dropout(dropout_rate)
        for _ in range(num_blocks):
            self.attn_norms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.q_layers.append(nn.Linear(hidden_units, hidden_units))
            self.k_layers.append(nn.Linear(hidden_units, hidden_units))
            self.v_layers.append(nn.Linear(hidden_units, hidden_units))
            self.ffn_norms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.ffn_linear1.append(nn.Linear(hidden_units, hidden_units))
            self.ffn_linear2.append(nn.Linear(hidden_units, hidden_units))
        self.final_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        for layers in [self.q_layers, self.k_layers, self.v_layers, self.ffn_linear1, self.ffn_linear2]:
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for norm in list(self.attn_norms) + list(self.ffn_norms) + [self.final_norm]:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.hidden_units)

    def _embed(self, input_seq, beh_seq):
        return self.item_emb(input_seq) * self.hidden_units**0.5

    def forward(self, input_seq, beh_seq=None):
        _, length = input_seq.shape
        valid_mask = input_seq != 0
        valid_float = valid_mask.unsqueeze(-1).float()
        positions = torch.arange(length, device=input_seq.device).unsqueeze(0)
        x = self.input_dropout(self._embed(input_seq, beh_seq) + self.pos_emb(positions)) * valid_float
        for block in range(self.num_blocks):
            q_input = self.attn_norms[block](x)
            q = self._split_heads(self.q_layers[block](q_input))
            k = self._split_heads(self.k_layers[block](x))
            v = self._split_heads(self.v_layers[block](x))
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
            scores = scores.masked_fill((~valid_mask).unsqueeze(1).unsqueeze(1), -2**32 + 1)
            scores = scores.masked_fill(self.causal_mask[:length, :length].unsqueeze(0).unsqueeze(0), -2**32 + 1)
            weights = self.attn_dropout(torch.softmax(scores, dim=-1))
            x = (self._merge_heads(torch.matmul(weights, v)) + q_input) * valid_float
            ffn_input = self.ffn_norms[block](x)
            ffn = self.ffn_dropout1(F.relu(self.ffn_linear1[block](ffn_input)))
            ffn = self.ffn_dropout2(self.ffn_linear2[block](ffn))
            x = (ffn + ffn_input) * valid_float
        return self.final_norm(x) * valid_float


class MultiBehaviorSASRec(SASRecBackbone):
    def __init__(self, num_items, **kwargs):
        super().__init__(num_items, **kwargs)
        self.beh_emb = nn.Embedding(4, self.hidden_units, padding_idx=0)
        nn.init.xavier_uniform_(self.beh_emb.weight)
        with torch.no_grad():
            self.beh_emb.weight[0].zero_()

    def _embed(self, input_seq, beh_seq):
        if beh_seq is None:
            raise ValueError('The multi-behavior model requires behavior IDs')
        return self.item_emb(input_seq) * self.hidden_units**0.5 + self.beh_emb(beh_seq)

    def candidate_scores(self, items, behaviors, candidates):
        last = self(items, behaviors)[:, -1, :]
        return torch.sum(last.unsqueeze(1) * self.item_emb(candidates), dim=-1)


def per_user_loss(model, seq, beh, positive, negative):
    hidden = model(seq, beh)
    pos_logits = torch.sum(hidden * model.item_emb(positive), dim=-1)
    neg_logits = torch.sum(hidden * model.item_emb(negative), dim=-1)
    mask = positive != 0
    terms = -F.logsigmoid(pos_logits) - F.logsigmoid(-neg_logits)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError('A training user has no valid next-event targets')
    return (terms * mask).sum(dim=1) / counts


def proximal_penalty(model, anchor, mu):
    if mu < 0:
        raise ValueError('mu must be nonnegative')
    return 0.5 * mu * sum((p - anchor[name]).square().sum()
        for name, p in model.named_parameters())
