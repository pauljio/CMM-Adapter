import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAdapter(nn.Module):
    def __init__(self, d_model, d_mid, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        return x


class Expert(nn.Module):
    def __init__(self, d_model, d_mid, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_down = self.activation(self.down_proj(x))
        x_down = self.dropout(x_down)
        x_up = self.up_proj(x_down)

        return x_up


class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.routing = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Calculate the routing probabilities
        routing_logits = self.routing(x)
        top_k_logits, top_k_indices = routing_logits.topk(self.top_k, dim=-1)

        infs = torch.full_like(routing_logits, float("-inf"))
        sparse_logits = infs.scatter(dim=-1, index=top_k_indices, src=top_k_logits)
        probs = F.softmax(sparse_logits, dim=-1)                                    

        return probs, top_k_indices


class SelfMoEAdapter(nn.Module):
    def __init__(self, d_model, d_mid, num_experts, top_k):
        """
        Args:
            d_model (int): the dimension of input tokens
            d_mid (int): the dimension of adapter hidden layer
            num_experts (int): the number of experts
            top_k (int): the number of experts to select
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(Expert(d_model, d_mid, dropout=0.1) for _ in range(num_experts))

    def forward(self, x):

        probs, top_k_indices = self.router(x)
        # flatten to concat every batch
        x_flatten = x.view(-1, x.shape[-1])
        probs_flatten = probs.view(-1, probs.shape[-1])

        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_output = expert(x_flatten[flat_mask])
                gating_scores = probs_flatten[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output

