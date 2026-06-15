import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAdapter(nn.Module):
    def __init__(self, d_model, d_mid, dropout=0.1):
        """
        Args:
            d_model (int): the dimension of input tokens
            d_mid (int): the dimension after the down projection
        """
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.linear_proj = nn.Linear(d_mid, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_down = self.activation(self.down_proj(x))
        x_down = self.dropout(x_down)
        x_linear = self.activation(self.linear_proj(x_down))
        x_linear = self.dropout(x_linear)
        x_up = self.up_proj(x_linear)
        # x_up = self.up_proj(x_down)

        return x_up


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
    def __init__(self, d_model, num_experts, top_k, pool_type="avg", fixed_experts=0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.pool_type = pool_type
        self.fixed_experts = fixed_experts  # number of fixed selected experts while routing
        assert fixed_experts < top_k, "the number of fixed experts should be less than top_k"

        self.routing = nn.Linear(d_model, num_experts)

    def forward(self, x, spatial_shapes, level_start_index, patches):
        bs, _, c = x.shape

        kernel_size = int(math.sqrt(patches))
        stride = kernel_size
        assert kernel_size * kernel_size == patches, "The patch size should be a square number."

        # [bs, h*w, c] -> [bs, h, w, c]
        num_features = len(level_start_index)
        split_size_or_sections = [None] * num_features
        for i in range(num_features):
            if i < num_features - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = x.shape[1] - level_start_index[i]
        x_split = torch.split(x, split_size_or_sections, dim=1)

        patch_probs_list = []
        patch_indices_list = []
        for i, (h, w) in enumerate(spatial_shapes):
            # Get the feature map of the current scale
            x_level = x_split[i]  # [bs, h, w, c]
            x_level = x_level.view(bs, h, w, c).permute(0, 3, 1, 2)  # [bs, c, h, w]

            # Pooling the feature map
            if self.pool_type == "avg":
                x_pooled = F.avg_pool2d(x_level, kernel_size=kernel_size, stride=stride)  # [bs, c, h//ks, w//ks]

            elif self.pool_type == "max":
                x_pooled = F.max_pool2d(x_level, kernel_size=kernel_size, stride=stride)  # [bs, c, h//ks, w//ks]

            else:
                raise ValueError(f"Unsupported pool type: {self.pool_type}")

            x_pooled = x_pooled.permute(0, 2, 3, 1)  # [bs, h//ks, w//ks, c]

            # Calculate the routing probabilities
            routing_logits = self.routing(x_pooled)  # [bs, h//ks, w//ks, n_e]

            # Logic for handling fixed experts and dynamically selected experts
            if self.fixed_experts > 0:
                # Select the first N fixed experts
                fixed_indices = torch.arange(self.fixed_experts, device=routing_logits.device)
                fixed_indices = fixed_indices.expand(routing_logits.shape[0],
                                                     routing_logits.shape[1],
                                                     routing_logits.shape[2],
                                                     self.fixed_experts)

                # Select the remaining top_k-fixed_experts experts
                dynamic_k = self.top_k - self.fixed_experts

                if dynamic_k > 0:
                    # Mask out the already fixed selected experts
                    mask = torch.ones_like(routing_logits, dtype=torch.bool)
                    for e in range(self.fixed_experts):
                        mask[..., e] = False

                    # Select top_k from the remaining experts
                    masked_logits = routing_logits.masked_fill(~mask, float('-inf'))
                    dynamic_logits, dynamic_indices = masked_logits.topk(dynamic_k, dim=-1)

                    # Merge fixed and dynamic experts
                    top_k_logits = torch.cat([
                        torch.gather(routing_logits, -1, fixed_indices),
                        dynamic_logits
                    ], dim=-1)
                    top_k_indices = torch.cat([fixed_indices, dynamic_indices], dim=-1)
                else:
                    # If only using fixed experts
                    top_k_logits = torch.gather(routing_logits, -1, fixed_indices)
                    top_k_indices = fixed_indices
            else:
                # Original logic: directly select top_k experts
                top_k_logits, top_k_indices = routing_logits.topk(self.top_k, dim=-1)

            infs = torch.full_like(routing_logits, float("-inf"))
            sparse_logits = infs.scatter(dim=-1, index=top_k_indices, src=top_k_logits)
            probs = F.softmax(sparse_logits, dim=-1)  # [bs, h//ks, w//ks, n_e]

            # Upsample probs and indices to the original spatial shapes
            probs = F.interpolate(
                probs.permute(0, 3, 1, 2),  # [bs, n_e, h//ks, w//ks]
                size=(h, w),
                mode="nearest"  # Nearest neighbor interpolation to copy values
            ).permute(0, 2, 3, 1)  # [bs, h, w, n_e]

            top_k_indices = F.interpolate(
                top_k_indices.float().permute(0, 3, 1, 2),  # [bs, top_k, h//ks, w//ks]
                size=(h, w),
                mode="nearest"
            ).long().permute(0, 2, 3, 1)  # [bs, h, w, top_k]

            probs = probs.view(bs, -1, self.num_experts)  # [bs, h*w, n_e]
            top_k_indices = top_k_indices.view(bs, -1, self.top_k)  # [bs, h*w, top_k]

            patch_probs_list.append(probs)
            patch_indices_list.append(top_k_indices)

        # Concatenate the routing probabilities and indices
        probs = torch.cat(patch_probs_list, dim=1)  # [bs, H*W, n_e]
        top_k_indices = torch.cat(patch_indices_list, dim=1)  # [bs, H*W, top_k]

        return probs, top_k_indices


class CrossModalMoEAdapter(nn.Module):
    def __init__(self, d_model, experts, num_experts, top_k, patches):
        """
        Args:
            d_model (int): the dimension of input tokens
            experts (nn.ModuleList): the list of experts
            num_experts (int): the number of experts
            top_k (int): the number of experts to select
            patches (int): the number of patches
        """
        super().__init__()
        self.experts = experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.patches = patches

        self.router_r = TopKRouter(d_model, num_experts, top_k)
        self.router_t = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x, router, spatial_shapes, level_start_index):

        top_k_probs, top_k_indices = router(x, spatial_shapes, level_start_index, self.patches)

        expert_outputs = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert = self.experts[expert_idx]
            expert_output = expert(x)
            expert_outputs.append(expert_output.unsqueeze(1))

        expert_outputs = torch.cat(expert_outputs, dim=1)
        weighted_expert_output = (expert_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)

        return weighted_expert_output
