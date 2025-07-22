import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_tokens, d_model)
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class MoELayer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.d_model = config.n_embd
        self.n_experts = config.n_experts
        self.capacity_factor = config.capacity_factor
        self.k = config.k  # typically 1 or 2
        self.experts_weight = config.experts_weight
        self.router_weight = config.router_weight

        # Gate: projects hidden state to expert logits
        self.gate = nn.Linear(self.d_model, self.n_experts, bias=False)

        # Experts: each is a 2-layer feedforward network
        self.experts = nn.ModuleList(
            [
                Expert(self.d_model, 4 * self.d_model, config.dropout)
                for _ in range(self.n_experts)
            ]
        )

    def forward(self, H: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            H: Hidden states (B, T, d_model)
        Returns:
            H_out: (B, T, d_model)
            balance_loss: scalar tensor
        """
        B, T, d = H.shape
        S = B * T
        H_flat = H.view(S, d)  # (S, d_model)

        # 1. Compute gate logits & top-k indices
        gate_logits = self.gate(H_flat)  # (S, n_experts)
        if self.k == 1:
            topk_vals, topk_indices = torch.topk(gate_logits, k=1, dim=-1)  # (S, 1)
            # (S,)
            topk_indices = topk_indices.squeeze(-1)
            # (S, n_experts)
            gate_probs = F.softmax(gate_logits, dim=-1)
        else:
            # e.g., for k=2, get top 2
            topk_vals, topk_indices = torch.topk(
                gate_logits, k=self.k, dim=-1
            )  # (S, k)
            gate_probs = F.softmax(gate_logits, dim=-1)

        # 2. Compute capacity
        capacity = int((S / self.n_experts) * self.capacity_factor)

        # 3. For k=1: create one-hot routing
        if self.k == 1:
            dispatch_mask = F.one_hot(
                # (S, n_experts)
                topk_indices,
                num_classes=self.n_experts,
            ).float()
        else:
            # For k=2: create mask per expert; tokens may appear twice in dispatch
            dispatch_mask = torch.zeros((S, self.n_experts), device=H.device)
            for i in range(self.k):
                expert_idx = topk_indices[:, i]  # (S,)
                dispatch_mask.scatter_add_(
                    1, expert_idx.unsqueeze(1), torch.ones((S, 1), device=H.device)
                )
            # Note: dispatch_mask entries are 0/1 for k=1, or up to count of times selected.

        # 4. Compute load & importance for balancing
        router_loss = (
            self.router_weight * (torch.logsumexp(gate_logits, dim=-1) ** 2.0).mean()
        )
        load = dispatch_mask.mean(dim=0)  # (n_experts,) [150, 200, 250,...]
        importance = gate_probs.mean(dim=0)  # (n_experts,) [20, 5, 30, ...]
        print(f"Load: {load}, Importance: {importance}")
        balance_loss = self.experts_weight * self.n_experts * (load * importance).mean()
        print(f"Router loss: {router_loss.item()}, Balance loss: {balance_loss.item()}")
        moe_loss = router_loss + balance_loss

        # 5. Capacities

        # For each expert, find the tokens assigned and keep up to capacity
        positions_in_expert = torch.cumsum(dispatch_mask, dim=0)  # (S, n_experts)
        # (S, n_experts), Boolean
        in_capacity = positions_in_expert <= capacity

        # 6. Dispatch tokens to experts
        expert_inputs = []
        for e in range(self.n_experts):
            mask_e = (dispatch_mask[:, e] > 0) & (in_capacity[:, e])  # (S,)
            # (<=capacity, d)
            tokens_e = H_flat[mask_e]
            expert_inputs.append(tokens_e)

        # 7. Forward each expert
        expert_outputs = []
        for e in range(self.n_experts):
            if expert_inputs[e].shape[0] > 0:
                expert_out_e = self.experts[e](expert_inputs[e])  # (n_tokens_e, d)
            else:
                expert_out_e = torch.zeros((0, d), device=H.device)
            expert_outputs.append(expert_out_e)

        # 8. Combine expert outputs
        H_out_flat = torch.zeros_like(H_flat)  # (S, d)
        for e in range(self.n_experts):
            mask_e = (dispatch_mask[:, e] > 0) & (in_capacity[:, e])
            # indices of tokens processed by expert e
            idxs = mask_e.nonzero(as_tuple=False).squeeze(-1)
            H_out_flat[idxs] = expert_outputs[e]

        # 9. Reshape back to (B, T, d)
        H_out = H_out_flat.view(B, T, d)

        return H_out, moe_loss
