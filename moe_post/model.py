import math
from torch.nn import functional as F
import torch
from torch import nn


class MLPExperts(nn.Module):
    def __init__(
        self,
        d,
        n_exp=8,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()
        self.bias = bias
        self.c_fc = nn.Parameter(torch.empty(n_exp, d, 4 * d))
        self.c_proj = nn.Parameter(torch.empty(n_exp, 4 * d, d))
        self.fc_bias = nn.Parameter(torch.empty(n_exp, 1, 4 * d)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(n_exp, 1, d)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x


class BasicSoftmaxRouter(nn.Module):
    def __init__(
        self,
        d,
        n_exp=8,
        top_k=2,
        use_noisy_top_k=True,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        """

        super().__init__()

        # router settings
        self.top_k = top_k
        assert self.top_k >= 1 and self.top_k <= n_exp
        self.use_noisy_top_k = use_noisy_top_k

        # linear projection for (noisy) softmax routing
        # no bias used, see page 4 eq (4) in https://arxiv.org/abs/1701.06538
        self.w_g = nn.Linear(d, n_exp, bias=False)
        self.w_noise = nn.Linear(d, n_exp, bias=False) if self.use_noisy_top_k else None

    def forward(self, x):
        # eq (4) in https://arxiv.org/abs/1701.06538
        logits = self.w_g(x)  # [B, C, d] -> [B, C, n_exp]
        if self.use_noisy_top_k:
            # (optionally) add noise into the router
            noise = F.softplus(self.w_noise(x))
            noise *= torch.randn_like(noise)
            logits += noise
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [B, C, k]
        return top_k_logits, top_k_indices


class Router(nn.Module):
    def __init__(
        self,
        d,
        n_exp=8,
        top_k=2,
        use_noisy_top_k=True,
        capacity_factor=1.25,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        """

        super().__init__()

        self.d = d
        self.n_exp = n_exp
        self.top_k = top_k
        assert self.top_k >= 1 and self.top_k <= n_exp
        self.use_noisy_top_k = use_noisy_top_k
        self.capacity_factor = capacity_factor
        self.w_g = nn.Linear(d, n_exp, bias=False)
        self.w_noise = nn.Linear(d, n_exp, bias=False) if self.use_noisy_top_k else None

    def forward(self, x):
        # get the total number of tokens in the batch
        B, C, _ = x.size()
        num_tokens = B * C

        # eq (4) in https://arxiv.org/abs/1701.06538
        logits = self.w_g(x)  # [B, C, d] -> [B, C, n_exp]
        if self.use_noisy_top_k:
            # (optionally) add noise into the router
            noise = F.softplus(self.w_noise(x))
            noise *= torch.randn_like(noise)
            logits += noise

        # top-K expert selection, compute probabilities over active experts
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [B, C, K]
        router_probs = torch.full_like(logits, float("-inf"))  # [B, C, n_exp]
        router_probs.scatter_(-1, top_k_indices, top_k_logits)
        router_probs = F.softmax(router_probs, dim=-1)

        # compute the expert capacity
        exp_capacity = math.floor(
            self.top_k * self.capacity_factor * num_tokens / self.n_exp
        )
        exp_capacity += exp_capacity % 2  # make sure expert capacity is an even integer
        exp_capacity = int(exp_capacity)

        # make a multi-hot mask of chosen experts
        # values are 0 if expert not chosen, 1 if expert chosen
        # [B, C, K, n_exp]
        exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)
        exp_mask = exp_mask.view(
            num_tokens, self.top_k, self.n_exp
        )  # [B * C, K, n_exp]
        exp_mask = exp_mask.permute(1, 0, 2)  # [K, B * C, n_exp]

        # compute index for each token in expert batch
        # NOTE: cumsum counts top-1 first, top-2 second, etc.
        # to prioritize top experts when dropping tokens
        exp_rank = exp_mask.reshape(
            self.top_k * num_tokens, self.n_exp
        )  # [K * B * C, n_exp]
        # cumsum of expert selections [K * B * C, n_exp]
        exp_rank = torch.cumsum(exp_rank, dim=0) - 1
        exp_rank = exp_rank.reshape(
            self.top_k, num_tokens, self.n_exp
        )  # [K, B * C, n_exp]

        # mask entries beyond expert capacity and compute used capacity
        exp_mask *= torch.lt(exp_rank, exp_capacity)  # [K, B * C, n_exp]

        # matrix storing token position in batch of corresponding expert
        exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [K, B * C]

        # mask probabilities to only include selected experts
        router_probs = router_probs.view(num_tokens, self.n_exp)[
            None, :
        ]  # [1, B * C, n_exp]
        exp_weights = exp_mask * router_probs  # [K, B * C, n_exp]

        # position of each token within the capacity of the selected expert
        # [K, B * C, exp_capacity]
        exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity)

        # weight of selected expert for each token at position the capacity of that expert
        exp_weights = torch.sum(
            exp_weights.unsqueeze(
                # [B * C, n_exp, exp_capacity]
                3
            )
            * exp_rank_sc.unsqueeze(2),
            dim=0,
        )
        exp_mask = exp_weights.bool()  # binary mask of selected experts for each token

        # reshape tokens into batches for each expert, return both weights and batches
        # [n_exp, exp_capacity, B * C] * [B * C, d] -> [n_exp, exp_capacity, n_embd]
        x = x.view(num_tokens, self.d)
        expert_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x
        return exp_weights, exp_mask, expert_batches


class MOELayer(nn.Module):
    def __init__(
        self,
        d,
        n_exp=8,
        top_k=2,
        use_noisy_top_k=True,
        capacity_factor=1.25,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()
        self.router = Router(  # (noisy) top k router
            d=d,
            n_exp=n_exp,
            top_k=top_k,
            use_noisy_top_k=use_noisy_top_k,
            capacity_factor=capacity_factor,
        )
        self.experts = MLPExperts(  # group of MLPs (experts)
            d=d,
            n_exp=n_exp,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        B, C, d = x.size()  # track original shape of input
        num_tokens = B * C

        # pass each token through the router
        exp_weight, exp_mask, exp_batches = self.router(x)

        # compute expert output
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, d]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # [B * C, n_exp * exp_capacity]
        exp_weight = exp_weight.view(num_tokens, -1)
        exp_out = exp_out.view(-1, d)  # [n_exp * exp_capacity, d]
        output = exp_weight @ exp_out  # [B * C, d]

        # resize output before return
        return output.view(B, T, d)


class MoEBlock(nn.Module):
    def __init__(
        self,
        d,
        H,
        C,
        n_exp,
        top_k,
        use_noisy_top_k=True,
        capacity_factor=1.25,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        C: maximum length of input sequences (in tokens)
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, H, T, bias, dropout)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = MOELayer(
            d,
            n_exp,
            top_k,
            use_noisy_top_k,
            capacity_factor,
            bias,
            dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
