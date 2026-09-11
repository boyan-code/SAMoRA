#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import LoRALayer


class SALinear(nn.Linear, LoRALayer):
    # Semantic-aware mixture-of-LoRA-experts implemented in a dense layer
    orth_weight = 1e-3
    kl_weight = 1e-2

    def __init__(
        self,
        in_features: int,
        out_features: int,
        B_num: int,
        lambda_num: int,
        diagonal_format: bool,
        B_scale: float = 1.0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        tunable_scaler: bool = False,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        in_features : int. The number of input features
        out_features : int. The number of output features
        B_num : int. The number of B matrices (experts)
        lambda_num : int. The number of lambda matrices (e.g., task number)
        diagonal_format : bool. Whether the lambda matrices are diagonal
        B_scale : float, optional. Softmax temperature for the router
        r : int, optional. The rank of the LoRA decomposition
        lora_alpha : int, optional. The scaling factor for the LoRA decomposition
        lora_dropout : float, optional. The dropout rate for the LoRA decomposition
        fan_in_fan_out : bool, optional. Whether the layer stores the weight in fan_in, fan_out format
        tunable_scaler : bool, optional. Whether to use a tunable scaler
        """
        nn.Linear.__init__(self, in_features, out_features, bias=False, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            tunable_scaler=tunable_scaler,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.B_num = B_num
        self.lambda_num = lambda_num
        self.diagonal_format = diagonal_format
        if B_scale <= 0:
            raise ValueError(f"B_scale (router temperature) must be > 0, got {B_scale}")
        self.B_scale = B_scale

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_lambdas = nn.Parameter(self.weight.new_zeros((B_num, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((B_num, out_features, r)))
            self.lora_scale = nn.Parameter(self.weight.new_ones(r))
            self.lora_task_embedding = nn.Embedding(self.lambda_num, 8)
            self.lora_task_gate_weight = nn.Parameter(self.weight.new_zeros(8, 1))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_lambdas, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            if hasattr(self, "lora_task_gate_weight"):
                nn.init.kaiming_uniform_(self.lora_task_gate_weight, a=math.sqrt(5))

    def _base_forward(self, x: torch.Tensor):
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        return F.linear(x, weight, bias=self.bias)

    def _expert_forward(self, x: torch.Tensor):
        """Route the low-rank activations to the B experts. Returns (B, T, out_features)."""
        after_A = self.lora_dropout(x) @ self.lora_A.T
        norm_input = F.normalize(after_A, dim=-1)
        norm_expert = F.normalize(self.lora_lambdas, dim=-1)
        cos_sim = norm_input @ norm_expert.T
        router_weights = F.softmax(cos_sim / self.B_scale, dim=-1)

        after_A = after_A * self.lora_scale
        # Contract router weights with the low-rank activations first: the
        # (B, E, T, r) intermediate is far smaller than (B, T, E, out_features).
        weighted_A = torch.einsum("bte, btr -> betr", router_weights, after_A)
        after_B = torch.einsum("betr, ehr -> bth", weighted_A, self.lora_B)
        return after_B

    def _regularization_loss(self):
        """Orthogonality regularization on A/B plus router-expert alignment KL."""
        AtA = self.lora_A @ self.lora_A.T
        I_r = torch.eye(self.r, device=self.lora_A.device, dtype=self.lora_A.dtype)
        orth_A = F.mse_loss(AtA, I_r)

        BtB = torch.matmul(self.lora_B.transpose(-2, -1), self.lora_B)
        orth_B = F.mse_loss(BtB, I_r.expand(self.B_num, self.r, self.r))

        router_key_norm = F.normalize(self.lora_lambdas, dim=-1)
        expert_key_norm = F.normalize(self.lora_B.mean(dim=1), dim=-1)
        log_P = torch.log(F.softmax(router_key_norm, dim=-1) + 1e-8)
        Q = F.softmax(expert_key_norm, dim=-1)
        kl_loss = F.kl_div(log_P, Q, reduction="batchmean")

        return self.orth_weight * (orth_A + orth_B) + self.kl_weight * kl_loss

    def forward(self, x: torch.Tensor, lambda_index: torch.Tensor, statistics=None):
        if (lambda_index == -1).any():
            return self.forward_new(x, lambda_index, statistics)

        result = self._base_forward(x)
        if self.r > 0:
            after_B = self._expert_forward(x)

            task_emb = self.lora_task_embedding(lambda_index)
            gate = torch.sigmoid(task_emb @ self.lora_task_gate_weight)  # (B, 1)
            result += gate.unsqueeze(1) * after_B

            if self.training and statistics is not None:
                statistics.setdefault("adapter_loss", []).append(
                    self._regularization_loss()
                )

        return result

    def forward_new(self, x: torch.Tensor, lambda_index: torch.Tensor, statistics=None):
        """Forward for unseen tasks (lambda_index == -1): use the average task gate."""
        result = self._base_forward(x)
        if self.r > 0:
            after_B = self._expert_forward(x)

            all_task_gates = self.lora_task_embedding.weight @ self.lora_task_gate_weight
            gate = torch.sigmoid(all_task_gates.mean(dim=0))  # (1,)
            result += gate * after_B

        return result