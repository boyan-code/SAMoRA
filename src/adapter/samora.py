#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import LoRALayer, should_gather


class SALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        B_num: int,
        lambda_num: int,
        diagonal_format: bool,
        B_scale: float = 0.0,
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
        B_num : int. The number of B matrices
        lambda_num : int. The number of lambda matrices (e.g., task number)
        diagonal_format : bool. Whether the lambda matrices are diagonal
        B_scale : float, optional. The scale of the B matrices. (e.g., tenpearature)
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

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor, lambda_index: torch.Tensor, statistics=None):
        is_new_task = (lambda_index == -1)
        if is_new_task.any():
            return self.forward_new(x, lambda_index, statistics)
        
        def T(w):
            return w.T if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            lora_A = self.lora_A
            dropout_x = self.lora_dropout(x)
            after_A = dropout_x @ lora_A.T
            norm_input = F.normalize(after_A, dim=-1)
            
            norm_expert = F.normalize(
                self.lora_lambdas, dim=-1
            )
            
            cos_sim = norm_input @ norm_expert.T
            
            router_weights = F.softmax(
                cos_sim / self.B_scale, dim=-1
            )
            after_A = after_A @ torch.diag(self.lora_scale)
            expert_out = torch.einsum("btr, ehr -> bteh", after_A, self.lora_B)
            after_B = torch.einsum("bte, bteh -> bth", router_weights, expert_out)

            task_emb = self.lora_task_embedding(lambda_index)
            gate = torch.matmul(task_emb, self.lora_task_gate_weight)
            gate = torch.sigmoid(gate)
            
            gate = gate.unsqueeze(1)

            result += gate * (
                after_B)
            
            AtA = self.lora_A @ self.lora_A.T
            I_A = torch.eye(self.r, device=self.lora_A.device, dtype=self.lora_A.dtype)
            orth_A = F.mse_loss(AtA, I_A)
            
            B_transposed = self.lora_B.transpose(-2, -1)
            BtB = torch.matmul(B_transposed, self.lora_B)
            I_B_expanded = I_A.unsqueeze(0).expand(self.B_num, self.r, self.r)
            orth_B = F.mse_loss(BtB, I_B_expanded)
            
            router_key = self.lora_lambdas
            expert_key = self.lora_B.mean(dim=1)
            
            router_key_norm = F.normalize(router_key, dim=-1)
            expert_key_norm = F.normalize(expert_key, dim=-1)
            
            P = F.softmax(router_key_norm, dim=-1)
            log_P = torch.log(P + 1e-8)
            Q = F.softmax(expert_key_norm, dim=-1)
            
            kl_loss = F.kl_div(log_P, Q, reduction="batchmean")
            
            orth_weight = 1e-3
            kl_weight = 1e-2
            adapter_loss = orth_weight * (orth_A + orth_B) + kl_weight * kl_loss
            
            if statistics is not None:
                if "adapter_loss" not in statistics:
                    statistics["adapter_loss"] = []
                statistics["adapter_loss"].append(adapter_loss)
        
        return result

    def forward_new(self, x: torch.Tensor, lambda_index: torch.Tensor, statistics=None):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            lora_A = self.lora_A
            dropout_x = self.lora_dropout(x)
            after_A = dropout_x @ lora_A.T
            norm_input = F.normalize(after_A, dim=-1)
            
            norm_expert = F.normalize(
                self.lora_lambdas, dim=-1
            )
            
            cos_sim = norm_input @ norm_expert.T
            
            router_weights = F.softmax(
                cos_sim / self.B_scale, dim=-1
            )
            after_A = after_A @ torch.diag(self.lora_scale)
            expert_out = torch.einsum("btr, ehr -> bteh", after_A, self.lora_B)
            after_B = torch.einsum("bte, bteh -> bth", router_weights, expert_out)

            all_task_gates = torch.matmul(self.lora_task_embedding.weight, self.lora_task_gate_weight)
            avg_gate = all_task_gates.mean(dim=0)
            gate = torch.sigmoid(avg_gate)
            
            batch_size = x.shape[0]
            gate = gate.unsqueeze(0).unsqueeze(1)
            gate = gate.expand(batch_size, -1, -1)

            result += gate * after_B
        
        return result