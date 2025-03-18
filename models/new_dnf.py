from enum import Enum
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Dict, Any, List
class SemiSymbolicLayerType(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"

# push the weight of SemiSymbolic to 0, -6, 6
class SemiSymbolic(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal",
        epsilon: float = 0.001,
    ) -> None:
        super(SemiSymbolic, self).__init__()

        self.layer_type = layer_type

        self.in_features = in_features  # P
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if weight_init_type == "normal":
            # nn.init.uniform_(self.weights, a=-6, b=6)
            nn.init.normal_(self.weights, mean=0.0, std=0.1)
        else:
            nn.init.uniform_(self.weights, a=-6, b=6)
        self.delta = delta

        # For DNF min
        self.epsilon = epsilon

    def forward(self, input: Tensor) -> Tensor:

        b = input.size(0)
        # Input: N x P
        abs_weight = torch.abs(input.unsqueeze(2).expand(-1, -1, self.out_features)*self.weights.T.unsqueeze(0).expand(b,-1,-1))
        # abs_weight: N, P, Q
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # max_abs_w: N, Q

        # nonzero_weight = torch.where(
        #     abs_weight > self.epsilon, abs_weight.double(), 100.0
        # )
        # nonzero_min = torch.min(nonzero_weight, dim=1)[0]

        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w:  N, Q
        if self.layer_type == SemiSymbolicLayerType.CONJUNCTION:
            bias = max_abs_w - sum_abs_w
            # bias = nonzero_min - sum_abs_w
        else:
            bias = sum_abs_w - max_abs_w
            # bias = sum_abs_w - nonzero_min
        # bias: N, Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum  # .float()

# 语义增强
class SemanticSemiSymbolic(SemiSymbolic):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 layer_type: SemiSymbolicLayerType,
                 delta: float,
                 predicate_embeddings: torch.Tensor,  # 新增语义嵌入
                 use_attention=True,  # 新增注意力机制
                 **kwargs):
        super().__init__(in_features, out_features, layer_type, delta, **kwargs)
        
        # 语义增强参数
        self.embedding_dim = predicate_embeddings.size(1)
        self.predicate_emb = nn.Parameter(predicate_embeddings, requires_grad=False)
        self.use_attention = use_attention
        
        if self.use_attention:
            # 注意力变换矩阵
            self.attention = nn.Linear(self.embedding_dim, out_features)
            self.attention_norm = nn.LayerNorm(out_features)
    
    def semantic_attention(self, inputs: Tensor) -> Tensor:
        """基于语义的注意力调节"""
        # inputs: [B, P]
        batch_size = inputs.size(0)
        
        # 计算语义相似度 [B, P, Q]
        emb = self.predicate_emb.unsqueeze(0).expand(batch_size, -1, -1)
        attn_scores = self.attention(emb)  # [B, P, Q]
        attn_scores = self.attention_norm(attn_scores)
        
        # 与输入激活值结合
        return torch.sigmoid(attn_scores) * inputs.unsqueeze(-1)

    def forward(self, input: Tensor) -> Tensor:
        # 原始逻辑计算
        b = input.size(0)
        abs_weight = torch.abs(
            input.unsqueeze(2).expand(-1, -1, self.out_features) 
            * self.weights.T.unsqueeze(0).expand(b, -1, -1)
        )  # [batch,8,10]
        
        # 语义增强（关键修正）
        if self.use_attention:
            attn_adjusted = self.semantic_attention(input)  # [batch,8,10]
            abs_weight = abs_weight * attn_adjusted  # 直接相乘
            
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        sum_abs_w = torch.sum(abs_weight, dim=1)
        
        # 后续计算保持不变
        if self.layer_type == SemiSymbolicLayerType.CONJUNCTION:
            bias = max_abs_w - sum_abs_w
        else:
            bias = sum_abs_w - max_abs_w
            
        out = input @ self.weights.T
        out_bias = self.delta * bias
        return out + out_bias
"""
A generic implementation of constraint layer that can mimic any sort of
constraint.
This is not required for the neural DNF-EO model, since the neural DNF-EO
model's constraint can be initialised easily as a full -6 matrix except 0 on the
diagonal.
class ConstraintLayer(SemiSymbolic):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        delta: float,
        ordered_constraint_list: List[List[int]],
        enable_training: bool = False,
    ):
        super(ConstraintLayer, self).__init__(
            in_features, out_features, SemiSymbolicLayerType.CONJUNCTION, delta
        )
        self.weights.data.fill_(0)
        for class_idx, cl in enumerate(ordered_constraint_list):
            if len(cl) == 0:
                self.weights.data[class_idx, class_idx] = 6
            else:
                for i in cl:
                    self.weights.data[class_idx, i] = -6
            if not enable_training:
                self.requires_grad_(False)
"""


class DNF(nn.Module):
    conjunctions: SemiSymbolic
    disjunctions: SemiSymbolic

    def __init__(
        self,
        num_preds: int,
        num_conjuncts: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal", binary_flag=False
    ) -> None:
        super(DNF, self).__init__()

        self.binary_flag = binary_flag
        self.conjunctions = SemiSymbolic(
            in_features=num_preds,  # P
            out_features=num_conjuncts,  # Q
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight: Q x P

        self.disjunctions = SemiSymbolic(
            in_features=num_conjuncts,  # Q
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
        )  # weight R x Q
        self.con_unary = nn.BatchNorm1d(num_conjuncts)
    def forward(self, input: Tensor, return_feat=False) -> tuple[Any, Any]:
        # Input: N x P
        conj_ = self.conjunctions(input)
        # conj_ = self.con_unary(conj_)
        # conj: N x Q
        # conj =  SignActivation.apply(conj_)
        conj = F.tanh(conj_)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R

        if return_feat and self.binary_flag:
            return disj, conj
        elif return_feat:
            return disj, conj_
        else:
            return disj

    def set_delta_val(self, new_delta_val):
        self.conjunctions.delta = new_delta_val
        self.disjunctions.delta = new_delta_val
        # 补充代码 ‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’
    def get_rules(self, threshold=0.5):
        """提取可读规则（修正版）"""
        # 获取权重矩阵
        conj_weights = self.conjunctions.weights.data.cpu().numpy()
        disj_weights = self.disjunctions.weights.data.cpu().numpy()

        # 生成修正后的合取式
        conjuncts = []
        for i, conj in enumerate(conj_weights):
            terms = []
            for j, w in enumerate(conj):
                if abs(w) > threshold:
                    # 修正点1：符号应为特征修饰符而非连接符
                    operator = "¬" if w < 0 else ""
                    terms.append(f"{operator}P{j+1}")
            
            # 修正点2：使用逻辑与符号连接特征项
            if not terms:
                conjuncts.append(f"conj{i} = ∅")
            else:
                conjuncts.append(f"conj{i} = " + " ∧ ".join(terms))

        # 生成析取式（保持原逻辑）
        class_rules = {}
        for class_idx in range(disj_weights.shape[0]):
            terms = []
            for j, w in enumerate(disj_weights[class_idx]):
                if abs(w) > threshold:
                    operator = "¬" if w < 0 else ""
                    terms.append(f"{operator}conj{j}")
            class_rules[class_idx] = " ∨ ".join(terms) if terms else "∅"

        return {"conjuncts": conjuncts, "disjuncts": class_rules}

class SemanticDNF(DNF):
    def __init__(self,
                 num_preds: int,
                 num_conjuncts: int,
                 n_out: int,
                 delta: float,
                 predicate_embeddings: torch.Tensor,
                 **kwargs):
        
        # 初始化语义嵌入
        super().__init__(num_preds, num_conjuncts, n_out, delta, **kwargs)
        
        # 替换为语义增强层
        self.conjunctions = SemanticSemiSymbolic(
            num_preds, num_conjuncts,
            SemiSymbolicLayerType.CONJUNCTION,
            delta, predicate_embeddings
        )
        
        # 添加语义解释层
        self.semantic_proj = nn.Linear(num_conjuncts + self.conjunctions.embedding_dim, n_out)
    
    def interpretable_forward(self, input: Tensor) -> tuple:
        """增强解释性的前向传播"""
        conj = torch.tanh(self.conjunctions(input))
        
        # 获取语义解释特征
        semantic_feat = self.conjunctions.predicate_emb.mean(dim=0).expand(input.size(0), -1)
        combined = torch.cat([conj, semantic_feat], dim=1)
        
        return self.semantic_proj(combined), conj

# 修改后的 OptimizedSemanticDNF
class OptimizedSemanticDNF(SemanticDNF):
    def __init__(self, num_preds, num_conjuncts, n_out, predicate_embeddings, delta=0.01):
        super().__init__(
            num_preds=num_preds,
            num_conjuncts=num_conjuncts,
            n_out=n_out,
            delta=delta,
            predicate_embeddings=predicate_embeddings
        )
        
        # 注意力层输出与合取项数量对齐
        #
        self.embedding_proj = nn.Linear(768, 64)
        self.proj_norm = nn.LayerNorm(64)
        self.attention = nn.Linear(64, num_conjuncts)  # 关键修正

    def semantic_attention(self, inputs):
        projected = F.gelu(self.embedding_proj(self.predicate_embeddings))
        projected = self.proj_norm(projected)
        print(projected.size())
        
        batch_size = inputs.size(0)
        emb = projected.unsqueeze(0).expand(batch_size, -1, -1)  # [batch,8,64]
        
        # 生成分组合取注意力 [batch,8,10]
        attn_scores = self.attention(emb)  # [batch,8,10]
        return torch.sigmoid(attn_scores) * inputs.unsqueeze(-1)  # [batch,8,10]

class DeltaDelayedExponentialDecayScheduler:
    initial_delta: float
    delta_decay_delay: int
    delta_decay_steps: int
    delta_decay_rate: float

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
    ):
        # initial_delta=0.01 for complicated learning
        self.initial_delta = initial_delta
        self.delta_decay_delay = delta_decay_delay
        self.delta_decay_steps = delta_decay_steps
        self.delta_decay_rate = delta_decay_rate

    def step(self, dnf, step: int) -> float:
        if step < self.delta_decay_delay:
            new_delta_val = self.initial_delta
        else:
            delta_step = step - self.delta_decay_delay
            new_delta_val = self.initial_delta * (
                self.delta_decay_rate ** (delta_step // self.delta_decay_steps)
            )
            # new_delta_val = self.initial_delta * (
            #    delta_step
            # )
        new_delta_val = 1 if new_delta_val > 1 else new_delta_val
        dnf.set_delta_val(new_delta_val)
        return new_delta_val



