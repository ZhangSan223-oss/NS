from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEmbedding(nn.Module):
    def __init__(self, n_students, n_exer, student_dim=32, item_dim=32, disc_dim=8):
        super().__init__()
        self.student_emb = nn.Embedding(n_students, student_dim)
        self.item_emb = nn.Embedding(n_exer, item_dim)
        self.item_disc = nn.Embedding(n_exer, disc_dim)
        self.emb_drop = nn.Dropout(0.2)

    def forward(self, stu_idx, exer_idx):
        stu = self.emb_drop(self.student_emb(stu_idx))
        item = self.emb_drop(self.item_emb(exer_idx))
        disc = self.emb_drop(self.item_disc(exer_idx))
        return stu, item, disc



class NeuralPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return torch.sigmoid(self.fc3(x)).squeeze(-1)



class DifferentiableRuleEngine(nn.Module):
    def __init__(self,
                 knowledge_n: int,
                 proj_hidden: Optional[int] = None,
                 prereq_rules: List[Tuple[int, int]] = [],
                 sim_pairs: List[Tuple[int, int]] = [],
                 compositional_rules: List[Tuple[List[int], int]] = [],
                 enable_smooth: bool = True,
                 enable_mono: bool = True):
        super().__init__()

        self.knowledge_n = knowledge_n
        self.prereq_rules = prereq_rules
        self.sim_pairs = sim_pairs
        self.compositional_rules = compositional_rules
        self.enable_smooth = enable_smooth
        self.enable_mono = enable_mono

        hidden = proj_hidden or min(128, knowledge_n)
        self.proj = nn.Linear(knowledge_n, hidden)
        self.out = nn.Linear(hidden, knowledge_n)

    def forward(self,
                mastery: torch.Tensor,
                rule_weights: dict,
                disc_emb: Optional[torch.Tensor] = None):

        x = torch.tanh(self.proj(mastery))
        alpha_sym = torch.sigmoid(self.out(x))

        logic_losses = {}

        
        if self.prereq_rules:
            loss = []
            for a, b in self.prereq_rules:
                loss.append(F.relu(alpha_sym[:, a] - alpha_sym[:, b]).mean())
            logic_losses['prereq'] = torch.stack(loss).mean()

        
        if self.sim_pairs:
            loss = []
            for i, j in self.sim_pairs:
                loss.append(F.mse_loss(alpha_sym[:, i], alpha_sym[:, j]))
            logic_losses['sim'] = torch.stack(loss).mean()

        
        if self.enable_smooth and self.prereq_rules:
            loss = []
            for a, b in self.prereq_rules:
                loss.append((alpha_sym[:, a] - alpha_sym[:, b]).pow(2).mean())
            logic_losses['smooth'] = torch.stack(loss).mean()

        
        if self.enable_mono:
            mean_alpha = alpha_sym.mean(dim=1, keepdim=True)
            logic_losses['mono'] = F.relu(mean_alpha - alpha_sym).mean()
        if self.compositional_rules:
            loss = []
            for comps, target in self.compositional_rules:
                comp_alpha = alpha_sym[:, comps]           # (B, k)
                min_comp = comp_alpha.min(dim=1).values    # (B,)
                loss.append(F.relu(min_comp - alpha_sym[:, target]).mean())
            logic_losses['comp'] = torch.stack(loss).mean()
        
        if disc_emb is not None:
            diff = torch.sigmoid(disc_emb.mean(dim=1))
            ability = alpha_sym.mean(dim=1)
            logic_losses['stu_diff'] = F.relu(diff - ability).mean()

        total_loss = torch.zeros(1, device=alpha_sym.device)
        for k, v in logic_losses.items():
            total_loss += rule_weights.get(k, 1.0) * v

        return alpha_sym, total_loss, logic_losses



class NeuroSymbolicCD(nn.Module):
    def __init__(self,
                 n_students: int,
                 n_exer: int,
                 knowledge_n: int,
                 student_dim: int = 64,
                 item_dim: int = 64,
                 disc_dim: int = 16,
                 q_proj_dim: int = 32,
                 prereq_rules: List[Tuple[int, int]] = [],
                 sim_pairs: List[Tuple[int, int]] = [],
                 compositional_rules: List[Tuple[List[int], int]] = []):
        super().__init__()

        self.embed = SharedEmbedding(n_students, n_exer,
                                     student_dim, item_dim, disc_dim)

        
        self.mastery_inner = nn.Embedding(n_students, 32)
        self.mastery_out = nn.Linear(32, knowledge_n)

        self.register_buffer('Q', torch.zeros(n_exer, knowledge_n))
        self.q_proj = nn.Linear(knowledge_n, q_proj_dim)

        concat_dim = student_dim + item_dim + disc_dim + q_proj_dim
        self.neural = NeuralPredictor(concat_dim)

        self.symbolic = DifferentiableRuleEngine(
            knowledge_n,
            prereq_rules=prereq_rules,
            sim_pairs=sim_pairs,
            compositional_rules=compositional_rules
        )
        
        self.rule_weights = {
            'prereq': 1.0,
            'sim': 0.5,
            'smooth': 0.2,
            'mono': 0.05,
            'comp': 0.3,      
            'stu_diff': 0.1
        }


    def load_Q(self, Q_tensor: torch.Tensor):
        self.Q.copy_(Q_tensor.float())

    def forward(self, stu_idx, exer_idx):
        stu_emb, item_emb, disc_emb = self.embed(stu_idx, exer_idx)
        q_vec = torch.relu(self.q_proj(self.Q[exer_idx]))

        
        x = torch.cat([stu_emb, item_emb, disc_emb, q_vec], dim=-1)
        prob_neural = self.neural(x)

        
        mastery = torch.sigmoid(self.mastery_out(self.mastery_inner(stu_idx)))
        alpha_sym, logic_loss, logic_losses = self.symbolic(
            mastery,
            rule_weights=self.rule_weights,
            disc_emb=disc_emb
        )

        
        masked = alpha_sym * self.Q[exer_idx]
        prob_sym = masked.sum(dim=1) / (self.Q[exer_idx].sum(dim=1) + 1e-6)

        
        prob = 0.7 * prob_neural + 0.3 * prob_sym

        return {
            'prob': prob,
            'prob_neural': prob_neural,
            'prob_sym': prob_sym,
            'alpha_sym': alpha_sym,
            'logic_loss': logic_loss,
            'logic_losses': logic_losses
        }
