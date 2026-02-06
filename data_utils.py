

import json
import torch
from torch.utils.data import Dataset, DataLoader

class FrcSubDataset(Dataset):
    def __init__(self, records, knowledge_dim):
        self.records = records
        self.knowledge_dim = knowledge_dim
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        log = self.records[idx]
        knowledge_emb = [0.] * self.knowledge_dim
        for knowledge_code in log['knowledge_code']:
            knowledge_emb[knowledge_code - 1] = 1.0
        
        return {
            "student_id": torch.tensor(log['user_id'] - 1, dtype=torch.long),
            "exer_id": torch.tensor(log['exer_id'] - 1, dtype=torch.long),
            "resp": torch.tensor(log['score'], dtype=torch.float32),
            "knowledge_emb": torch.tensor(knowledge_emb, dtype=torch.float32)
        }


class ValTestDataset(Dataset):
    def __init__(self, data, knowledge_dim):
        self.data = data
        self.knowledge_dim = knowledge_dim
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_data = self.data[idx]
        user_id = user_data['user_id']
        records = []
        for log in user_data['logs']:
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            
            records.append({
                "student_id": user_id - 1,
                "exer_id": log['exer_id'] - 1,
                "resp": log['score'],
                "knowledge_emb": knowledge_emb
            })
        
        return records


def load_config(config_path):
    with open(config_path, 'r') as f:
        f.readline()  
        line = f.readline().strip()
        student_n, exercise_n, knowledge_n = map(int, line.split(','))
    return student_n, exercise_n, knowledge_n


def load_q_matrix(q_path, exercise_n, knowledge_n):
    Q = torch.zeros((exercise_n, knowledge_n), dtype=torch.float32)
    with open(q_path, 'r') as f:
        for i, line in enumerate(f):
            values = list(map(float, line.strip().split()))
            if len(values) != knowledge_n:
                raise ValueError(f"Q矩阵第{i}行维度不匹配: 期望{knowledge_n}, 实际{len(values)}")
            Q[i] = torch.tensor(values)
    return Q


def load_rules(rules_path):
    with open(rules_path, 'r', encoding='utf8') as f:
        rules_data = json.load(f)
    
    prereq_rules = []
    sim_pairs = []
    compositional_rules = []  
    comp_rules_count = 0 
    

    if 'rules' in rules_data:
        
        for rule in rules_data['rules']:
            rule_type = rule.get('type')
            
            if rule_type == 'prerequisite':
                pre = rule.get('prerequisite')
                target = rule.get('target')
                if pre is not None and target is not None:
                    
                    prereq_rules.append((pre, target))
                    
            elif rule_type == 'similar':
                skill_a = rule.get('skill_a')
                skill_b = rule.get('skill_b')
                if skill_a is not None and skill_b is not None:
                    sim_pairs.append((skill_a, skill_b))
                    
            elif rule_type == 'compositional':
                components = rule.get('components')
                target = rule.get('target')
                if components is not None and target is not None:
                    compositional_rules.append((components, target))
                    
    else:
        
        prereq_rules = rules_data.get('prerequisite', [])
        sim_pairs = rules_data.get('similarity', [])
        
        
        prereq_rules = [(a-1, b-1) for a, b in prereq_rules]
        sim_pairs = [(a-1, b-1) for a, b in sim_pairs]
    
    print(
    f"加载规则: {len(prereq_rules)} 条先修规则, "
    f"{len(sim_pairs)} 条相似关系, "
    f"{len(compositional_rules)} 条组合规则"
    )
    if comp_rules_count > 0:
        print(f"注意: 忽略了 {comp_rules_count} 条组合规则")
    
    
    if prereq_rules:
        print("先修规则:")
        for pre, post in prereq_rules:
            print(f"  Skill_{pre} → Skill_{post}")
    
    if sim_pairs:
        print("相似关系:")
        for a, b in sim_pairs:
            print(f"  Skill_{a} ↔ Skill_{b}")
    
    return prereq_rules, sim_pairs, compositional_rules



def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)


def create_dataloaders(data_dir='data/FrcSub', batch_size=32):
    
    config_path = f'{data_dir}/config.txt'
    q_path = f'{data_dir}/q.txt'
    rules_path = f'{data_dir}/rules.json'
    
    print(f"加载配置文件: {config_path}")
    student_n, exercise_n, knowledge_n = load_config(config_path)
    print(f"配置信息: 学生数={student_n}, 题目数={exercise_n}, 知识点数={knowledge_n}")
    
    print(f"加载Q矩阵: {q_path}")
    Q = load_q_matrix(q_path, exercise_n, knowledge_n)
    
    
    print(f"加载规则文件: {rules_path}")
    prereq_rules, sim_pairs, compositional_rules = load_rules(rules_path)

    
    
    print(f"加载训练数据...")
    train_data_path = f'{data_dir}/train_set.json'
    train_data = load_json_data(train_data_path)
    train_dataset = FrcSubDataset(train_data, knowledge_n)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"训练集: {len(train_dataset)} 条记录")
    
    
    print(f"加载验证数据...")
    val_data_path = f'{data_dir}/val_set.json'
    val_data = load_json_data(val_data_path)
    val_dataset = ValTestDataset(val_data, knowledge_n)
    print(f"验证集: {len(val_dataset)} 个用户")
    
    
    print(f"加载测试数据...")
    test_data_path = f'{data_dir}/test_set.json'
    test_data = load_json_data(test_data_path)
    test_dataset = ValTestDataset(test_data, knowledge_n)
    print(f"测试集: {len(test_dataset)} 个用户")
    
    return {
        'train_loader': train_loader,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'Q': Q,
        'student_n': student_n,
        'exercise_n': exercise_n,
        'knowledge_n': knowledge_n,
        'prereq_rules': prereq_rules,
        'sim_pairs': sim_pairs,
        'compositional_rules': compositional_rules
    }
