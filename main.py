# main.py
"""
训练与评估主程序（兼容你当前 NeuroSymbolicCD 实现）
假设：
- data_utils.create_dataloaders(...) 返回字典（与你之前代码一致）
- models.NeuroSymbolicCD 已按之前给出的签名实现
"""
import os
import time
from typing import Dict
from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# user modules (确保这些模块在 Python path 中)
from data_utils import create_dataloaders
from models import NeuroSymbolicCD

# ----------------------------
# 辅助 / 超参
# ----------------------------
SEED = 42
torch.manual_seed(SEED)

# 训练参数 
BATCH_SIZE = 32
EPOCHS = 10 #训练轮数
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
PATIENCE = 3  # 早停忍耐 epoch 数
LAMBDA_LOGIC = 0.05   # 逻辑规则 loss 权重

#PyTorch 学习率调度器 StepLR 的“旋钮”，用来每过 STEP_SIZE 个 epoch 就把学习率乘以 GAMMA，让模型后期走得更稳
STEP_SIZE = 5
GAMMA = 0.8

# Checkpoint names
BEST_CKPT = "result/best_model.pth"
FULL_SAVE = "result/full_model_and_results.pth"


# 损失函数：
# total_loss = BCE(response) + λ_logic · logic_loss
# - BCE：模型预测 prob 与真实作答 response
# - logic_loss：symbolic 规则对 latent mastery 的结构化正则
def compute_losses(output, target, lambda_logic=LAMBDA_LOGIC):
    """
    total_loss = BCE(response) + lambda_logic * logic_loss
    """
    pred = output['prob']
    bce = F.binary_cross_entropy(pred, target)

    logic_loss = output.get(
        'logic_loss',
        torch.tensor(0.0, device=pred.device)
    )

    total = bce + lambda_logic * logic_loss

    breakdown = {
        'bce': bce.item(),
        'logic_total': logic_loss.item()
    }

    # 仅用于日志分析（不参与反向传播）
    if 'logic_losses' in output:
        for k, v in output['logic_losses'].items():
            breakdown[f'logic_{k}'] = v.item()

    return total, breakdown






# ----------------------------
# 训练 epoch
# ----------------------------
def train_epoch(model: NeuroSymbolicCD,
                dataloader: DataLoader,
                optimizer,
                device,
                lambda_logic: float):

    model.train()
    total_loss = 0.0
    stats = {}
    n_samples = 0

    for batch in dataloader:
        stu = batch['student_id'].to(device)
        exer = batch['exer_id'].to(device)
        resp = batch['resp'].to(device).float()

        out = model(stu, exer)
        loss, breakdown = compute_losses(out, resp, lambda_logic)


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bsz = stu.size(0)
        total_loss += loss.item() * bsz
        n_samples += bsz
        for k, v in breakdown.items():
            if k not in stats:
                stats[k] = 0.0
            stats[k] += v * bsz

    avg_loss = total_loss / n_samples
    avg_stats = {k: stats[k] / n_samples for k in stats}
    return avg_loss, avg_stats



# ----------------------------
# evaluate（兼容按用户分组的 val/test dataset）
# dataset 元素示例：
# {
#   "user_id": 123,
#   "log_num": 4,
#   "logs": [ { "exer_id": 7, "score": 1, "knowledge_code": [...] }, ... ]
# }
# ----------------------------
def evaluate(model: NeuroSymbolicCD, dataset, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for user in dataset:

            # -------------------------------------------------------
            # 格式 1：{ "user_id": xxx, "logs": [...] }
            # -------------------------------------------------------
            if isinstance(user, dict):
                user_id = user.get("user_id", None)

                # 若 user_id 不存在，则从 logs 中读取 student_id
                logs = user.get("logs") or user.get("records") or user.get("log") or None
                if logs is None:
                    raise ValueError("验证集字典格式中缺少 logs / records 字段")

                for log in logs:
                    stu_id = user_id if user_id is not None else log.get("student_id", None)
                    if stu_id is None:
                        raise ValueError("log 中未找到 student_id")

                    stu = torch.tensor([int(stu_id)], dtype=torch.long).to(device)
                    exer = torch.tensor([int(log["exer_id"])], dtype=torch.long).to(device)

                    out = model(stu, exer)
                    pred = float(out["prob"].item())
                    label = float(log.get("score") or log.get("resp") or 0)

                    all_preds.append(pred)
                    all_labels.append(label)

            # -------------------------------------------------------
            # 格式 2：list of dict （每个 user 是记录列表）
            # e.g.  user = [ {student_id, exer_id, score}, ... ]
            # -------------------------------------------------------
            elif isinstance(user, list):
                # 获取 user_id（取 list 中第一条记录的 student_id）
                if len(user) == 0:
                    continue
                stu_id = user[0].get("student_id")
                if stu_id is None:
                    raise ValueError("验证集 list-of-dicts 格式中未找到 student_id")

                for log in user:
                    stu = torch.tensor([int(log["student_id"])], dtype=torch.long).to(device)
                    exer = torch.tensor([int(log["exer_id"])], dtype=torch.long).to(device)

                    out = model(stu, exer)
                    pred = float(out["prob"].item())
                    label = float(log.get("score") or log.get("resp") or 0)

                    all_preds.append(pred)
                    all_labels.append(label)

            else:
                raise ValueError("验证集格式不支持。")

    # ------------------
    # 统计
    # ------------------
    if len(all_labels) == 0:
        return 0.0, 0.0, [], []

    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, preds_binary)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    # ★ 新增 RMSE
    mse = sum((p - y) ** 2 for p, y in zip(all_preds, all_labels)) / len(all_labels)
    rmse = mse ** 0.5

    return accuracy, auc, rmse, all_preds, all_labels



# ----------------------------
# 主函数
# ----------------------------
def main():
    # --- config (可替换) ---
    data_dir = 'data/Assist'   # 修改为你的数据目录
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    lr = LR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data & dataloaders
    print("加载数据与构建 dataloaders ...")
    data_info = create_dataloaders(data_dir, batch_size)

    train_loader = data_info['train_loader']
    val_dataset = data_info['val_dataset']
    test_dataset = data_info['test_dataset']
    Q = data_info['Q']
    student_n = data_info['student_n']
    exercise_n = data_info['exercise_n']
    knowledge_n = data_info['knowledge_n']
    prereq_rules = data_info.get('prereq_rules', [])
    sim_pairs = data_info.get('sim_pairs', [])
    compositional_rules = data_info.get('compositional_rules', [])


    print(f"数据集: students={student_n}, exercises={exercise_n}, knowledge={knowledge_n}")
    print(f"训练样本数: {len(train_loader.dataset)}, 验证用户数: {len(val_dataset)}, 测试用户数: {len(test_dataset)}")
    print(
    f"rules: prereq={len(prereq_rules)}, "
    f"sim_pairs={len(sim_pairs)}, "
    f"comp={len(compositional_rules)}"
    )

    # --- model init ---
    print("初始化模型 ...")
    model = NeuroSymbolicCD(
    n_students=student_n,
    n_exer=exercise_n,
    knowledge_n=knowledge_n,
    student_dim=64,
    item_dim=64,
    disc_dim=16,
    q_proj_dim=32,
    prereq_rules=prereq_rules,
    sim_pairs=sim_pairs,
    compositional_rules=compositional_rules
    ).to(device)


    # load Q into model buffer (非常关键)
    if isinstance(Q, torch.Tensor):
        model.Q.copy_(Q.float())
    else:
        model.Q.copy_(torch.tensor(Q, dtype=torch.float32))

    # print params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embed_params = model.count_non_embedding_params() if hasattr(model, 'count_non_embedding_params') else None

    print(f"模型总参数: {total_params:,}, 可训练: {trainable_params:,}")
    if non_embed_params is not None:
        print(f"非 embedding 参数（估计）: {non_embed_params:,}")

    # optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # training loop with early stopping
    best_val_auc = 0.0
    best_epoch = 0
    val_auc_window = []
    history = {'train_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if epoch <= 3:
            lambda_logic = 0.0
        else:
            lambda_logic = LAMBDA_LOGIC

        train_loss, train_stats = train_epoch(
            model, train_loader, optimizer, device, lambda_logic
        )
        scheduler.step()

        val_acc, val_auc, val_rmse, _, _ = evaluate(model, val_dataset, device)
        val_auc_window.append(val_auc)
        if len(val_auc_window) > 3:
            val_auc_window.pop(0)

        avg_val_auc = sum(val_auc_window) / len(val_auc_window)

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} - time: {elapsed:.1f}s "
            f"- TrainLoss: {train_loss:.4f} "
            f"(BCE={train_stats['bce']:.4f}, logic={train_stats['logic_total']:.4f}) "
            f"ValAcc={val_acc:.4f} ValAUC={val_auc:.4f} ValRMSE={val_rmse:.4f}"
        )
        RULE_ORDER = [
            "prereq", "sim", "smooth", "mono",
            "comp", "stu_diff"
        ]


        rule_items = []
        for r in RULE_ORDER:
            key = f"logic_{r}"
            if key in train_stats:
                rule_items.append((r, train_stats[key]))

        if len(rule_items) > 0:
            msg = "    RuleLoss | "
            msg += "  ".join([f"{k}={v:.4f}" for k, v in rule_items])
            print(msg)

        # checkpoint
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"  Saved best model (avg AUC={avg_val_auc:.4f})")

        # early stopping
        if epoch - best_epoch >= PATIENCE:
            print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
            break

    # load best model and test
    if os.path.exists(BEST_CKPT):
        model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
        print(f"Loaded best model from epoch {best_epoch} with val AUC={best_val_auc:.4f}")

    print("Evaluating on test set ...")
    test_acc, test_auc, test_rmse, test_preds, test_labels = evaluate(model, test_dataset, device)
    print(f"Test Accuracy: {test_acc:.4f}, "f"Test AUC: {test_auc:.4f}, "f"Test RMSE: {test_rmse:.4f}")

    # save full artifact
    print("Saving full model and results ...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_students': student_n,
            'n_exer': exercise_n,
            'knowledge_n': knowledge_n,
            'prereq_rules': prereq_rules,
            'sim_pairs': sim_pairs
        },
        'history': history,
        'test_results': {
            'accuracy': test_acc,
            'auc': test_auc,
            'rmse': test_rmse,
            'predictions': test_preds,
            'labels': test_labels
        }
    }, FULL_SAVE)

    print(f"Saved: {FULL_SAVE}")
    print(f"Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print("Done.")


if __name__ == "__main__":
    main()
