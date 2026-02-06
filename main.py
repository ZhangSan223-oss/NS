import os
import time
from typing import Dict
from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from data_utils import create_dataloaders
from models import NeuroSymbolicCD
SEED = 42
torch.manual_seed(SEED)


BATCH_SIZE = 32
EPOCHS = 10 
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
PATIENCE = 3 
LAMBDA_LOGIC = 0.05 

STEP_SIZE = 5
GAMMA = 0.8


BEST_CKPT = "result/best_model.pth"
FULL_SAVE = "result/full_model_and_results.pth"



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

    
    if 'logic_losses' in output:
        for k, v in output['logic_losses'].items():
            breakdown[f'logic_{k}'] = v.item()

    return total, breakdown







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

            
            elif isinstance(user, list):
                
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

    
    if len(all_labels) == 0:
        return 0.0, 0.0, [], []

    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, preds_binary)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    
    mse = sum((p - y) ** 2 for p, y in zip(all_preds, all_labels)) / len(all_labels)
    rmse = mse ** 0.5

    return accuracy, auc, rmse, all_preds, all_labels




def main():
    
    data_dir = 'data/Assist'  
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    lr = LR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    
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


    
    if isinstance(Q, torch.Tensor):
        model.Q.copy_(Q.float())
    else:
        model.Q.copy_(torch.tensor(Q, dtype=torch.float32))

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embed_params = model.count_non_embedding_params() if hasattr(model, 'count_non_embedding_params') else None

    print(f"模型总参数: {total_params:,}, 可训练: {trainable_params:,}")
    if non_embed_params is not None:
        print(f"非 embedding 参数（估计）: {non_embed_params:,}")

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    
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

        
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"  Saved best model (avg AUC={avg_val_auc:.4f})")

        
        if epoch - best_epoch >= PATIENCE:
            print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
            break

    
    if os.path.exists(BEST_CKPT):
        model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
        print(f"Loaded best model from epoch {best_epoch} with val AUC={best_val_auc:.4f}")

    print("Evaluating on test set ...")
    test_acc, test_auc, test_rmse, test_preds, test_labels = evaluate(model, test_dataset, device)
    print(f"Test Accuracy: {test_acc:.4f}, "f"Test AUC: {test_auc:.4f}, "f"Test RMSE: {test_rmse:.4f}")

    
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
