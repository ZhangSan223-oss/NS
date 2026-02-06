import numpy as np
from sklearn.preprocessing import normalize

def load_q_matrix(path="q.txt"):
    Q = np.loadtxt(path, dtype=np.float32)
    return Q   # shape: (n_items, n_knowledge)

def svd_denoise_q(Q, rank=10, threshold=0.3):
    """
    Q: 原始 Q 矩阵
    rank: SVD 保留的有效特征维度
    threshold: 超过阈值则置 1，否则置 0
    """
    # SVD 分解
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)

    # 保留前 rank 个奇异值
    S_reduced = np.zeros_like(S)
    S_reduced[:rank] = S[:rank]

    # 低秩重构
    Q_reconstructed = (U * S_reduced) @ Vt

    # 归一化到 [0,1]
    Q_norm = (Q_reconstructed - Q_reconstructed.min()) / (Q_reconstructed.max() - Q_reconstructed.min())

    # 二值化
    Q_binary = (Q_norm > threshold).astype(np.float32)

    return Q_binary, Q_norm

# ======== 主入口 ========
if __name__ == "__main__":
    Q = load_q_matrix("q.txt")
    print("原始 Q 形状:", Q.shape)

    Q_new, Q_soft = svd_denoise_q(Q, rank=8, threshold=0.25)
    print("降噪后 Q 形状:", Q_new.shape)

    np.savetxt("q_denoised.txt", Q_new, fmt="%d")
    np.savetxt("q_soft.txt", Q_soft, fmt="%.5f")

    print("已保存: q_denoised.txt 和 q_soft.txt")
