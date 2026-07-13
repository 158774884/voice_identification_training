"""
评估指标

ASR: WER (词错误率), CER (字符错误率)
Dialect: Accuracy, Confusion Matrix
Speaker: EER, minDCF
"""

import numpy as np
from typing import List, Tuple


def levenshtein_distance(ref: List, hyp: List) -> int:
    """Levenshtein 编辑距离"""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """
    Character Error Rate (字符错误率)

    CER = (S + D + I) / N * 100%

    Args:
        references: 参考文本列表
        hypotheses: 识别文本列表

    Returns:
        cer: 字符错误率 (%)
    """
    total_edits = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.replace(' ', ''))
        hyp_chars = list(hyp.replace(' ', ''))
        total_edits += levenshtein_distance(ref_chars, hyp_chars)
        total_chars += max(len(ref_chars), 1)

    return (total_edits / total_chars) * 100.0


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Word Error Rate (词错误率)

    WER = (S + D + I) / N * 100%

    Args:
        references: 参考文本列表
        hypotheses: 识别文本列表

    Returns:
        wer: 词错误率 (%)
    """
    total_edits = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_edits += levenshtein_distance(ref_words, hyp_words)
        total_words += max(len(ref_words), 1)

    return (total_edits / total_words) * 100.0


def compute_eer(genuine_scores: List[float],
                impostor_scores: List[float]) -> Tuple[float, float]:
    """
    Equal Error Rate (等错误率)

    EER = 当 FAR == FRR 时的错误率

    Args:
        genuine_scores: 正例比对分数 (同一人)
        impostor_scores: 负例比对分数 (不同人)

    Returns:
        eer: 等错误率
        threshold: 对应阈值
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # 搜索所有可能的阈值
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))

    best_eer = 1.0
    best_threshold = 0.0

    for threshold in thresholds:
        # FAR: False Accept Rate
        far = np.mean(impostor_scores >= threshold)
        # FRR: False Reject Rate
        frr = np.mean(genuine_scores < threshold)

        eer_candidate = (far + frr) / 2.0
        if abs(far - frr) < abs(best_eer * 2 - 1):
            best_eer = eer_candidate
            best_threshold = threshold

    return best_eer, best_threshold


def compute_min_dcf(genuine_scores: List[float],
                    impostor_scores: List[float],
                    p_target: float = 0.01,
                    c_miss: float = 1.0,
                    c_fa: float = 1.0) -> float:
    """
    Minimum Detection Cost Function

    minDCF 是声纹验证的标准评估指标

    DCF = C_miss * P_miss * P_target + C_fa * P_fa * (1 - P_target)

    Args:
        genuine_scores: 正例分数
        impostor_scores: 负例分数
        p_target: 先验目标概率 (默认 0.01)
        c_miss: 漏检代价
        c_fa: 误报代价

    Returns:
        min_dcf: 最小检测代价
    """
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    thresholds = np.sort(np.concatenate([genuine, impostor]))

    min_dcf = float('inf')
    best_threshold = 0.0

    for t in thresholds:
        p_miss = np.mean(genuine < t)  # False Reject
        p_fa = np.mean(impostor >= t)  # False Accept

        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)

        if dcf < min_dcf:
            min_dcf = dcf
            best_threshold = t

    return min_dcf
