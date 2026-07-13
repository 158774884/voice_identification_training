"""
声纹说话人推理模块

支持:
- 声纹嵌入提取
- 1:1 声纹比对 (verification)
- 1:N 声纹识别 (identification)
- 多注册样本融合
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class SpeakerInference:
    """
    声纹嵌入提取器

    Args:
        model: MultiTaskVoiceModel 或仅 Speaker 分支
        device: 运行设备
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract_embedding(self, audio: torch.Tensor,
                          audio_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取声纹嵌入向量

        Args:
            audio: [B, 1, T] 音频
            audio_lengths: [B]

        Returns:
            embeddings: [B, embed_dim] L2 归一化嵌入
        """
        audio = audio.to(self.device)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)

        self.model.eval()
        outputs = self.model(audio, audio_lengths,
                             task_mask={'asr': False, 'dialect': False, 'speaker': True})

        return outputs['speaker_embedding']  # 已经是 L2 归一化的

    def extract_single(self, audio: torch.Tensor) -> np.ndarray:
        """
        提取单条音频的声纹嵌入

        Args:
            audio: [T] 或 [1, T] 音频

        Returns:
            embedding: [embed_dim] numpy array
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        emb = self.extract_embedding(audio)
        return emb[0].cpu().numpy()

    def extract_from_chunks(self, audio_chunks: List[torch.Tensor],
                            aggregation: str = 'mean') -> np.ndarray:
        """
        从多个 chunk 提取融合声纹

        Args:
            audio_chunks: 音频片段列表
            aggregation: 'mean' | 'max_sim' | 'attention'

        Returns:
            fused_embedding: [embed_dim]
        """
        embeddings = []

        for chunk in audio_chunks:
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0).unsqueeze(0)
            emb = self.extract_embedding(chunk.to(self.device))
            embeddings.append(emb.cpu())

        # 聚合
        if aggregation == 'mean':
            fused = torch.stack(embeddings).mean(dim=0)
        elif aggregation == 'max_sim':
            # 取与均值最相似的作为代表
            mean_emb = torch.stack(embeddings).mean(dim=0, keepdim=True)
            similarities = F.cosine_similarity(
                torch.stack(embeddings).squeeze(1),
                mean_emb.squeeze(0),
                dim=-1
            )
            best_idx = similarities.argmax().item()
            fused = embeddings[best_idx]
        else:
            fused = torch.stack(embeddings).mean(dim=0)

        # L2 归一化
        fused = F.normalize(fused.squeeze(0), p=2, dim=0)

        return fused.numpy()


class SpeakerVerification:
    """
    声纹比对系统

    支持:
    - 1:1 验证 (是否同一人)
    - 1:N 识别 (在一组注册者中查找)
    - 阈值自适应 (EER 阈值)
    """

    def __init__(self, speaker_inference: SpeakerInference,
                 threshold: float = 0.65,
                 enroll_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            speaker_inference: SpeakerInference 实例
            threshold: 验证阈值 (需在验证集上调优)
            enroll_embeddings: 预注册的声纹库 {name: embedding}
        """
        self.extractor = speaker_inference
        self.threshold = threshold
        self.enroll_db = enroll_embeddings or {}

    def enroll(self, name: str, audio: torch.Tensor,
               num_enroll_utterances: int = 1) -> np.ndarray:
        """
        注册说话人

        Args:
            name: 说话人名称/ID
            audio: [num_utts, T] 或 [T] 注册音频
            num_enroll_utterances: 多句注册时取平均

        Returns:
            enrollment_embedding: [embed_dim]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, T]

        # 多句注册: 提取每条后取平均
        embeddings = []
        for i in range(min(audio.size(0), num_enroll_utterances)):
            emb = self.extractor.extract_single(audio[i])
            embeddings.append(emb)

        enrolled = np.mean(embeddings, axis=0)
        enrolled = enrolled / (np.linalg.norm(enrolled) + 1e-8)  # L2 normalize

        # 存入数据库
        self.enroll_db[name] = enrolled

        print(f"[Enroll] Speaker '{name}' enrolled successfully")

        return enrolled

    def verify(self, name: str, test_audio: torch.Tensor) -> Tuple[bool, float]:
        """
        1:1 验证: 测试音频是否属于声明的说话人

        Args:
            name: 声明的说话人名称
            test_audio: 测试音频 [T]

        Returns:
            is_match: bool
            similarity: float
        """
        if name not in self.enroll_db:
            raise ValueError(f"Speaker '{name}' not enrolled. Enroll first.")

        enroll_emb = self.enroll_db[name]
        test_emb = self.extractor.extract_single(test_audio)

        similarity = float(np.dot(enroll_emb, test_emb))

        is_match = similarity >= self.threshold

        return is_match, similarity

    def identify(self, test_audio: torch.Tensor,
                 top_k: int = 3) -> List[Tuple[str, float]]:
        """
        1:N 识别: 在注册库中查找测试音频的说话人

        Args:
            test_audio: 测试音频 [T]
            top_k: 返回 top-K 匹配

        Returns:
            results: [(name, similarity), ...] 降序排列
        """
        if not self.enroll_db:
            raise ValueError("Enrollment database is empty.")

        test_emb = self.extractor.extract_single(test_audio)

        similarities = []
        for name, enroll_emb in self.enroll_db.items():
            sim = float(np.dot(enroll_emb, test_emb))
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def remove(self, name: str):
        """从注册库中删除说话人"""
        if name in self.enroll_db:
            del self.enroll_db[name]
            print(f"[Enroll] Speaker '{name}' removed")

    def list_speakers(self) -> List[str]:
        """列出所有注册说话人"""
        return list(self.enroll_db.keys())

    def calibrate_threshold(self, genuine_scores: List[float],
                            impostor_scores: List[float]) -> float:
        """
        校准验证阈值 (基于 EER)

        Args:
            genuine_scores: 正例比对分数列表
            impostor_scores: 负例比对分数列表

        Returns:
            eer_threshold: 等错误率阈值
        """
        # 可能的阈值范围
        all_scores = sorted(set(genuine_scores + impostor_scores))

        best_threshold = 0.0
        best_eer_diff = float('inf')

        for threshold in all_scores:
            # False Accept Rate (FAR)
            far = sum(1 for s in impostor_scores if s >= threshold) / len(impostor_scores)
            # False Reject Rate (FRR)
            frr = sum(1 for s in genuine_scores if s < threshold) / len(genuine_scores)

            diff = abs(far - frr)
            if diff < best_eer_diff:
                best_eer_diff = diff
                best_threshold = threshold

        self.threshold = best_threshold
        print(f"[Calibrate] EER threshold: {best_threshold:.4f}")

        return best_threshold
