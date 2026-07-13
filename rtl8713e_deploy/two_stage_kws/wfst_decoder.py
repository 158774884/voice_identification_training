"""
小型 WFST 语法解码器 — 用于命令词 CTC ASR 解码

原理:
  命令列表 → 构建有限状态语法图 (FSG) → CTC beam search 约束在语法空间内

语法图构建:
  命令: "打开客厅的灯" "打开卧室的灯" "关闭空调"
  自动机:
           打 → 开 → 客/卧 → 厅/室 → 的 → 灯 → [EOS]
                          ↘ 空 → 调 → [EOS]
                    ↗ 关 → 闭

  搜索空间: ~200条命令 → ~500条路径 (vs 全文听写 5000^N 条)
  → 这就是为什么 512KB SRAM 也能跑 "离线ASR"

内存:
  WFST 图: ~150KB (200条命令, 500+子词)
  解码状态: ~20KB (beam=10)

AC7916AB: WFST 解码在 CPU 上跑, DNN 声学模型在 MVA 上跑
"""

from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import json


class WFSTGrammarDecoder:
    """
    小型 WFST 语法解码器 — 限缩在预设命令集的搜索空间

    不是通用 ASR decoder, 而是带语法约束的命令词解码器
    """

    def __init__(self, blank_id=0, beam_width=8, lm_weight=0.1):
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.lm_weight = lm_weight

        # 语法图: token → next_tokens 转移表
        self.grammar: Dict[int, Set[int]] = defaultdict(set)

        # 每个 token 是否为终止符
        self.terminals: Set[int] = set()

        # 语言模型分数 (简单的 unigram)
        self.lm_scores: Dict[Tuple[int, int], float] = {}

    def build_from_commands(self, commands: List[str],
                            tokenizer,  # ChineseVocab or similar
                            ) -> int:
        """
        从命令列表构建语法图

        Args:
            commands: ["打开客厅的灯", "打开卧室的灯", "关闭空调", ...]
            tokenizer: 有 encode() 方法的对象

        Returns:
            num_paths: 语法图中的总路径数
        """
        self.grammar.clear()
        self.terminals.clear()

        # 统计转移频率 (构建 unigram LM)
        bigram_counts = defaultdict(int)

        for cmd in commands:
            tokens = tokenizer.encode(cmd)  # [t1, t2, ..., tn]
            if not tokens:
                continue

            for i, t in enumerate(tokens):
                if i == 0:
                    # 可以从 blank 或 sos 进入
                    self.grammar[self.blank_id].add(t)
                else:
                    self.grammar[tokens[i-1]].add(t)

                # 统计 bigram
                if i > 0:
                    bigram_counts[(tokens[i-1], t)] += 1
                else:
                    bigram_counts[(-1, t)] += 1  # -1 = sentence start

            # 最后一个 token → EOS (终止)
            last = tokens[-1]
            self.terminals.add(last)

        # 归一化 bigram 为概率 → log prob
        unigram_total = sum(bigram_counts.values())
        for (prev, curr), count in bigram_counts.items():
            self.lm_scores[(prev, curr)] = -np.log(count / unigram_total)

        # 计算总路径数 (近似)
        num_paths = sum(1 for _ in self._enumerate_paths(max_depth=10))
        return num_paths

    def decode(self, log_probs: 'np.ndarray',
               top_k: int = 3) -> List[Tuple[List[int], float]]:
        """
        Grammar-constrained CTC beam search

        Args:
            log_probs: [T, V] 对数概率 (CTC输出)
            top_k: 返回 top-k 结果

        Returns:
            [(token_ids, score), ...] 排序好的结果
        """
        import numpy as np
        T, V = log_probs.shape

        # 初始 beam: (token_sequence, log_prob, last_token)
        beams = [([], 0.0, self.blank_id)]

        for t in range(T):
            frame_lp = log_probs[t]  # [V]
            new_beams = []

            for seq, score, last in beams:
                # 对于每个 beam, 考虑:
                # 1. 保持 blank (不变)
                blank_score = score + frame_lp[self.blank_id]
                new_beams.append((seq[:], blank_score, self.blank_id))

                # 2. 语法允许的下一个 token
                allowed = self.grammar.get(last, set())
                # 如果是 blank, 可以从头开始
                if last == self.blank_id:
                    allowed = allowed | self.grammar.get(self.blank_id, set())

                # 保留当前 token (continuation)
                if last != self.blank_id:
                    allowed = allowed | {last}

                for token in allowed:
                    if token >= V:
                        continue

                    new_score = score + frame_lp[token]

                    # LM bonus
                    if last != self.blank_id and token != last:
                        lm_key = (last, token)
                    elif last == self.blank_id:
                        lm_key = (-1, token)
                    else:
                        lm_key = None

                    if lm_key and lm_key in self.lm_scores:
                        new_score += self.lm_weight * self.lm_scores[lm_key]

                    new_seq = seq[:]
                    if token != last and token != self.blank_id:
                        new_seq.append(token)

                    new_beams.append((new_seq, new_score, token))

            # 剪枝: 保留 top beam_width
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_width]

        # 筛选有效结果 (以终止符结尾)
        results = []
        for seq, score, last in beams:
            is_valid = len(seq) > 0 and (last in self.terminals or
                                          seq[-1] in self.terminals)
            # 长度归一化: 用 per-token 平均分, 避免短序列不公平优势
            adjusted_score = score / max(len(seq), 1)
            results.append((seq, adjusted_score, is_valid))

        # 排序: 有效结果优先
        results.sort(key=lambda x: (x[2], x[1]), reverse=True)

        return [(r[0], r[1]) for r in results[:top_k]]

    def _enumerate_paths(self, max_depth=10):
        """枚举所有路径 (用于计数)"""
        visited = set()
        stack = [([], self.blank_id)]

        while stack:
            path, node = stack.pop()
            if len(path) >= max_depth:
                yield path
                continue

            for next_node in self.grammar.get(node, set()):
                if (tuple(path), next_node) in visited:
                    continue
                visited.add((tuple(path), next_node))
                new_path = path + [next_node]
                stack.append((new_path, next_node))
                yield new_path

    def to_dict(self) -> dict:
        """序列化 (用于部署时加载)"""
        return {
            'grammar': {str(k): list(v) for k, v in self.grammar.items()},
            'terminals': list(self.terminals),
            'lm_scores': {f'{k[0]},{k[1]}': v for k, v in self.lm_scores.items()},
            'blank_id': self.blank_id,
            'beam_width': self.beam_width,
            'lm_weight': self.lm_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WFSTGrammarDecoder':
        obj = cls(d['blank_id'], d['beam_width'], d['lm_weight'])
        obj.grammar = {int(k): set(v) for k, v in d['grammar'].items()}
        obj.terminals = set(d['terminals'])
        obj.lm_scores = {}
        for k, v in d['lm_scores'].items():
            parts = k.split(',')
            obj.lm_scores[(int(parts[0]), int(parts[1]))] = v
        return obj

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> 'WFSTGrammarDecoder':
        with open(path) as f:
            return cls.from_dict(json.load(f))


def build_command_grammar(commands: List[str], vocab,
                           output_path: Optional[str] = None) -> WFSTGrammarDecoder:
    """
    便捷函数: 从命令列表构建并保存 WFST 语法

    Args:
        commands: 命令文本列表
        vocab: ChineseVocab 实例
        output_path: 保存路径 (optional)

    Returns:
        decoder: WFSTGrammarDecoder 实例

    Example:
        commands = [
            "打开客厅的灯", "打开卧室的灯", "打开厨房的灯",
            "关闭客厅的灯", "关闭卧室的灯",
            "打开空调", "关闭空调",
            "温度调高", "温度调低",
            "风速加大", "风速减小",
            "制冷模式", "制热模式", "除湿模式", "送风模式",
            "定时一小时", "定时两小时",
        ]
        decoder = build_command_grammar(commands, vocab)
        print(f"Grammar: {len(decoder.grammar)} states, "
              f"{len(decoder.terminals)} terminals")
    """
    decoder = WFSTGrammarDecoder(blank_id=vocab.blank_id)
    num_paths = decoder.build_from_commands(commands, vocab)

    mem_estimate = len(decoder.grammar) * 32 + len(decoder.lm_scores) * 16
    print(f"[WFST] Built grammar: {len(commands)} commands -> "
          f"{len(decoder.grammar)} states, ~{mem_estimate/1024:.0f} KB")

    if output_path:
        decoder.save(output_path)
        print(f"[WFST] Saved to {output_path}")

    return decoder


import numpy as np
