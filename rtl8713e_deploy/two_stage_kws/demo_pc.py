#!/usr/bin/env python3
"""
PC 端 Demo — 测试两级语音唤醒+命令识别流水线

用法:
    # 麦克风实时测试
    python demo_pc.py --mic

    # 音频文件测试
    python demo_pc.py --wav test_audio.wav

    # 录音交互模式
    python demo_pc.py --record

依赖:
    pip install sounddevice soundfile
"""

import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_wakeword import UltraTinyWakeWord, WakeWordDetector
from stage2_command import CTCEncoder, CommandRecognizer
from wfst_decoder import WFSTGrammarDecoder
from unified_pipeline import TwoStagePipeline, PipelineState


class VoiceDemo:
    """PC 端演示器"""

    def __init__(self, stage1_ckpt, stage2_ckpt, grammar_path,
                 wake_words="小倍小倍", device='cpu'):
        self.device = device

        # --- 加载 Stage 1 ---
        ckpt1 = torch.load(stage1_ckpt, map_location=device)
        n_classes = ckpt1['num_classes']
        self.model1 = UltraTinyWakeWord(num_wake_words=n_classes, n_mels=40, size='micro')
        self.model1.load_state_dict(ckpt1['model_state_dict'])
        self.model1.eval().to(device)

        self.wake_detector = WakeWordDetector(self.model1,
                                              wake_labels=wake_words.split(','))

        # --- 加载 Stage 2 ---
        ckpt2 = torch.load(stage2_ckpt, map_location=device)
        n_tokens = len(ckpt2['tokenizer']['c2i'])
        self.model2 = CTCEncoder(num_tokens=n_tokens)
        self.model2.load_state_dict(ckpt2['model_state_dict'])
        self.model2.eval().to(device)

        # Tokenizer and WFST
        self.token_info = ckpt2['tokenizer']
        self.decoder = WFSTGrammarDecoder.load(grammar_path)

        self.cmd_recognizer = CommandRecognizer(plan='B')

        # --- Pipeline ---
        self.pipeline = TwoStagePipeline(self.wake_detector, None)

        # Callbacks
        self.pipeline.on_wake = self._on_wake
        self.pipeline.on_command = self._on_command

        self.sr = 16000

    def _on_wake(self, label, confidence):
        print(f"\n  Wake! '{label}' (confidence: {confidence:.3f})")
        print(f"  Listening for command...")

    def _on_command(self, text, confidence):
        print(f"\n  >>> Command: {text} (confidence: {confidence:.3f})")

    def process_audio_file(self, wav_path):
        """从 WAV 文件离线测试"""
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype='float32')
        if sr != self.sr:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * self.sr / sr))

        if audio.ndim > 1:
            audio = audio[:, 0]

        self.pipeline.start()
        print(f"Processing: {wav_path} ({len(audio)/self.sr:.1f}s)")

        hop = int(self.sr * 0.01)  # 10ms
        for i in range(0, len(audio) - hop, hop):
            chunk = audio[i:i + hop]
            if len(chunk) < hop:
                break

            # Mel 提取 (简化, 复现 40-bin log-mel)
            mel_frame = self._extract_mel_frame(chunk)
            result = self.pipeline.feed(mel_frame)
            if result:
                print(f"  [{i/self.sr:.1f}s] {result['type']}: {result}")
                if result['type'] == 'command':
                    break

        self.pipeline.stop()
        return result

    def run_microphone(self, duration_seconds=30):
        """实时麦克风测试"""
        try:
            import sounddevice as sd
        except ImportError:
            print("请安装: pip install sounddevice")
            return

        self.pipeline.start()
        print(f"\n{'='*50}")
        print(f"  实时麦克风测试 ({duration_seconds}s)")
        print(f"  唤醒词: '小倍小倍'")
        print(f"  请对着麦克风说话...")
        print(f"{'='*50}\n")

        hop = int(self.sr * 0.01)
        buffer = np.zeros(0, dtype=np.float32)
        last_state = None

        with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32',
                            blocksize=hop, callback=None) as stream:
            start = time.time()
            while time.time() - start < duration_seconds:
                data, overflow = stream.read(hop)
                if overflow:
                    continue

                audio = data.flatten()
                mel_frame = self._extract_mel_frame(audio)
                result = self.pipeline.feed(mel_frame)

                # 显示状态变化
                state = self.pipeline.get_state()
                if state != last_state:
                    print(f"  [{time.time()-start:.1f}s] State: {state.value}")
                    last_state = state

                if result:
                    print(f"\n  >>> {result}\n")
                    if result['type'] == 'command':
                        break

        self.pipeline.stop()
        print("\nDone.")

    def run_record_mode(self):
        """按键录音模式 (按 Enter 开始说话, 自动识别)"""
        self.pipeline.start()
        hop = int(self.sr * 0.01)
        buffer = []

        print(f"\n{'='*50}")
        print(f"  录音交互模式")
        print(f"  唤醒词: '小倍小倍'")
        print(f"  按 Enter → 说唤醒词 → 听到'滴' → 说命令")
        print(f"  按 Ctrl+C 退出")
        print(f"{'='*50}\n")

        try:
            while True:
                input("按 Enter 开始...")
                print("  说唤醒词...")

                # 录 3 秒, 检测唤醒词
                import sounddevice as sd
                audio = sd.rec(int(self.sr * 3), samplerate=self.sr,
                               channels=1, dtype='float32')
                sd.wait()

                audio = audio.flatten()
                for i in range(0, len(audio) - hop, hop):
                    chunk = audio[i:i + hop]
                    if len(chunk) < hop:
                        break
                    mel = self._extract_mel_frame(chunk)
                    result = self.pipeline.feed(mel)
                    if result and result['type'] == 'wake':
                        print(f"  Wake! ({result['confidence']:.2f})")
                        print("  (滴) 说命令...")

                        # 录 3 秒命令
                        cmd_audio = sd.rec(int(self.sr * 3), samplerate=self.sr,
                                           channels=1, dtype='float32')
                        sd.wait()
                        cmd_audio = cmd_audio.flatten()

                        for j in range(0, len(cmd_audio) - hop, hop):
                            chunk2 = cmd_audio[j:j + hop]
                            if len(chunk2) < hop:
                                break
                            mel2 = self._extract_mel_frame(chunk2)
                            result2 = self.pipeline.feed(mel2)
                            if result2 and result2['type'] == 'command':
                                break
                        break

        except KeyboardInterrupt:
            self.pipeline.stop()
            print("\nDone.")

    def _extract_mel_frame(self, audio_chunk):
        """提取单帧 40-dim Mel 特征 (在线流式)"""
        if not hasattr(self, '_audio_buffer'):
            self._audio_buffer = np.zeros(0, dtype=np.float32)
            self._prev_sample = 0.0

        # 累积音频
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        # 至少需要 25ms 窗口 (400 samples) 才能提取一帧 Mel
        win_samples = 400
        hop_samples = 160
        if len(self._audio_buffer) < win_samples:
            return np.zeros(40, dtype=np.float32)  # 不够, 返回零向量

        # 取可用音频
        audio = self._audio_buffer.copy()

        # Pre-emphasis
        emph = np.zeros_like(audio)
        emph[0] = audio[0]
        emph[1:] = audio[1:] - 0.97 * audio[:-1]

        # 单帧 STFT
        n_fft = 512
        frame = emph[:win_samples]
        frame = np.pad(frame, (0, max(0, n_fft - len(frame))))
        window = np.hanning(win_samples)
        frame[:win_samples] = frame[:win_samples] * window

        spec = np.fft.rfft(frame, n=n_fft)
        power = np.abs(spec) ** 2

        if not hasattr(self, '_mel_fb'):
            self._mel_fb = self._make_mel_fb_np(40, n_fft // 2 + 1, self.sr)

        mel = np.dot(self._mel_fb, power)
        mel = np.log(mel + 1e-6)

        # 滑动窗口: 丢掉 hop_samples
        self._audio_buffer = self._audio_buffer[hop_samples:]

        return mel.astype(np.float32)

    def _make_mel_fb_np(self, n_mels, n_freq, sr):
        mel = np.linspace(2595 * np.log10(1 + 80 / 700),
                           2595 * np.log10(1 + 7600 / 700), n_mels + 2)
        hz = 700 * (10 ** (mel / 2595) - 1)
        bins = np.floor(n_freq * hz / (sr / 2)).astype(int)
        fb = np.zeros((n_mels, n_freq))
        for m in range(n_mels):
            for k in range(bins[m], bins[m + 1]):
                fb[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
            for k in range(bins[m + 1], min(bins[m + 2], n_freq)):
                fb[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)
        return fb


def parse_args():
    p = argparse.ArgumentParser(description='Voice KWS Demo')
    p.add_argument('--mic', action='store_true', help='实时麦克风测试')
    p.add_argument('--record', action='store_true', help='按键录音模式')
    p.add_argument('--wav', type=str, help='测试 WAV 文件路径')
    p.add_argument('--stage1', default='checkpoints/stage1/best_model.pt',
                   help='Stage1 checkpoint')
    p.add_argument('--stage2', default='checkpoints/stage2/best_model.pt',
                   help='Stage2 checkpoint')
    p.add_argument('--grammar', default='checkpoints/stage2/grammar.json',
                   help='WFST grammar')
    return p.parse_args()


def main():
    args = parse_args()

    # 路径解析 (支持相对路径, 从 cwd 或脚本目录查找)
    base = os.getcwd()
    s1 = args.stage1 if os.path.isabs(args.stage1) else os.path.join(base, args.stage1)
    s2 = args.stage2 if os.path.isabs(args.stage2) else os.path.join(base, args.stage2)
    gr = args.grammar if os.path.isabs(args.grammar) else os.path.join(base, args.grammar)

    if not os.path.exists(s1):
        print(f"Stage1 not found: {s1}")
        return
    if not os.path.exists(s2):
        print(f"Stage2 not found: {s2}")
        return

    demo = VoiceDemo(s1, s2, gr, wake_words="小倍小倍")

    if args.mic:
        demo.run_microphone()
    elif args.record:
        demo.run_record_mode()
    elif args.wav:
        demo.process_audio_file(args.wav)
    else:
        print("用法:")
        print("  python demo_pc.py --mic          # 实时麦克风")
        print("  python demo_pc.py --record       # 按键录音")
        print("  python demo_pc.py --wav test.wav # 文件测试")


if __name__ == '__main__':
    main()
