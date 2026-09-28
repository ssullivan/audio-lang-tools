"""Forced alignment: where each syllable is in a clip.

Uses torchaudio's MMS_FA (Meta's MMS wav2vec2 model trained for alignment
in 1,100+ languages, on text in Latin letters). The toneless romanization
of each syllable is aligned to the audio. CTC marks each letter at a
single frame (20 ms), so a syllable runs from its first letter to the
next syllable's first letter (the last one to the end of speech), and its
rhyme (the part carrying the tone) from its first rhyme letter.

    Aligner().align(x, sylls, rhyme_at) → [{start, rhyme, end, score}] (s)
"""
import numpy as np
import torch

from .audio import RATE, speech_span


class Aligner:
    def __init__(self):
        from torchaudio.pipelines import MMS_FA
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.model = MMS_FA.get_model(with_star=False).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()

    def align(self, x: np.ndarray, words: list[str], rhyme_at: list[int]) -> list[dict]:
        """words: toneless syllables ("neoi"); rhyme_at: index of each one's
        first rhyme letter."""
        with torch.inference_mode():
            em, _ = self.model(torch.from_numpy(x).unsqueeze(0))
            spans = self.aligner(em[0], self.tokenizer(words))
        sec = len(x) / em.shape[1] / RATE
        _, speech_end = speech_span(x)
        out = []
        for i, sp in enumerate(spans):
            start = sp[0].start * sec
            end = spans[i + 1][0].start * sec if i + 1 < len(spans) else max(speech_end, sp[-1].end * sec)
            rhyme = sp[min(rhyme_at[i], len(sp) - 1)].start * sec
            score = float(np.mean([t.score for t in sp]))
            out.append({'start': start, 'rhyme': rhyme, 'end': end, 'score': score})
        return out
