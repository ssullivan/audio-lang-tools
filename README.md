# audio-lang-tools

Machine checks for text-to-speech clips in tonal languages (Cantonese so far), so a human listener only needs to hear the doubtful ones, with a benchmark that measures how often the checks are right.

For each clip and its expected jyutping:

1. **File**: decode with ffmpeg; flag broken, silent, clipped, or too short or long for its syllable count.
2. **Words**: Azure speech-to-text with no hint; compare its syllables with the expected ones, ignoring tone.
3. **Tones**:
   - find each syllable by forced alignment (torchaudio's MMS_FA, on the toneless jyutping);
   - track pitch with RMVPE (vendored; CREPE is an option);
   - give the probability of each tone with a gradient-boosted classifier on the rhyme's pitch contour and its neighbours'. There is one model per voice, trained on that voice from clips whose tones are known (Azure honours tones given as sapi phonemes), and a shape-only model for any other voice.

A clearly unlikely tone is **CHECK**; a doubtful one is **LISTEN**.

## Benchmark
`altools bench make` builds labelled clips from a word list: each word read right, and read with one syllable in another tone it really has, plus single syllables and random strings for training, damaged files, and real fixtures (`bench/fixtures.json`). Words are split into train and test by hash. `altools bench score` reports:
- false flags on right clips;
- wrong-tone recall by syllable count;
- file-fault recall;
- fixtures;
- a tone confusion matrix.

It exits 1 below `bench/thresholds.json`.

## Results (test split, 2026-09-28)
Tone messages only. CHECK means likely wrong; "any" includes LISTEN.

| | Old checker (learn-cantonese, rules) | CREPE tiny | **RMVPE, per-voice models** |
|---|---|---|---|
| Sapi-read wrong tone CHECKed, 1 / 2–3 / 4+ syllables | 60 / 50 / 0% | 84 / 84 / 77% | **86 / 92 / 85%** |
| Sapi-read right clips CHECKed | 1.5% | 3.1% | 3.1% |
| Character-read wrong tone CHECKed, 1 / 2–3 / 4+ syllables | – | 92 / 75 / 65% | **95 / 84 / 82%** |
| Character-read wrong tone flagged at all | – | 95 / 87 / 89% | **100 / 94 / 98%** |
| Character-read right clips CHECKed / flagged | – | 8.4 / 21% | 8.4 / 21% |
| Tone accuracy per syllable (sapi, HiuMaan) | – | 90.6% | **94.2%** |
| Fixtures right (real cases) | 9 of 11 | 9 of 11 | **11 of 11** |
| Damaged files caught | 100% | 100% | 100% |

WanLung and HiuGaai (sapi-read) score 91/91/82% and 91/84/79% wrong tones CHECKed, with 5% of right clips CHECKed.

Character-read clips are the realistic measure: labels come from the text, so some "false" flags are the voice really saying something else (a question-final 呀 said high, a colloquial reading). Tried and not adopted: comparing each clip with the voice's own readings of every tone variant (analysis by synthesis, `experiments/synth_compare.py`): AUC 0.935 against the classifier's 0.939, and only +4% combined, at ten extra TTS calls a clip.

## Setup
```sh
uv sync                                   # Python 3.12, CPU torch
uv run --group dev pytest -q tests
```
Keys: `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` in the environment or `~/.config/audio-lang-tools/config.env`. Models and caches go to `~/.cache/audio-lang-tools`.

See CLAUDE.md for the layout and rules.
