# audio-lang-tools

Machine checks for text-to-speech clips in tonal languages, so a human listener only hears the doubtful ones. Cantonese (jyutping, six tones) is the only language so far. Used by learn-cantonese (`~/learn-cantonese/tools/audio-check.mjs` calls `altools check`), but it knows nothing about that site: it checks clips against expected text.

## Layout
```
audio_lang_tools/
  cli.py            altools check | bench make/train/score/items
  check.py          Checker: file checks + word check + tones for one clip; Analyser caches alignment and pitch
  audio.py          ffmpeg decode, file checks (silent, clipped, too short/long)
  align.py          forced alignment (torchaudio MMS_FA) → syllable and rhyme spans
  pitch.py          pitch trackers (ALTOOLS_PITCH): rmvpe (default), crepe-tiny, crepe-full
  rmvpe.py          RMVPE, vendored from RVC (MIT), CPU only
  tones.py          features per syllable, an abs classifier per voice, a shape classifier, thresholds
  stt.py            Azure speech-to-text word check (cached)
  tts.py            Azure TTS with sapi phonemes (benchmark clips)
  bench.py          the benchmark: make, train, score
  lang/cantonese.py jyutping, tones, attested syllables (pycantonese), voice GAPS
  config.py         cache dir and keys
bench/fixtures.json, bench/fixtures/   real cases with a known answer
bench/thresholds.json                   scores `bench score` must keep
bench/old-checker.mjs                   scores learn-cantonese's first checker (the baseline)
tests/              pytest on synthetic signals: no network, no Azure (RMVPE's test needs its weights)
experiments/        tried ideas, with their results in the docstring
```

## Rules
- Measure every change: `uv run altools bench score` must pass (it exits 1 below `bench/thresholds.json`). Raise a threshold when a change beats it; never lower one to make a change pass. Report before/after numbers.
- Tune on the train split only (`bench train` picks thresholds from cross-validated predictions). The test split is for scoring.
- Tests must be able to fail: after writing one, break the code on purpose and confirm it fails.
- Keys: `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` from the environment or `~/.config/audio-lang-tools/config.env` (or `$ALTOOLS_CONFIG`). Never put a key in any file in the repo, never print one. `tests/test_no_keys.py` fails if a file holds one.
- Azure costs the user money: each run makes at most `ALTOOLS_AZURE_BUDGET` new (uncached) requests, default 1000, and reports what it used. Rebuilding the benchmark needs about 12,000 TTS requests plus ~3,000 speech-to-text on scoring: ask the user before raising the budget for that. Never delete the cache (`~/.cache/audio-lang-tools/tts`, `stt`): it is what makes reruns free.
- Generated things (models, Azure results, benchmark clips, analysis) live in `~/.cache/audio-lang-tools` (`$ALTOOLS_CACHE`), never in the repo.
- Adding a language: a `lang/<name>.py` with the same functions as `lang/cantonese.py`, its own fixtures and thresholds.

## Commands
```sh
uv sync                                   # first time: CPU torch, models download on first use (~1.2 GB aligner)
uv run --group dev pytest -q tests
uv run altools bench make --words words.json --voice zh-HK-HiuMaanNeural --voice zh-HK-WanLungNeural --voice zh-HK-HiuGaaiNeural
                                          # words: [{ id, text, jyutping, plain }]; the first voice is the main one
uv run altools bench train
uv run altools bench score
uv run altools check < items.json > results.json
```
