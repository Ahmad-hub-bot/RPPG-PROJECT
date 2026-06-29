# rPPG Outdoor/Skin-Tone Benchmark — Thesis Repo

Evaluating DeepPhys, PhysNet, and EfficientPhys (pretrained, no retraining) on
outdoor lighting and Fitzpatrick skin-tone subgroups, using MMPD as the primary
dataset and UBFC-rPPG as an indoor baseline.

## Workflow

1. **Develop locally in VS Code.** Edit code, configs, and scripts here.
   Test logic on tiny samples (2-3 clips) using CPU only.
2. **Push to GitHub** once a change runs cleanly on the small test sample.
3. **Run on Kaggle** for actual GPU inference. The Kaggle notebook
   (`notebooks/kaggle_runner.ipynb`) pulls this repo, attaches the relevant
   Kaggle Dataset (raw or preprocessed), and runs `scripts/run_all_inference.sh`.
4. **Pull results back** into `results/` and run `scripts/analyze_results.py`
   locally for subgroup breakdowns and statistical tests.

## Repo map

- `configs/infer_configs/` — one YAML per model x dataset combination
- `scripts/preprocess_subgroups.py` — builds Fitzpatrick/lighting metadata CSV
  + flags MediaPipe face-detection confidence per window (see note below)
- `scripts/run_all_inference.sh` — runs all 6 inference configs sequentially
- `scripts/analyze_results.py` — merges predictions + metadata, computes
  MAE/RMSE/Pearson/SNR per subgroup, runs paired t-tests
- `data/raw/` — NEVER committed (see .gitignore). Lives on Kaggle Datasets.
- `data/metadata/` — small CSVs (subject_id, fitzpatrick, light, motion,
  face_detection_ok) — these ARE committed, they're small
- `results/` — prediction CSVs and stats output, small enough to commit

## Important: face-detection validity check

Before trusting any MAE/RMSE result, especially for outdoor or dark-skin
subgroups, check that window's `face_detection_ok` flag in the metadata CSV.
A high HR error could mean the model genuinely struggled (a real finding) OR
that MediaPipe failed to find/track the face correctly (a preprocessing
artifact, not a model finding). These get reported as two separate layers
of result — see `scripts/analyze_results.py` for the split logic.

## Pretrained checkpoint provenance

Record which dataset each model's checkpoint was originally trained on
(e.g., DeepPhys-trained-on-PURE vs DeepPhys-trained-on-SCAMPS) in
`configs/infer_configs/` filenames and in the results CSV. Cross-dataset
generalization numbers vary a lot by training source — this must be tracked,
not assumed.

## Status

- [ ] Toolbox cloned and environment verified
- [ ] Pretrained checkpoints downloaded, training source confirmed
- [ ] UBFC-rPPG pipeline validation run (compare against published benchmark)
- [ ] MMPD access obtained
- [ ] MMPD pilot run (2 subjects) completed
- [ ] MMPD subgroup metadata CSV built
- [ ] Full MMPD preprocessing + inference run
- [ ] Subgroup analysis + statistical tests complete
