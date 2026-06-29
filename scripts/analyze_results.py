"""
analyze_results.py

Merges model predictions with subgroup metadata (Fitzpatrick type, lighting,
face-detection validity) and produces two distinct layers of result:

  Layer 1 (feasibility): how often does MediaPipe face detection fail/degrade,
  broken down by lighting and skin tone? This is reported on its own.

  Layer 2 (RQ1-4, model comparison): MAE/RMSE/Pearson/SNR per model, per
  subgroup, computed ONLY on windows where face_detection_ok == True. This
  is the layer that actually answers the thesis's research questions.

Run this after pulling results.zip back from Kaggle.
"""

import pandas as pd
from scipy import stats

# --- Config: expected file locations -----------------------------------
PREDICTIONS_DIR = "results/predictions"
METADATA_PATH = "data/metadata/mmpd_subgroups.csv"
STATS_OUT_DIR = "results/stats"

MODELS = ["deepphys", "physnet", "efficientphys"]
DATASETS = ["UBFC", "MMPD"]


def load_predictions(model: str, dataset: str) -> pd.DataFrame:
    """Load one model's predictions for one dataset.

    Expected columns: window_id, subject_id, pred_hr, gt_hr
    Adjust path/column names once actual toolbox output format is confirmed.
    """
    path = f"{PREDICTIONS_DIR}/{dataset.lower()}_{model}_predictions.csv"
    return pd.read_csv(path)


def load_metadata() -> pd.DataFrame:
    """Load subgroup metadata: window_id, fitzpatrick, light, motion,
    face_detection_ok, mean_confidence, detection_rate, bbox_jitter.
    Only applies to MMPD - UBFC-rPPG has no subgroup labels (it's the
    indoor baseline, not part of the subgroup RQs).
    """
    return pd.read_csv(METADATA_PATH)


def layer1_feasibility_report(merged: pd.DataFrame) -> pd.DataFrame:
    """% of windows flagged face_detection_ok == False, by light x fitzpatrick.
    This is reported as its own finding - NOT folded into model comparison.
    """
    summary = (
        merged.groupby(["light", "fitzpatrick"])["face_detection_ok"]
        .agg(fail_rate=lambda x: (~x).mean(), n_windows="count")
        .reset_index()
    )
    return summary


def layer2_model_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE/Pearson per model x light x fitzpatrick, computed ONLY on
    valid (face_detection_ok == True) windows. This answers RQ1-4.
    """
    valid = merged[merged["face_detection_ok"] == True].copy()
    valid["abs_err"] = (valid["pred_hr"] - valid["gt_hr"]).abs()
    valid["sq_err"] = (valid["pred_hr"] - valid["gt_hr"]) ** 2

    summary = (
        valid.groupby(["model", "light", "fitzpatrick"])
        .agg(
            mae=("abs_err", "mean"),
            rmse=("sq_err", lambda x: x.mean() ** 0.5),
            n=("abs_err", "count"),
        )
        .reset_index()
    )

    # Pearson correlation per group (needs at least 2 points)
    pearsons = []
    for (model, light, fitz), grp in valid.groupby(["model", "light", "fitzpatrick"]):
        if len(grp) >= 2:
            r, _ = stats.pearsonr(grp["pred_hr"], grp["gt_hr"])
        else:
            r = float("nan")
        pearsons.append({"model": model, "light": light, "fitzpatrick": fitz, "pearson_r": r})

    summary = summary.merge(pd.DataFrame(pearsons), on=["model", "light", "fitzpatrick"])
    return summary


def paired_ttest_indoor_vs_outdoor(merged: pd.DataFrame, model: str) -> dict:
    """Paired t-test: per-subject MAE indoor vs outdoor, for one model.
    Only on valid windows. Requires matched subjects in both conditions.
    """
    valid = merged[(merged["face_detection_ok"] == True) & (merged["model"] == model)].copy()
    valid["abs_err"] = (valid["pred_hr"] - valid["gt_hr"]).abs()

    per_subject = valid.groupby(["subject_id", "light"])["abs_err"].mean().reset_index()
    pivot = per_subject.pivot(index="subject_id", columns="light", values="abs_err").dropna()

    if "indoor" not in pivot.columns or "outdoor" not in pivot.columns:
        return {"model": model, "error": "missing indoor or outdoor data for paired test"}

    t_stat, p_val = stats.ttest_rel(pivot["indoor"], pivot["outdoor"])
    return {
        "model": model,
        "n_subjects": len(pivot),
        "mean_indoor_mae": pivot["indoor"].mean(),
        "mean_outdoor_mae": pivot["outdoor"].mean(),
        "t_stat": t_stat,
        "p_value": p_val,
    }


if __name__ == "__main__":
    print("Skeleton ready. Fill in once predictions exist:")
    print("1. Load predictions for each model x dataset")
    print("2. Merge with metadata CSV on window_id")
    print("3. Run layer1_feasibility_report() -> save to results/stats/")
    print("4. Run layer2_model_comparison() -> save to results/stats/")
    print("5. Run paired_ttest_indoor_vs_outdoor() per model -> save to results/stats/")
