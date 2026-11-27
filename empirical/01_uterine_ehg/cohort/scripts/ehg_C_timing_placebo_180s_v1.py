#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NNDR EHG | 180-s C-exclusion sensitivity + fixed-cohort 120-s comparator. (Spyder F5; no NNDR refit).

Inputs: ehg_final_outputs/records/<record>/windows.csv and annotations.csv,
        plus ehg_final_outputs/dataset_C_marker_all_rows_screening.csv.

The REAL C markers are those accepted by EHG v2 strict screening. The 180-s
sensitivity placebo samples the same number of pseudo-markers per included recording,
uniformly over eligible 5-s window endpoints, with at least 240 s between
pseudo-markers. Pseudo-markers avoid true C centers (180 s), (c) (90 s), and
movement/equipment markers (150 s). Each placebo recording has exactly the
same number of tested events as the observed counterpart. Recordings that
cannot supply this many placebo times are excluded from BOTH arms and logged.

A separate activity-tilted sensitivity null samples from the same candidate
pool, favoring candidates with a similar within-record EHG-activity-quintile
mix. This is NOT exact matching; the achieved balance is reported.

For each event, compare medians of trailing-60-s-window diagnostics indexed by
window END time in: pre [-120,-60], marker [-20,20], post [60,120] s. Require
>=3 admissible points per phase for BOTH RA and R_sigma in BOTH M2/M3
at observed events and pseudo-events. Subject is the independent unit: event -> record -> subject
median, then median across subjects. Includes distinct null distributions for
rise (marker-pre) and decline (post-marker). A decline is not assumed.

This is a timing-association sensitivity control, not a model-adequacy proof or a precise
physiological onset test. EHG `C` timestamps may be imprecise. Centered
activity smoothing is descriptive and not causal.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings
import hashlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ======================= SPYDER F5 CONFIGURATION ===========================
HERE = Path(__file__).resolve().parent
# Works when the script is either inside ehg_final_outputs or beside it.
if (HERE / "records").is_dir():
    DATASET_OUTPUT = HERE
elif (HERE / "ehg_final_outputs" / "records").is_dir():
    DATASET_OUTPUT = HERE / "ehg_final_outputs"
else:
    raise FileNotFoundError(
        "Place this script INSIDE ehg_final_outputs or in its PARENT folder. "
        "Expected ehg_final_outputs/records/<record>/windows.csv."
    )
RECORD_DIR = DATASET_OUTPUT / "records"
OUTPUT_DIR = DATASET_OUTPUT / "C_timing_placebo_180s_v1"

SEED = 20261119
N_PLACEBO = 2000              # use 100 for quick smoke test; 2000 for final
N_SUBJECT_BOOT = 3000        # 95% bootstrap CI for observed subject median
PHASE_STEP_SEC = 5.0
PHASES = {
    "pre": (-120.0, -60.0),
    "marker": (-20.0, 20.0),
    "post": (60.0, 120.0),
}
MIN_PHASE_POINTS = 3
ALIGN_TOL_SEC = 2.51
C_EXCLUSION_SEC = 180.0    # STRICT sensitivity; placebo center versus ALL real C centers
PLACEBO_SEPARATION_SEC = 240.0
ARTIFACT_EXCLUSION_SEC = 150.0
POSSIBLE_C_EXCLUSION_SEC = 90.0
SAVE_EXAMPLE_REPLICATE = True
FAIL_ON_INPUT_ERRORS = True  # do not silently discard missing/corrupted cached records
RUN_ACTIVITY_TILTED_NULL = True
RUN_MATCHED_120_COMPARATOR = True  # comparison on EXACT 180-s observed-event cohort
# Activity quintile reweighting is an approximate balance sensitivity,
# not exact matched sampling. Its realized balance must be inspected.
ACTIVITY_WEIGHT_SMOOTH = 0.02
ACTIVITY_WEIGHT_CAP = (0.05, 20.0)
ACTIVITY_BALANCE_WARNING = 0.15  # mean absolute quintile-frequency difference

METHODS = ("M2", "M3")
METRICS = ("RA", "R_sigma", "Delta", "support")
EFFECTS = ("marker_minus_pre", "post_minus_marker")
COLORS = {"M2": "#E69F00", "M3": "#009E73"}

# ============================= INPUT HELPERS ===============================

def parse_bool(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes")).to_numpy()


def read_strict_markers() -> pd.DataFrame:
    f = DATASET_OUTPUT / "dataset_C_marker_all_rows_screening.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}; run the v2 EHG cohort first.")
    d = pd.read_csv(f)
    needed = {"record", "time_sec", "strict_accepted"}
    if not needed.issubset(d):
        raise ValueError(f"Marker CSV missing {needed-set(d.columns)}")
    d = d.loc[parse_bool(d["strict_accepted"])].copy()
    d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
    return d.loc[np.isfinite(d.time_sec)].sort_values(["record", "time_sec"])


def read_window_data(stem: str) -> dict[str, pd.DataFrame]:
    path = RECORD_DIR / stem / "windows.csv"
    fields = {"time_end_sec", "activity_at_end", "method", *METRICS}
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, usecols=lambda c: c in fields | {
        "RA_status", "R_sigma_status", "method_status", "reduced_stable"})
    if not fields.issubset(d.columns):
        raise ValueError(f"{stem}: missing {fields - set(d.columns)}")
    d["time_end_sec"] = pd.to_numeric(d["time_end_sec"], errors="coerce")
    for metric in METRICS:
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
    if "RA_status" in d:
        d.loc[d.RA_status.ne("ok"), "RA"] = np.nan
    if "R_sigma_status" in d:
        d.loc[d.R_sigma_status.ne("ok"), "R_sigma"] = np.nan
    if "reduced_stable" in d:
        bad = ~parse_bool(d.reduced_stable)
        d.loc[bad, ["RA", "R_sigma"]] = np.nan
    out = {}
    for method in METHODS:
        sub = d.loc[d.method.eq(method)].sort_values("time_end_sec").reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"{stem}: missing method {method}")
        if sub.time_end_sec.duplicated().any():
            raise ValueError(f"{stem}/{method}: duplicate time_end_sec")
        out[method] = sub
    t2 = out["M2"].time_end_sec.to_numpy(float)
    t3 = out["M3"].time_end_sec.to_numpy(float)
    if not (len(t2) == len(t3) and np.allclose(t2, t3, atol=1e-6, rtol=0)):
        raise ValueError(f"{stem}: M2/M3 time grids differ; no silent interpolation")
    return out


def read_annotations(stem: str) -> dict[str, np.ndarray]:
    path = RECORD_DIR / stem / "annotations.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    a = pd.read_csv(path, keep_default_na=False)
    if not {"time_sec", "label"}.issubset(a.columns):
        raise ValueError(f"{stem}: annotation CSV needs time_sec and label")
    a["label"] = a["label"].astype(str).str.replace("\\x00", "", regex=False).str.strip()
    a["time_sec"] = pd.to_numeric(a.time_sec, errors="coerce")
    a = a.loc[np.isfinite(a.time_sec)]
    return {
        "C": a.loc[a.label.eq("C"), "time_sec"].to_numpy(float),
        "possible": a.loc[a.label.eq("(c)"), "time_sec"].to_numpy(float),
        "artifact": a.loc[a.label.isin(("pm", "pos", "em")), "time_sec"].to_numpy(float),
    }


def far_from_all(t: np.ndarray, forbidden: np.ndarray, seconds: float) -> np.ndarray:
    if not len(forbidden):
        return np.ones(len(t), dtype=bool)
    forbidden = np.sort(np.asarray(forbidden, dtype=float))
    ix = np.searchsorted(forbidden, t)
    d1 = np.where(ix > 0, np.abs(t - forbidden[np.clip(ix-1, 0, len(forbidden)-1)]), np.inf)
    d2 = np.where(ix < len(forbidden),
                  np.abs(t - forbidden[np.clip(ix, 0, len(forbidden)-1)]), np.inf)
    return np.minimum(d1, d2) >= seconds


# ============================= EVENT EFFECTS ===============================

def phase_medians_for_events(group: pd.DataFrame, event_times: np.ndarray) -> np.ndarray:
    """Return event x metric x phase; exact v2 window-END and nearest-grid rule."""
    times = group.time_end_sec.to_numpy(float)
    event_times = np.asarray(event_times, float)
    ans = np.full((len(event_times), len(METRICS), len(PHASES)), np.nan)
    if len(event_times) == 0:
        return ans
    for k, (phase, (lo, hi)) in enumerate(PHASES.items()):
        taus = np.arange(lo, hi + .001, PHASE_STEP_SEC)
        targets = event_times[:, None] + taus[None, :]
        right = np.searchsorted(times, targets, side="left")
        left = np.clip(right - 1, 0, len(times) - 1)
        right = np.clip(right, 0, len(times) - 1)
        idx = np.where(np.abs(times[right]-targets) < np.abs(times[left]-targets), right, left)
        time_ok = np.abs(times[idx]-targets) <= ALIGN_TOL_SEC
        for m, metric in enumerate(METRICS):
            vals = group[metric].to_numpy(float)[idx].copy()
            vals[~time_ok] = np.nan
            nvalid = np.isfinite(vals).sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(vals, axis=1)
            med[nvalid < MIN_PHASE_POINTS] = np.nan
            ans[:, m, k] = med
    return ans


def all_effects(groups: dict[str, pd.DataFrame], times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns event x method x metric x effect, and event x method x metric x phase."""
    p = np.stack([phase_medians_for_events(groups[method], times) for method in METHODS], axis=1)
    eff = np.stack([p[..., 1]-p[..., 0], p[..., 2]-p[..., 1]], axis=-1)
    return eff, p


def activity_at_marker(g: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    tt = g.time_end_sec.to_numpy(float)
    idx = np.searchsorted(tt, times)
    l = np.clip(idx-1, 0, len(tt)-1)
    r = np.clip(idx, 0, len(tt)-1)
    idx = np.where(np.abs(tt[r]-times) < np.abs(tt[l]-times), r, l)
    return g.activity_at_end.to_numpy(float)[idx]


def make_activity_weights(all_ref_activity, cand_activity, real_activity):
    """Approximate activity-quintile balancing. No outcome values used."""
    v = np.asarray(all_ref_activity, float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.ones(len(cand_activity)), np.zeros(len(cand_activity), int), np.zeros(len(real_activity), int)
    edges = np.quantile(v, [.2, .4, .6, .8])
    cbin = np.searchsorted(edges, cand_activity, side="right")
    rbin = np.searchsorted(edges, real_activity, side="right")
    pobs = np.bincount(rbin, minlength=5) / max(1,len(rbin))
    pcand = np.bincount(cbin, minlength=5) / max(1,len(cbin))
    weights_by_bin = np.clip((pobs+ACTIVITY_WEIGHT_SMOOTH)/(pcand+ACTIVITY_WEIGHT_SMOOTH),
                             *ACTIVITY_WEIGHT_CAP)
    return weights_by_bin[cbin], cbin, rbin


# ===== EXACT fixed-count sampling subject to 240-s separation ==============
class SeparatedSampler:
    """Exact weighted fixed-cardinality subset sampler via suffix DP.

    Weight of a feasible sample is product of its candidate weights.
    Uniform weights gives uniform sampling among all feasible k-subsets,
    not an unreported greedy approximation.
    """
    def __init__(self, times, k, separation, weights=None):
        self.t = np.asarray(times, float)
        self.k = int(k)
        self.n = len(self.t)
        if self.n and (np.diff(self.t) < 0).any():
            raise ValueError("Candidate times must be sorted")
        self.successor = np.searchsorted(self.t, self.t + separation - 1e-9, side="left")
        w = np.ones(self.n) if weights is None else np.asarray(weights, float)
        if not (len(w) == self.n and np.all(np.isfinite(w)) and np.all(w > 0)):
            raise ValueError("Candidate weights must be finite and positive")
        self.logw = np.log(w)
        self.dp = np.full((self.n+1, self.k+1), -np.inf)
        self.dp[:, 0] = 0.0
        for i in range(self.n-1, -1, -1):
            for j in range(1, self.k+1):
                take = self.logw[i] + self.dp[self.successor[i], j-1]
                skip = self.dp[i+1, j]
                self.dp[i, j] = np.logaddexp(skip, take)
        self.possible = self.k == 0 or bool(np.isfinite(self.dp[0, self.k]))
        self.first = {
            j: self.logw + self.dp[self.successor, j-1]
            for j in range(1, self.k+1)
        }

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if not self.possible:
            raise RuntimeError("Cannot place requested event count with the separation rule")
        ix, need, picks = 0, self.k, []
        while need:
            scores = self.first[need][ix:]
            finite = np.isfinite(scores)
            if not np.any(finite):
                raise AssertionError("DP sampler reached infeasible state")
            maxlog = np.max(scores[finite])
            probs = np.exp(scores-maxlog)
            probs[~finite] = 0
            cdf = np.cumsum(probs)
            pos = min(int(np.searchsorted(cdf, rng.random()*cdf[-1])), len(scores)-1)
            chosen = ix+pos
            picks.append(chosen)
            ix = int(self.successor[chosen])
            need -= 1
        return np.asarray(picks, int)


# ======================== RECORD PREPARATION ================================

def prepare_one(stem: str, marker_times: np.ndarray):
    groups = read_window_data(stem)
    ann = read_annotations(stem)
    t = groups["M2"].time_end_sec.to_numpy(float)
    marker_times = np.sort(np.unique(np.asarray(marker_times, float)))
    observed_effects, observed_phases = all_effects(groups, marker_times)
    # Joint RA/R_sigma phase admissibility across BOTH methods is fixed before
    # selecting the placebo-comparable cohort. Not based on response sign/size.
    keep_observed = np.all(np.isfinite(observed_phases[:, :, :2, :]), axis=(1,2,3))
    true_times = marker_times[keep_observed]
    observed_effects = observed_effects[keep_observed]
    if not len(true_times):
        return None, dict(record=stem, status="no_common_valid_observed_C",
                          n_strict_markers=len(marker_times), n_eligible_observed=0)
    # Candidate endpoints need same full +/-120-s timeline support as original.
    mask = (t >= t[0]+120) & (t <= t[-1]-120)
    n_before_exclusions = int(mask.sum())
    mask &= far_from_all(t, ann["C"], C_EXCLUSION_SEC)
    mask &= far_from_all(t, ann["possible"], POSSIBLE_C_EXCLUSION_SEC)
    mask &= far_from_all(t, ann["artifact"], ARTIFACT_EXCLUSION_SEC)
    candidates = t[mask]
    candidate_effects, candidate_phases = all_effects(groups, candidates)
    good = np.all(np.isfinite(candidate_phases[:, :, :2, :]), axis=(1,2,3))
    candidates, candidate_effects = candidates[good], candidate_effects[good]
    activity_candidate = activity_at_marker(groups["M2"], candidates)
    activity_observed = activity_at_marker(groups["M2"], true_times)
    aw, cbin, rbin = make_activity_weights(
        groups["M2"].activity_at_end.to_numpy(float),
        activity_candidate, activity_observed)
    n = len(true_times)
    uniform = SeparatedSampler(candidates, n, PLACEBO_SEPARATION_SEC)
    if not uniform.possible:
        return None, dict(record=stem, status="insufficient_separated_placebos",
                          n_strict_markers=len(marker_times), n_eligible_observed=n,
                          n_candidates=len(candidates), n_timeline_candidates=n_before_exclusions)
    tilted = SeparatedSampler(candidates, n, PLACEBO_SEPARATION_SEC, aw)
    out = dict(record=stem, subject_id=stem.split('_')[0], record_type=stem.split('_')[1],
               real_times=true_times, real_effects=observed_effects, candidates=candidates,
               cand_effects=candidate_effects, candidate_activity=activity_candidate,
               real_activity=activity_observed, cbin=cbin, rbin=rbin,
               uniform=uniform, tilted=tilted)
    audit = dict(record=stem, subject_id=out['subject_id'], status="included",
                 n_strict_markers=len(marker_times), n_eligible_observed=n,
                 n_candidates=len(candidates), n_timeline_candidates=n_before_exclusions,
                 n_true_C=len(ann['C']), n_possible_c=len(ann['possible']),
                 n_motion_equipment=len(ann['artifact']),
                 max_possible_placebo=int(max_separated_count(candidates, PLACEBO_SEPARATION_SEC)))
    return out, audit


def max_separated_count(t, sep):
    if not len(t): return 0
    count, last = 1, t[0]
    for x in t[1:]:
        if x-last >= sep-1e-9: count += 1; last = x
    return count


# ========================= COHORT STATISTIC =================================

def verify_observed_against_saved_v2(records):
    """Compare our independent event phase effects with the v2 saved effects.

    This is a numerical agreement check, NOT independent data replication.
    """
    path = DATASET_OUTPUT / 'dataset_C_strict_event_phase_effects.csv'
    if not path.exists():
        print('Source event-phase CSV unavailable: exact v2 comparison skipped.')
        return pd.DataFrame()
    old = pd.read_csv(path)
    rows = []
    for rec in records:
        stem = rec['record']
        raw = old.loc[old.record.eq(stem)]
        for j,t in enumerate(rec['real_times']):
            for mi,method in enumerate(METHODS):
                sel = raw.loc[(raw.method==method)&
                     np.isclose(raw.event_time_sec.to_numpy(float),t,rtol=0,atol=1e-4)]
                if len(sel)!=1:
                    raise RuntimeError(f'Observed marker missing or duplicated in saved v2 phase CSV: {stem} {method} {t}')
                for qi,metric in enumerate(METRICS):
                    for ei,eff in enumerate(EFFECTS):
                        col=f'{metric}_{eff}'
                        a=float(sel.iloc[0][col]); b=float(rec['real_effects'][j,mi,qi,ei])
                        if np.isfinite(a) != np.isfinite(b):
                            raise RuntimeError(f'Finite mismatch against saved v2 event effects: {stem} {method} {col} at {t}')
                        if np.isfinite(a):
                            err=abs(a-b)
                            rows.append(dict(record=stem,method=method,metric=metric,
                                             effect=eff,event_time_sec=t,abs_error=err))
                            if err > 1e-7:
                                raise RuntimeError(f'Numerical disagreement with v2 ({err}): {stem}, {col}, {t}')
    out = pd.DataFrame(rows)
    print('Independent reproduction of original v2 event contrasts:',len(out),
          'finite values, max abs error',out.abs_error.max() if len(out) else np.nan)
    return out


def record_effects_median(events):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(events, axis=0)  # method x metric x effect


def aggregate_subjects(records, rec_medians):
    ids = sorted(set(r['subject_id'] for r in records))
    byid = {sid: [] for sid in ids}
    for rec, row in zip(records, rec_medians):
        byid[rec['subject_id']].append(row)
    out = []
    for sid in ids:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            val = np.nanmedian(np.stack(byid[sid]), axis=0)
        out.append(val)
    return ids, np.stack(out)  # subject x method x metric x effect


def cohort_median(arr):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(arr, axis=0)


def plot_test(summary: pd.DataFrame, reps: pd.DataFrame, mode: str, out_stem: Path):
    plt.rcParams.update({'font.family':'sans-serif',
                         'font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
                         'font.size':8, 'axes.labelsize':8.5,
                         'axes.linewidth':.65, 'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(18/2.54, 11.5/2.54))
    combos = [("R_sigma","marker_minus_pre"), ("R_sigma","post_minus_marker"),
              ("RA","marker_minus_pre"), ("RA","post_minus_marker")]
    for i, (ax, (metric, effect)) in enumerate(zip(axes.flat,combos)):
        for method in METHODS:
            q = reps.loc[(reps.null_mode==mode)&(reps.method==method)&
                         (reps.metric==metric)&(reps.effect==effect), 'cohort_median'].to_numpy(float)
            s = summary.loc[(summary.null_mode==mode)&(summary.method==method)&
                            (summary.metric==metric)&(summary.effect==effect)]
            if not len(s): continue
            obs = float(s.iloc[0].observed)
            color = COLORS[method]
            ax.hist(q[np.isfinite(q)], bins=45, color=color, alpha=.22, density=True,
                    edgecolor='none')
            ax.axvline(obs,color=color,ls='-',lw=1.5)
        ax.axvline(0, color='0.55',lw=.8,ls=':')
        ax.set_title((r'$R_{\sigma}$' if metric=='R_sigma' else r'$R_A$')+
                     (' : marker $-$ pre' if effect=='marker_minus_pre' else ' : post $-$ marker'),
                     fontsize=9)
        ax.set_ylabel('Placebo density' if i%2==0 else '')
        ax.set_xlabel('Subject-level median difference')
        ax.grid(False)
        ax.text(-.14,1.06,'('+chr(97+i)+')',transform=ax.transAxes,
                fontsize=10,weight='bold',clip_on=False)
        for spine in ax.spines.values(): spine.set_linewidth(.65)
    handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=m+': observed | shaded: placebo')
             for m in METHODS]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,
               bbox_to_anchor=(.5,.01),fontsize=8)
    fig.subplots_adjust(left=.105,right=.985,top=.90,bottom=.23,wspace=.32,hspace=.62)
    fig.savefig(out_stem.with_suffix('.pdf'))
    fig.savefig(out_stem.with_suffix('.png'),dpi=400)
    plt.close(fig)


# ============================== MAIN ========================================

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    marker_df = read_strict_markers()
    stems = sorted(marker_df.record.unique())
    print(f'Strict C markers in {len(stems)} recordings; checking cached 16-channel windows')
    records, audit = [], []
    for i, stem in enumerate(stems, 1):
        raw = marker_df.loc[marker_df.record.eq(stem), 'time_sec'].to_numpy(float)
        try:
            r, row = prepare_one(stem, raw)
            if r is not None: records.append(r)
            audit.append(row)
        except Exception as exc:
            audit.append(dict(record=stem,status='input_or_alignment_error',
                              n_strict_markers=len(raw),reason=f'{type(exc).__name__}: {exc}'))
            print('ERROR',stem,':',exc)
        if i%10==0 or i==len(stems):
            print(f'  processed {i}/{len(stems)} annotated recordings',flush=True)
    aud = pd.DataFrame(audit)
    aud.to_csv(OUTPUT_DIR/'placebo_eligibility_by_record.csv',index=False)
    if FAIL_ON_INPUT_ERRORS and (aud.status=='input_or_alignment_error').any():
        raise RuntimeError('Some annotated recordings could not be read. Inspect '
                           'placebo_eligibility_by_record.csv; no placebo inference was performed.')
    if not records:
        raise RuntimeError('No complete placebo-comparable recordings. Inspect eligibility CSV. '
                           'The 180-s exclusion may leave no feasible fixed-count placebos; inspect screening and do not relax it silently.')
    checks = verify_observed_against_saved_v2(records)
    if not checks.empty:
        checks.to_csv(OUTPUT_DIR/'placebo_v2_numerical_verification.csv',index=False)
    print('Placebo-comparable cohort:',len(records),'recordings,',
          len({r['subject_id'] for r in records}),'subjects,',
          sum(len(r['real_times']) for r in records),'accepted C markers')

    # Both arms use exactly the same included real events and subject list.
    real_rec = [record_effects_median(r['real_effects']) for r in records]
    subject_ids, obs_sub = aggregate_subjects(records, real_rec)
    observed = cohort_median(obs_sub)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        observed_boot = np.full((N_SUBJECT_BOOT, len(METHODS),len(METRICS),len(EFFECTS)),np.nan)
        rng_boot = np.random.default_rng(SEED+100001)
        for b in range(N_SUBJECT_BOOT):
            resamp = rng_boot.integers(0, len(subject_ids), len(subject_ids))
            observed_boot[b] = cohort_median(obs_sub[resamp])
    obs_ci_lo = np.nanpercentile(observed_boot,2.5,axis=0)
    obs_ci_hi = np.nanpercentile(observed_boot,97.5,axis=0)

    obs_rows = []
    for si,sid in enumerate(subject_ids):
        for mi,method in enumerate(METHODS):
            for qi,metric in enumerate(METRICS):
                for ei,effect in enumerate(EFFECTS):
                    obs_rows.append(dict(subject_id=sid,method=method,metric=metric,effect=effect,
                                         observed_subject_effect=obs_sub[si,mi,qi,ei]))
    pd.DataFrame(obs_rows).to_csv(OUTPUT_DIR/'observed_subject_effects_placebo_subset.csv',index=False)
    pd.DataFrame([dict(record=r['record'],subject_id=r['subject_id'],
                       time_sec=t, activity_at_marker=a)
                  for r in records for t,a in zip(r['real_times'],r['real_activity'])]).to_csv(
                       OUTPUT_DIR/'observed_C_markers_placebo_subset.csv',index=False)

    rng = np.random.default_rng(SEED)
    modes = ('uniform','activity_tilted') if RUN_ACTIVITY_TILTED_NULL else ('uniform',)
    nulls = {mode: np.full((N_PLACEBO, len(METHODS),len(METRICS),len(EFFECTS)),np.nan)
             for mode in modes}
    activity_balances = []
    example_rows = []
    for mode in modes:
        for b in range(N_PLACEBO):
            medians = []
            if b%200==0 or b==N_PLACEBO-1:
                print(f'{mode}: {b+1}/{N_PLACEBO} permutations',flush=True)
            for rec in records:
                draw = (rec['uniform'] if mode=='uniform' else rec['tilted']).sample(rng)
                medians.append(record_effects_median(rec['cand_effects'][draw]))
                # Quintile match assessed on original event/candidate same recording.
                pbins = rec['cbin'][draw]
                realbins = rec['rbin']
                h_placebo = np.bincount(pbins,minlength=5)/max(1,len(pbins))
                h_real = np.bincount(realbins,minlength=5)/max(1,len(realbins))
                balance_err = float(np.sum(np.abs(h_placebo-h_real))/2)
                activity_balances.append(dict(null_mode=mode,replicate=b,
                        record=rec['record'],subject_id=rec['subject_id'],
                        n_events=len(draw), activity_quintile_total_variation=balance_err))
                if b==0 and SAVE_EXAMPLE_REPLICATE:
                    for ii in draw:
                        example_rows.append(dict(null_mode=mode,record=rec['record'],
                            subject_id=rec['subject_id'],time_sec=rec['candidates'][ii],
                            activity_at_marker=rec['candidate_activity'][ii],
                            activity_quintile=int(rec['cbin'][ii])))
            _, sub = aggregate_subjects(records, medians)
            # Each metric uses exactly the subjects whose observed effect
            # exists; never compare medians over different subject cohorts.
            sub = np.where(np.isfinite(obs_sub), sub, np.nan)
            nulls[mode][b] = cohort_median(sub)

    rows, dist_rows = [], []
    for mode in modes:
        z = nulls[mode]
        for mi,method in enumerate(METHODS):
            for qi,metric in enumerate(METRICS):
                for ei,effect in enumerate(EFFECTS):
                    v = z[:,mi,qi,ei]
                    o = float(observed[mi,qi,ei])
                    valid = v[np.isfinite(v)]
                    nsub = int(np.isfinite(obs_sub[:,mi,qi,ei]).sum())
                    if len(valid) != N_PLACEBO or not np.isfinite(o):
                        warnings.warn(f'{mode}/{method}/{metric}/{effect}: missing replicate values')
                        p_dir = p_two = np.nan
                        med,lo,hi = np.nan,np.nan,np.nan
                    else:
                        med=float(np.median(valid))
                        lo,hi=np.quantile(valid,[.025,.975]).tolist()
                        # H1 rise > chance; H1 recovery < chance (both prespecified).
                        tail = np.sum(valid>=o) if effect=='marker_minus_pre' else np.sum(valid<=o)
                        p_dir = (1+int(tail))/(1+len(valid))
                        p_two = (1+int(np.sum(np.abs(valid-med) >= abs(o-med))))/(1+len(valid))
                    rows.append(dict(null_mode=mode,method=method,metric=metric,effect=effect,
                                     observed=o,n_subjects=nsub,n_recordings=len(records),
                                     n_real_markers=sum(len(r['real_times']) for r in records),
                                     observed_subject_bootstrap_CI95_low=obs_ci_lo[mi,qi,ei],
                                     observed_subject_bootstrap_CI95_high=obs_ci_hi[mi,qi,ei],
                                     placebo_median=med,placebo_q025=lo,placebo_q975=hi,
                                     p_one_sided_directional=p_dir,p_two_sided_centered=p_two,
                                     n_placebo=len(valid)))
                    for b in range(N_PLACEBO):
                        dist_rows.append(dict(null_mode=mode,replicate=b,method=method,
                                              metric=metric,effect=effect,
                                              cohort_median=float(v[b])))
    summary = pd.DataFrame(rows)
    reps = pd.DataFrame(dist_rows)
    summary.to_csv(OUTPUT_DIR/'placebo_test_summary.csv',index=False)
    reps.to_csv(OUTPUT_DIR/'placebo_replicate_statistics.csv',index=False)
    bal = pd.DataFrame(activity_balances)
    bal.to_csv(OUTPUT_DIR/'placebo_activity_balance.csv',index=False)
    if example_rows:
        pd.DataFrame(example_rows).to_csv(OUTPUT_DIR/'placebo_example_selected_times.csv',index=False)
    for mode in modes:
        plot_test(summary,reps,mode,OUTPUT_DIR/f'Fig_C_placebo_{mode}')

    report = summary.loc[(summary.metric=='R_sigma')&(summary.effect=='marker_minus_pre'),
                         ['null_mode','method','observed','n_subjects','placebo_median',
                          'placebo_q025','placebo_q975','p_one_sided_directional']]
    print('\nPRIMARY READOUT: R_sigma, marker minus pre (primary method M3)')
    print(report.to_string(index=False))
    print('Activity matching diagnostic (total variation, 0=perfect):')
    print(bal.groupby('null_mode').activity_quintile_total_variation.agg(['median','mean','max']).to_string())
    if 'activity_tilted' in modes and float(bal.loc[bal.null_mode=='activity_tilted',
                  'activity_quintile_total_variation'].mean())>ACTIVITY_BALANCE_WARNING:
        print('WARNING: activity-tilted null is NOT well matched by quintile; '
              'treat this as a weighted sensitivity, not a matched placebo.')

    info = dict(seed=SEED, n_placebo=N_PLACEBO, n_subject_boot=N_SUBJECT_BOOT,
                phases=PHASES, time_reference='rolling window END',
                min_points_each_phase=MIN_PHASE_POINTS,
                primary_test='strict C, R_sigma marker_minus_pre, M3, one-sided high',
                exclusions=dict(real_C_center_sec=C_EXCLUSION_SEC,
                                possible_C_sec=POSSIBLE_C_EXCLUSION_SEC,
                                artifact_sec=ARTIFACT_EXCLUSION_SEC,
                                mutual_placebo_sec=PLACEBO_SEPARATION_SEC),
                activity_control='quintile-weighted time sampling; approximate, balance reported',
                subject_aggregation='event median -> recording median -> subject median -> cohort median',
                n_strict_markers_total=int(len(marker_df)),
                n_records_included=len(records), n_subjects_included=len(subject_ids),
                n_included_events=sum(len(r['real_times']) for r in records),
                analysis_scope='Only cached EHG 16-channel windows; no NNDR recomputation',
                warnings=['EHG C marker timestamps have uncertainty',
                          'The observed comparison is conditional on joint M2/M3 admissibility',
                          '180-second true-marker exclusion: phase-window boundaries can touch a true C; annotations are imprecise',
                          'No model-adequacy gate is claimed'])
    (OUTPUT_DIR/'placebo_run_metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    report_original_120_coverage(records)
    fixed_cohort_120_comparator(records, marker_df)
    print('\nOutputs in:',OUTPUT_DIR)

# ======== FIXED-COHORT 120-s COMPARATOR FOR THE 180-s SENSITIVITY ===========
def fixed_cohort_120_comparator(records_180, marker_df):
    """Re-run 120-s pseudo-timing control on EXACT 180-s observed event cohort.

    This is a second placebo resampling, not a new NNDR operator fit. Unlike a
    comparison to the original v1 120-s p-value, it holds real events, records,
    subjects and admissibility fixed while changing ONLY the exclusion radius.
    """
    if not RUN_MATCHED_120_COMPARATOR:
        print('Matched 120-s comparator disabled.')
        return

    if abs(C_EXCLUSION_SEC-180.) > 1e-9:
        raise ValueError('The fixed-cohort comparator is only implemented for 180-s sensitivity.')
    print('\nReconstructing 120-s null on exactly the 180-s eligible cohort...')

    original_exc = C_EXCLUSION_SEC
    try:
        globals()['C_EXCLUSION_SEC'] = 120.0
        records_120 = []
        for k, r180 in enumerate(records_180, start=1):
            stem = r180['record']
            times = marker_df.loc[marker_df.record.eq(stem), 'time_sec'].to_numpy(float)
            r120, diag = prepare_one(stem, times)
            if r120 is None or diag.get('status') != 'included':
                raise RuntimeError(f'{stem}: accepted under 180 s but failed under 120 s')
            if not np.array_equal(r120['real_times'], r180['real_times']):
                raise RuntimeError(f'{stem}: real event set changed between exclusion radii')
            if not np.allclose(r120['real_effects'], r180['real_effects'], equal_nan=True, rtol=0, atol=1e-10):
                raise RuntimeError(f'{stem}: real event effects changed across exclusion radii')
            records_120.append(r120)
            if k % 10 == 0:
                print(f'  rebuilt {k}/{len(records_180)} records at 120 s', flush=True)
    finally:
        globals()['C_EXCLUSION_SEC'] = original_exc

    observed_rec = [record_effects_median(r['real_effects']) for r in records_180]
    ids180, observed_sub = aggregate_subjects(records_180, observed_rec)
    ids120, observed_sub120 = aggregate_subjects(records_120, [record_effects_median(r['real_effects']) for r in records_120])
    if ids180 != ids120 or not np.allclose(observed_sub, observed_sub120, equal_nan=True, rtol=0, atol=1e-10):
        raise RuntimeError('Fixed-cohort comparator failed subject alignment verification')
    obs = cohort_median(observed_sub)
    rng = np.random.default_rng(SEED + 120000)
    modes = ('uniform', 'activity_tilted') if RUN_ACTIVITY_TILTED_NULL else ('uniform',)
    records_by_mode = records_120
    nulls = {m: np.full((N_PLACEBO, len(METHODS), len(METRICS), len(EFFECTS)), np.nan) for m in modes}
    balance_rows = []
    replicate_rows = []
    for mode in modes:
        for b in range(N_PLACEBO):
            medians = []
            if b % 200 == 0 or b == N_PLACEBO-1:
                print(f'  matched-120 {mode}: {b+1}/{N_PLACEBO}', flush=True)
            for rec in records_by_mode:
                draw = (rec['uniform'] if mode == 'uniform' else rec['tilted']).sample(rng)
                medians.append(record_effects_median(rec['cand_effects'][draw]))
                p_hist = np.bincount(rec['cbin'][draw], minlength=5) / max(1, len(draw))
                r_hist = np.bincount(rec['rbin'], minlength=5) / max(1, len(rec['rbin']))
                balance_rows.append(dict(null_mode=mode, replicate=b, record=rec['record'],
                         subject_id=rec['subject_id'], activity_quintile_total_variation=
                         float(.5*np.sum(np.abs(p_hist-r_hist)))))
            ids, subj = aggregate_subjects(records_by_mode, medians)
            if ids != ids180:
                raise RuntimeError('Subject population mismatch during matched placebo')
            subj = np.where(np.isfinite(observed_sub), subj, np.nan)
            nulls[mode][b] = cohort_median(subj)

    summary_rows = []
    for mode in modes:
        for mi, method in enumerate(METHODS):
            for qi, metric in enumerate(METRICS):
                for ei, effect in enumerate(EFFECTS):
                    z = nulls[mode][:, mi, qi, ei]
                    real = float(obs[mi, qi, ei])
                    valid = z[np.isfinite(z)]
                    if len(valid) != N_PLACEBO or not np.isfinite(real):
                        raise RuntimeError(f'Nonfinite matched-null series: {mode} {method} {metric} {effect}')
                    median = float(np.median(valid))
                    left, right = np.quantile(valid, [.025,.975]).tolist()
                    tail = np.sum(valid >= real) if effect == 'marker_minus_pre' else np.sum(valid <= real)
                    pdir = (1 + int(tail)) / (N_PLACEBO + 1)
                    ptwo = (1 + int(np.sum(np.abs(valid-median) >= abs(real-median))))/(N_PLACEBO+1)
                    summary_rows.append(dict(null_mode=mode, method=method, metric=metric,
                        effect=effect, observed=real,
                        n_subjects=int(np.isfinite(observed_sub[:,mi,qi,ei]).sum()),
                        n_recordings=len(records_180),
                        n_real_markers=sum(len(rec['real_times']) for rec in records_180),
                        placebo_median=median, placebo_q025=left, placebo_q975=right,
                        p_one_sided_directional=pdir,p_two_sided_centered=ptwo,n_placebo=len(valid),
                        placebo_C_exclusion_sec=120, cohort_fixed_to_180=True))
                    replicate_rows.extend(dict(null_mode=mode,replicate=int(i),method=method,metric=metric,
                                effect=effect,cohort_median=float(v)) for i,v in enumerate(z))
    compare = pd.DataFrame(summary_rows)
    compare.to_csv(OUTPUT_DIR/'placebo_matched120_test_summary.csv',index=False)
    pd.DataFrame(replicate_rows).to_csv(OUTPUT_DIR/'placebo_matched120_replicate_statistics.csv',index=False)
    pd.DataFrame(balance_rows).to_csv(OUTPUT_DIR/'placebo_matched120_activity_balance.csv',index=False)
    results180 = pd.read_csv(OUTPUT_DIR/'placebo_test_summary.csv')
    join = results180.merge(compare,on=['null_mode','method','metric','effect'],
                  suffixes=('_180','_120_matched'),validate='one_to_one')
    for col in ('observed', 'n_recordings', 'n_real_markers', 'n_subjects'):
        x = join[col+'_180'].to_numpy(float)
        y = join[col+'_120_matched'].to_numpy(float)
        if not np.allclose(x,y,equal_nan=True,rtol=0,atol=1e-10):
            raise RuntimeError(f'Fixed-cohort test: {col} differs between 120 and 180')
    join.to_csv(OUTPUT_DIR/'placebo_120vs180_same_cohort_comparison.csv',index=False)
    print('\nPRIMARY: fixed-cohort 120 vs 180 exclusion (same subject list / real C events):')
    view = join.loc[(join.method=='M3')&(join.metric=='R_sigma')&
                    (join.effect=='marker_minus_pre'),
              ['null_mode','observed_180','n_subjects_180',
               'p_one_sided_directional_120_matched','p_one_sided_directional_180',
               'placebo_median_120_matched','placebo_median_180']]
    print(view.to_string(index=False))
    info_path=OUTPUT_DIR/'placebo_run_metadata.json'
    metadata=json.loads(info_path.read_text(encoding='utf-8'))
    metadata['fixed_cohort_120_comparator'] = dict(enabled=True,
            seed=SEED+120000,
            description='120-s exclusion rerun on identical 180-s recording, subject, and observed-event set',
            n_subjects=len(ids180), n_recordings=len(records_180),
            n_real_events=sum(len(rec['real_times']) for rec in records_180))
    info_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def report_original_120_coverage(records_180):
    """Compare cohort coverage only; old 120 p values not a fixed-cohort comparison."""
    new = pd.read_csv(OUTPUT_DIR/'placebo_eligibility_by_record.csv')
    oldp = DATASET_OUTPUT/'C_timing_placebo_v1'/'placebo_eligibility_by_record.csv'
    if not oldp.exists():
        print('Original 120-s eligibility CSV missing: coverage comparison omitted.')
        return
    old = pd.read_csv(oldp)
    if not {'record','status'}.issubset(old):
        print('Original eligibility file has unexpected columns: coverage comparison omitted.')
        return
    merged = new.merge(old[['record','status']].rename(columns={'status':'status_120_original'}),
                       on='record',how='outer',validate='one_to_one')
    merged.rename(columns={'status':'status_180'},inplace=True)
    merged['retained_at_120'] = merged.status_120_original.eq('included')
    merged['retained_at_180'] = merged.status_180.eq('included')
    merged.to_csv(OUTPUT_DIR/'eligibility_120vs180_by_record.csv',index=False)
    print(f'Original 120-s records: {merged.retained_at_120.sum()}; '
          f'180-s records: {merged.retained_at_180.sum()}; '
          f'retained in both: {(merged.retained_at_120 & merged.retained_at_180).sum()}')


if __name__ == '__main__':
    main()
