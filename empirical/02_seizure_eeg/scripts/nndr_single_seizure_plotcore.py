#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrected NNDR diagnostics in ONE seizure recording, manuscript-style.

RUN IN SPYDER (F5), NO ARGUMENTS.  The user-specified scientific task is to
illustrate the marked increase in a selected seizure, not to average patients
or compare with placebo onsets.

Panels, all from CHB-MIT chb01_03.edf:
  (a) EEG mean field, with annotated seizure shaded;
  (b) corrected continuous-time reduced ratio R_A, M2 and M3;
  (c) corrected discrete one-step reduced ratio R_sigma, M2 and M3;
  (d) condition number of the unit-normalized right eigenvector basis of
      the full fitted VAR(1) operator F (a related descriptive diagnostic).
      A separate supplement compares the full and reduced condition numbers.

This is not the same condition number or fitting procedure as the historical
Pseudo-Bifurcations paper; the full fitted transition operator is F, and the
reduced operator is Q.T @ F @ Q.  Eigenvector condition numbers require a
normalization convention and are sensitive to near-repeated eigenvalues.

We use the exact existing Stage-6 fitted-operator and corrected-ratio functions
from nndr_seizure_stage6_cohort.py; the EDF must be available locally. Output
is never manufactured when EDF is missing. Repeated execution resumes from
checkpointed window rows.

DATA: CHB-MIT chb01_03 EDF; 256 Hz -> (notch and 0.5-15 Hz filtering) ->
32 Hz; fixed core16 bipolar montage; full 40s trailing windows, 2s hop;
regularization relative ridge .003, shrink .05; stable full VAR(1) required.
"""
from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svdvals

import nndr_seizure_stage6_cohort as s6

# User settings (no command-line arguments)
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
EDF_NAME = 'chb01_03.edf'
WINDOW_SEC = 40
HOP_SEC = 2
END_RELATIVE_RANGE = (-160, 240)  # end of trailing window relative to seizure start
MAIN_FOCUS_RANGE = (-120, 70)      # main manuscript plot; onset, seizure, early recovery
METHODS = ('M2', 'M3')
SMOOTH_POINT_COUNT = 5            # 5 endpoints = approximately 10-second median smoothing
FOURTH_PANEL = 'full'             # 'full' follows the older paper's full eigenbasis idea; 'reduced' also possible
SHOW_RAW = True
VERIFY_SAVED_RA = True
MAX_PARITY_DIFFERENCE = 5e-6
SAVE_AFTER_EVERY_N_WINDOWS = 8
OUTPUT = HERE.parent / 'results'

assert s6.MONTAGE_MODE == 'core16', 'Set helper to the canonical core16 montage.'
assert int(round(s6.FS)) == 32
assert np.isclose(s6.V4_RIDGE_REL, .003)
assert np.isclose(s6.V4_SHRINK_ALPHA, .05)

COLORS = {'M2': '#d46b24', 'M3': '#008f70'}


def find_edf():
    if DATA_ROOT is not None:
        candidates = [Path(DATA_ROOT).expanduser()]
    else:
        candidates = [HERE, HERE.parent, HERE / 'chbmit_seizure_data',
                      HERE.parent / 'chbmit_seizure_data', Path.cwd(), Path.cwd().parent]
    paths = []
    for root in candidates:
        paths.extend([root/'chb01'/EDF_NAME, root/EDF_NAME])
    for p in paths:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        'Could not find chb01_03.edf. Put this script + the companion helper '
        'inside your chbmit_seizure_data folder alongside chb01/, or set '
        'DATA_ROOT at the top of the script.\nChecked:\n' +
        '\n'.join(str(x) for x in paths))


def condition_number_eigenbasis(M, relative_gap=1e-8):
    """cond(V) with separately unit-normalized right eigenvectors.

    Multiplying eigenvectors by arbitrary scalars changes cond(V), so a
    unit-norm-per-column convention is imposed. Eigenvalue degeneracies and
    near-collisions are marked undefined, not silently regularized.
    """
    eig, V = np.linalg.eig(np.asarray(M))
    scales = np.linalg.norm(V, axis=0)
    if not np.all(np.isfinite(eig)) or np.any(scales < 1e-12):
        return np.nan
    if len(eig) > 1:
        gap = np.min(abs(eig[:, None] - eig[None, :] + np.eye(len(eig)) * 1e100))
        if gap < relative_gap * max(1., max(abs(eig))):
            return np.nan
    V /= scales
    kappa = float(np.linalg.cond(V))
    return kappa if np.isfinite(kappa) else np.nan


def metrics_for_one_window(X):
    F = s6.fit_v4_full_window(X)
    rho = float(np.max(abs(np.linalg.eigvals(F))))
    full_eig_basis_kappa = condition_number_eigenbasis(F)
    ret = {'rho_F': rho, 'kappa_full_eigvec': full_eig_basis_kappa,
           'full_stable': bool(np.isfinite(rho) and rho < 1. - s6.TOL)}
    if not ret['full_stable']:
        for method in METHODS:
            ret.update({f'{method}_{x}': np.nan for x in
                        ('R_A', 'R_sigma', 'kappa_reduced_eigvec', 'sigma_G',
                         'rho_G', 'K', 'Kc_A', 'complex_pair')})
            ret[f'{method}_status'] = 'full_operator_unstable'
        return ret
    for method in METHODS:
        Q, _ = s6.method2(F) if method == 'M2' else s6.method3(F)
        if Q is None:
            for x in ('R_A', 'R_sigma', 'kappa_reduced_eigvec', 'sigma_G',
                      'rho_G', 'K', 'Kc_A', 'complex_pair'):
                ret[f'{method}_{x}'] = np.nan
            ret[f'{method}_status'] = 'plane_not_found'
            continue
        G = Q.T @ F @ Q
        data = s6.ratio_metrics(G, 1 / s6.FS)
        if data['sigma_G'] > float(svdvals(F)[0]) + 1e-8:
            raise AssertionError('The reduced singular value exceeds the full singular value.')
        for name in ('R_A', 'R_sigma', 'sigma_G', 'rho_G', 'K', 'Kc_A'):
            ret[f'{method}_{name}'] = float(data[name])
        ret[f'{method}_complex_pair'] = bool(data['complex_pair'])
        ret[f'{method}_kappa_reduced_eigvec'] = condition_number_eigenbasis(G)
        ret[f'{method}_status'] = str(data['reason'] or 'ok')
    return ret


def verified_parity(rows):
    """Test M2 R_A against the already-verified EDF-computed single-event values.

    Optional sanity check only. Never replace new data with saved values.
    """
    p = HERE / 'chb01_03_verified_RA_40s.csv'
    if not p.exists():
        return {'reference_present': False}
    saved = pd.read_csv(p)
    saved = saved[(saved.event_type == 'real') & np.isfinite(saved.ratio)]
    d = rows.merge(saved[['tau_end_s', 'ratio']], on='tau_end_s', how='inner')
    d = d[np.isfinite(d['M2_R_A'])]
    if d.empty:
        return {'reference_present': True, 'compared_windows': 0}
    err = abs(d['M2_R_A'] - d['ratio'])
    maxerr = float(err.max())
    summary = {'reference_present': True, 'compared_windows': int(len(d)),
               'max_abs_difference': maxerr, 'tolerance': MAX_PARITY_DIFFERENCE}
    if VERIFY_SAVED_RA and maxerr > MAX_PARITY_DIFFERENCE:
        raise AssertionError('The new M2 R_A results disagree with the saved '
                             f'EDF results (max error {maxerr:g}). Review the '
                             'EDF, preprocessing, montage, and estimator before plotting.')
    return summary


def compute(edf):
    print('Reading representative EDF:', edf, flush=True)
    events, annotation_source = s6.get_annotations(edf)
    if len(events) != 1:
        raise ValueError(f'Expected one annotated seizure in {EDF_NAME}; found {events}')
    onset, offset = map(float, events[0])
    if not offset > onset:
        raise ValueError(f'Invalid seizure annotation: {events[0]}')
    print(f'Seizure onset {onset:g} s, offset {offset:g} s, annotation: {annotation_source}', flush=True)

    X, manifest = s6.read_eeg(edf)
    if X.shape[1] != 16:
        raise AssertionError('Not the required fixed 16-channel montage')
    nwin = int(round(WINDOW_SEC * s6.FS))
    tgrid = np.arange(END_RELATIVE_RANGE[0], END_RELATIVE_RANGE[1] + 1,
                      HOP_SEC, dtype=int)
    checkpoint = OUTPUT / 'representative_40s_corrected_diagnostics_checkpoint.csv'
    if checkpoint.exists():
        old = pd.read_csv(checkpoint)
        if 'tau_end_s' not in old.columns or old.duplicated('tau_end_s').any():
            raise ValueError('Invalid checkpoint; rename/delete before retrying.')
        cache = {int(r['tau_end_s']): r.to_dict() for _, r in old.iterrows()}
        print('Reusing', len(cache), 'previously completed windows.', flush=True)
    else:
        cache = {}
    for j, tau in enumerate(tgrid, 1):
        if int(tau) in cache:
            continue
        end_sample = int(round((onset + float(tau)) * s6.FS))
        beg_sample = end_sample - nwin
        if beg_sample < 0 or end_sample > len(X):
            raise ValueError(f'Window exceeds EDF bounds: end relative to onset {tau:g} s')
        r = metrics_for_one_window(X[beg_sample:end_sample, :])
        r['tau_end_s'] = int(tau)
        r['window_start_relative_s'] = float(tau - WINDOW_SEC)
        r['window_fraction_ictal'] = float(max(0., min(tau, offset-onset) -
                                                max(tau-WINDOW_SEC, 0.)) / WINDOW_SEC)
        cache[int(tau)] = r
        if j % SAVE_AFTER_EVERY_N_WINDOWS == 0 or j == len(tgrid):
            pd.DataFrame([cache[t] for t in sorted(cache)]).to_csv(checkpoint, index=False)
            print(f'Computed {len(cache)}/{len(tgrid)} window endpoints', flush=True)
    rows = pd.DataFrame([cache[t] for t in tgrid]).sort_values('tau_end_s')
    rows.to_csv(OUTPUT / 'individual_seizure_corrected_ratios_all_windows.csv', index=False)
    parity = verified_parity(rows)

    ts = np.arange(len(X), dtype=float) / float(s6.FS) - onset
    region = (ts >= END_RELATIVE_RANGE[0]) & (ts <= END_RELATIVE_RANGE[1])
    eeg_time = ts[region]
    eeg_uv = X[region, :].mean(axis=1) * 1e6
    wave = pd.DataFrame({'t_rel_s': eeg_time, 'mean_eeg_microvolt': eeg_uv})
    wave.to_csv(OUTPUT/'representative_EEG_meanfield.csv', index=False)
    metadata = {
        'edf': str(edf), 'annotation_source': annotation_source,
        'onset_s': onset, 'offset_s': offset,
        'data_channels': int(X.shape[1]), 'sampling_hz': float(s6.FS),
        'window_sec': WINDOW_SEC, 'hop_sec': HOP_SEC,
        'ridge_rel': float(s6.V4_RIDGE_REL),
        'shrinkage_alpha': float(s6.V4_SHRINK_ALPHA),
        'methods': list(METHODS),
        'min_rho_full_required': 'rho(F)<1',
        'R_A_definition': 'corrected continuous-time reduced threshold',
        'R_sigma_definition': 'corrected one-step reduced threshold',
        'kappa_full': 'condition number of column-normalized eigenvectors of fitted F',
        'kappa_reduced': 'condition number of column-normalized eigenvectors of Q^T F Q',
        'parity_with_previous_verified_EDF': parity,
        'counts': {method: {
            'R_A_finite': int(np.isfinite(rows[f'{method}_R_A']).sum()),
            'R_sigma_finite': int(np.isfinite(rows[f'{method}_R_sigma']).sum()),
            'kappa_reduced_finite': int(np.isfinite(rows[f'{method}_kappa_reduced_eigvec']).sum())
        } for method in METHODS},
    }
    (OUTPUT/'figure_run_metadata.json').write_text(json.dumps(metadata, indent=2))
    return rows, wave, offset-onset, metadata


def draw_trace(ax, t, y, color, label, smooth_n=SMOOTH_POINT_COUNT):
    # Keep unsmoothed line visible, as in the old manuscript figure.
    y = np.asarray(y, dtype=float)
    if SHOW_RAW:
        ax.plot(t, y, color=color, alpha=.23, lw=.58)
    # Rolling median with no interpolation across missing/invalid observations.
    ys = pd.Series(y).rolling(smooth_n, center=True,
                              min_periods=max(1, (smooth_n+1)//2)).median()
    # Invalid windows remain gaps in both raw and smoothed traces.
    ys[~np.isfinite(y)] = np.nan
    ax.plot(t, ys, color=color, lw=1.45, label=label)


def plot_panels(rows, wave, seizure_length, focus, suffix, recording_label="chb01_03"):
    # Figure width 18 cm for journal two-column figures.
    fig, axs = plt.subplots(4, 1, figsize=(18/2.54, 19.2/2.54),
                            dpi=180, sharex=True,
                            gridspec_kw={'height_ratios': [1.0,1.0,1.0,.95]})
    x = rows['tau_end_s'].to_numpy(dtype=float)
    wt = wave['t_rel_s'].to_numpy(dtype=float)
    wv = wave['mean_eeg_microvolt'].to_numpy(dtype=float)
    axs[0].plot(wt, wv, color='.10', linewidth=.48)
    axs[0].set_ylabel(r'$m_{\rm EEG}$ ($\mu$V)')
    axs[0].text(.99,.90, 'CHB-MIT: ' + recording_label, transform=axs[0].transAxes,
                ha='right', va='top', fontsize=7.5)
    for ax, col, label in [(axs[1], 'R_A', r'$R_A$'),
                           (axs[2], 'R_sigma', r'$R_\sigma$')]:
        for method in METHODS:
            draw_trace(ax, x, rows[f'{method}_{col}'], COLORS[method],
                       'Optimization (M2)' if method=='M2' else 'Commutator (M3)')
        ax.axhline(1, color='.42', ls='--', lw=.7)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.17, linewidth=.4)
    if FOURTH_PANEL == 'full':
        draw_trace(axs[3], x, rows['kappa_full_eigvec'], '#222222',
                   r'Full VAR(1) eigenvector basis')
        axs[3].set_ylabel(r'$\kappa_{\rm eig}(\widehat F)$')
    elif FOURTH_PANEL == 'reduced':
        for method in METHODS:
            draw_trace(axs[3], x, rows[f'{method}_kappa_reduced_eigvec'],
                       COLORS[method],
                       'Optimization (M2)' if method=='M2' else 'Commutator (M3)')
        axs[3].set_ylabel(r'$\kappa_{\rm eig}(\Gamma)$')
    else:
        raise ValueError('FOURTH_PANEL must be full or reduced')
    axs[3].axhline(1, color='.42', ls='--', lw=.7)
    axs[3].grid(axis='y', alpha=.17, linewidth=.4)
    axs[3].set_xlabel('Window end relative to seizure onset (s)')

    for idx, ax in enumerate(axs):
        ax.axvspan(0, seizure_length, facecolor='0.80', alpha=.29, linewidth=0)
        ax.axvline(0, color='.34', ls='--', lw=.72)
        ax.axvline(seizure_length, color='.34', ls='--', lw=.72)
        ax.set_xlim(*focus)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.text(-.07,1.02, '(' + 'abcd'[idx] + ')', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, fontweight='bold')
    # Panel (a) time limits should not be dominated by data outside plotted range.
    a = wv[(wt>=focus[0])&(wt<=focus[1])]
    if len(a)>0 and np.any(np.isfinite(a)):
        ymax = max(np.nanmax(abs(a)), 1e-4)
        axs[0].set_ylim(-1.08*ymax,1.08*ymax)
    # Scale each panel using ALL its finite observations IN the displayed
    # time range. A separate full-span figure retains every later spike.
    for i, ax in enumerate(axs[1:], 1):
        vals = [1.0]  # retain the R=1 or kappa=1 reference line
        for line in ax.lines:
            ly = np.asarray(line.get_ydata(), dtype=float).ravel()
            lx = np.asarray(line.get_xdata(), dtype=float).ravel()
            if len(ly) != len(x) or len(lx) != len(x):
                continue
            valid = (lx >= focus[0]) & (lx <= focus[1]) & np.isfinite(ly)
            vals.extend(ly[valid])
        vv = np.asarray(vals, dtype=float)
        if len(vv) > 0:
            lo, hi = float(np.min(vv)), float(np.max(vv))
            if lo > 0 and hi / lo >= 20:
                ax.set_yscale('log')
                ax.set_ylim(lo / 1.20, hi * 1.20)
            else:
                pad = .065 * max(hi-lo, .2)
                ax.set_ylim(lo-pad, hi+pad)
    hh,lab=axs[1].get_legend_handles_labels()
    fig.legend(hh,lab,loc='lower center',ncol=2,frameon=False,
               bbox_to_anchor=(.51,.014), fontsize=8)
    fig.subplots_adjust(left=.14,right=.985,top=.975,bottom=.12,hspace=.16)
    pdf = OUTPUT / f'Figure_CHB01_03_corrected_individual_{suffix}.pdf'
    png = OUTPUT / f'Figure_CHB01_03_corrected_individual_{suffix}.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=250)
    plt.close(fig)
    return pdf,png


def plot_basis_comparison(rows, seizure_length):
    fig, ax = plt.subplots(figsize=(18/2.54,6/2.54),dpi=160)
    t=rows['tau_end_s'].to_numpy(float)
    draw_trace(ax,t,rows['kappa_full_eigvec'].to_numpy(float),'#424242',
               r'Full fitted $F$ eigenbasis')
    draw_trace(ax,t,rows['M2_kappa_reduced_eigvec'].to_numpy(float),
               COLORS['M2'], r'Reduced $\Gamma$ (M2) eigenbasis')
    ax.set_yscale('log')
    ax.axvspan(0,seizure_length,facecolor='.8',alpha=.32)
    ax.axvline(0,color='.33',ls='--',lw=.7)
    ax.axvline(seizure_length,color='.33',ls='--',lw=.7)
    ax.set_ylabel(r'Normalized eigenbasis condition number')
    ax.set_xlabel('Window end relative to seizure onset (s)')
    ax.legend(frameon=False,fontsize=8)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.tight_layout()
    pdf=OUTPUT/'Supplement_eigenbasis_condition_numbers.pdf'
    fig.savefig(pdf);fig.savefig(pdf.with_suffix('.png'),dpi=220)
    plt.close(fig)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    edf = find_edf()
    rows,wave,ictal_dur,meta=compute(edf)
    plot_panels(rows,wave,ictal_dur,MAIN_FOCUS_RANGE,'focus')
    plot_panels(rows,wave,ictal_dur,END_RELATIVE_RANGE,'full_span')
    plot_basis_comparison(rows,ictal_dur)
    bundle = HERE / 'NNDR_SingleSeizure_Figure_Output_Upload.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(OUTPUT.iterdir()):
            if file.suffix.lower() in {'.pdf', '.png', '.csv', '.json'}:
                archive.write(file, arcname=file.name)
    print('\nCompleted actual EDF figures and diagnostics in', OUTPUT, flush=True)
    print('Upload this one ZIP for scientific audit:', bundle, flush=True)
    print(json.dumps(meta['counts'], indent=2), flush=True)

if __name__ == '__main__':
    run()
