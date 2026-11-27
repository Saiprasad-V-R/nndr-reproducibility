"""Orthogonally constrained input-response plane extractors.

For NNDR: max_{||u||=||v||=1,u.T v=0} u.T F v.
The first basis column returned is response u, second is input v.
No spectral stabilization or whitening is applied.
"""
from __future__ import annotations
import numpy as np


def m2_multistart(F, *, seed=0, restarts=12, iterations=350, tol=1e-11):
    """Multiple projected alternating updates; returns diagnostics per restart.

    This solves the non-convex M2 optimization approximately, not a certified
    global optimum.  The best attained objective is retained.
    """
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or not np.all(np.isfinite(F)):
        raise ValueError('F must be finite real square')
    N = F.shape[0]
    if N < 2 or restarts < 1:
        raise ValueError('Require N>=2, restarts>=1')
    rng = np.random.default_rng(seed)
    best_val, best_u, best_v = -np.inf, None, None
    all_values = []
    for j in range(restarts):
        v = rng.standard_normal(N); v /= np.linalg.norm(v)
        prev = -np.inf
        for k in range(iterations):
            u = F @ v
            u -= float(v @ u) * v
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-13:
                break
            u /= norm_u
            vnew = F.T @ u
            vnew -= float(u @ vnew) * u
            norm_v = np.linalg.norm(vnew)
            if norm_v < 1e-13:
                break
            v = vnew / norm_v
            value = float(u @ F @ v)
            if abs(value-prev) <= tol*max(1.0, abs(value)):
                break
            prev = value
        # Recompute u at converged v (satisfy exact u-perp-v constraint).
        u = F @ v; u -= float(v @ u)*v
        nu = float(np.linalg.norm(u))
        if nu < 1e-13:
            value = 0.0
        else:
            u /= nu; value = float(u @ F @ v)
        all_values.append(value)
        if nu >= 1e-13 and value > best_val:
            best_val, best_u, best_v = value, u.copy(), v.copy()
    if best_u is None:
        raise RuntimeError('No nondegenerate M2 plane found')
    Q = np.column_stack([best_u, best_v])
    if not np.allclose(Q.T@Q, np.eye(2), rtol=0, atol=1e-9):
        raise RuntimeError('M2 output not orthonormal')
    return {'Q':Q, 'r':best_u, 'n':best_v,
            'objective':best_val, 'restart_objectives':np.asarray(all_values),
            'min_restart_objective':float(np.min(all_values)),
            'max_restart_objective':float(np.max(all_values))}


def principal_angle_deg(Q1,Q2):
    s = np.linalg.svd(np.asarray(Q1).T@np.asarray(Q2),compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(float(np.min(s)),0,1))))
