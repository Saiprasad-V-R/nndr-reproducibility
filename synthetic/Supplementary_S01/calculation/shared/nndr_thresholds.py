"""Numerically stable NNDR reduced threshold diagnostics (rebuttal Option 1).

Continuous-time variance-response criterion R_A and discrete one-step R_sigma
are different scientific observables.  All norms use the Euclidean metric.

References: CNSNS-D-26-03856 response, Appendix B eq. (B.4), Appendix C
(C.1)-(C.12); Sornette October 7 2026 internal technical note.

This module does not impose stability through rescaling and does not silently
replace complex-eigenvalue thresholds with real formulas.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.linalg import logm, schur

EPS = np.finfo(float).eps


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def henrici_2x2(matrix: np.ndarray) -> float:
    """Unitary-invariant 2D Henrici departure, computed via *real Schur*.

    Avoids cancellation in ||A||_F^2-sum(|lambda|^2) near normal matrices.
    For a real triangular Schur form, d_F=|offdiag|. For a real Schur
    conjugate block, d_F=||upper_offdiag|-|lower_offdiag||.
    """
    a = np.asarray(matrix, dtype=float)
    if a.shape != (2, 2) or not np.all(np.isfinite(a)):
        raise ValueError("Requires a finite real 2x2 matrix")
    t, _ = schur(a, output="real")
    scale = max(1.0, np.linalg.norm(a, ord="fro"))
    if abs(t[1, 0]) <= 1e-14 * scale:
        return abs(float(t[0, 1]))
    # Complex Schur block; off-diagonal product is negative.
    if t[0, 1] * t[1, 0] < 0:
        return abs(float(abs(t[0, 1]) - abs(t[1, 0])))
    # Defensive fallback: compute invariant directly if a quasi-block
    # is numerically close to two resolved real eigenvalues.
    evals = np.linalg.eigvals(a)
    return float(np.sqrt(max(0, np.linalg.norm(a, "fro") ** 2 - np.sum(abs(evals) ** 2))))


def kc_variance_real(gamma_1: float, gamma_2: float) -> float:
    """Eq. B.4; gamma1, gamma2 >0, returns +inf for repeated rates."""
    g1, g2 = sorted([float(gamma_1), float(gamma_2)])
    if g1 <= 0:
        return float("nan")
    denom = abs(np.sqrt(g2) - np.sqrt(g1))
    return float(np.sqrt(2 * np.sqrt(g1 * g2)) / denom) if denom > 0 else float("inf")


def kc_variance_complex(gamma: float, omega: float) -> float:
    """Eq. C.4; stable rate pair -gamma +/- i omega."""
    if gamma <= 0:
        return float("nan")
    beta = abs(float(omega)) / float(gamma)
    if beta == 0:
        return float("inf")
    # rationalization of sqrt(1+beta^2)-1 avoids cancellation
    return float(np.sqrt((np.hypot(1, beta) + 1)) / beta)


def r_variance_from_generator(generator: np.ndarray) -> dict[str, Any]:
    """R_A via Schur d_F and *rates*, with stable degeneracy limit (C.12).

    Formulae: real rates -g1,-g2:
        R = d_F / [(sqrt(g1)+sqrt(g2))*sqrt(2*sqrt(g1*g2))].
    complex rates -gamma +/- i*omega:
        R = d_F / [2*sqrt(gamma*(hypot(gamma,omega)+gamma))].
    The denominators are nonsingular for stable, repeated eigenvalues.
    """
    L = np.asarray(generator, dtype=float)
    if L.shape != (2, 2) or not np.all(np.isfinite(L)):
        raise ValueError("Generator must be finite real 2x2")
    ev = np.linalg.eigvals(L)
    dF = henrici_2x2(L)
    maxreal = float(np.max(np.real(ev)))
    result: dict[str, Any] = {"status": "ok" if maxreal < 0 else "unstable_generator",
                                  "dF_generator": dF, "max_real_rate": maxreal,
                                  "spectrum_type": "complex" if np.max(abs(ev.imag)) > 1e-12 else "real",
                                  "R_variance": float("nan"), "K": float("nan"),
                                  "Kc_variance": float("nan"), "R_reactivity": float("nan")}
    if maxreal >= 0:
        return result
    if result["spectrum_type"] == "real":
        g1, g2 = sorted([-float(np.real(ev[0])), -float(np.real(ev[1]))])
        splitting = abs(g2 - g1)
        denom = (np.sqrt(g1) + np.sqrt(g2)) * np.sqrt(2 * np.sqrt(g1 * g2))
        result["R_variance"] = dF / denom
        result["Kc_variance"] = kc_variance_real(g1, g2)
        result["R_reactivity"] = dF / (2 * np.sqrt(g1 * g2))
        if splitting > 1e-10 * max(g1, g2):
            result["K"] = dF / splitting
        elif dF == 0:
            result["K"] = 0.0
        result["gamma_fast"] = g2
        result["gamma_slow"] = g1
    else:
        gamma = -float(np.mean(ev.real))
        omega = float(np.max(abs(ev.imag)))
        denom = 2 * np.sqrt(gamma * (np.hypot(gamma, omega) + gamma))
        result["R_variance"] = dF / denom
        result["K"] = dF / (2 * omega)
        result["Kc_variance"] = kc_variance_complex(gamma, omega)
        result["R_reactivity"] = dF / (2 * gamma)
        result["gamma"] = gamma
        result["omega"] = omega
    result["numerical_abscissa"] = float(np.linalg.eigvalsh((L + L.T) / 2)[-1])
    return result


def reduced_diagnostics(gamma_matrix: np.ndarray, delta_t: float = 1.0) -> dict[str, Any]:
    """Compute both rates-based R_A and exact one-step R_sigma.

    The principal matrix log requires no negative-real multipliers.  The
    mathematical R_sigma remains computable for any spectrally stable real map,
    but R_A is flagged outside the real-log principal-branch domain.
    """
    G = np.asarray(gamma_matrix, dtype=float)
    if G.shape != (2, 2) or not np.all(np.isfinite(G)) or delta_t <= 0:
        raise ValueError("Provide finite 2x2 real Gamma and delta_t>0")
    eig = np.linalg.eigvals(G)
    rho = float(np.max(abs(eig)))
    svals = np.linalg.svd(G, compute_uv=False)
    dF = henrici_2x2(G)
    out: dict[str, Any] = {"status": "ok", "rho": rho, "sigma_max": float(svals[0]),
                           "sigma_min": float(svals[-1]), "dF": dF,
                           "R_sigma": float("nan"), "R_variance": float("nan"),
                           "R_reactivity": float("nan"), "K": float("nan"),
                           "Kc_variance": float("nan"), "full_rotation_warning": False,
                           "eig_real": [float(z.real) for z in eig],
                           "eig_imag": [float(z.imag) for z in eig]}
    if rho >= 1:
        out["status"] = "unstable_reduced_map"
        return out
    # Stable 2x2 maps have a positive one-step margin.
    margin = float(np.sqrt(np.prod([max(0.0, -np.expm1(2 * np.log(abs(z)))) if abs(z) > 0 else 1.0 for z in eig])))
    out["R_sigma"] = dF / margin if margin > 0 else float("nan")
    if np.any(abs(eig) <= 1e-15):
        out["status"] = "singular_map_no_finite_log"
        return out
    if np.any((abs(eig.imag) <= 1e-12) & (eig.real < 0)):
        out["status"] = "negative_real_multiplier_principal_log_not_real"
        return out
    Llog = logm(G)
    imag_norm = float(np.max(np.abs(Llog.imag)))
    if imag_norm > 1e-8 * max(1.0, float(np.linalg.norm(Llog.real))):
        out["status"] = "complex_principal_log"
        return out
    L = np.asarray(Llog.real, float) / float(delta_t)
    rates = r_variance_from_generator(L)
    for k in ("R_variance", "R_reactivity", "K", "Kc_variance", "numerical_abscissa", "spectrum_type", "dF_generator"):
        out[k] = rates.get(k, np.nan)
    out["rate_matrix"] = L
    # Phase near Nyquist cannot be unambiguously unwrapped from samples.
    out["full_rotation_warning"] = bool(np.max(abs(np.angle(eig))) > 0.85 * np.pi)
    return out


def variance_kernel(generator: np.ndarray, times: np.ndarray) -> np.ndarray:
    """True white-forcing impulse kernel ||exp(tL)||_F^2, NOT stationary variance."""
    from scipy.linalg import expm
    t = np.asarray(times, float)
    return np.asarray([np.linalg.norm(expm(tt * generator), ord="fro") ** 2 for tt in t])
