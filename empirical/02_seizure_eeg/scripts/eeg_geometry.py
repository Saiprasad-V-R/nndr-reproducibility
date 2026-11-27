"""Single-window geometry from the validated corrected EEG analysis."""
import numpy as np

RA_AND_RSIGMA_REQUIRE_FINITE = True

def reaction_mode_from_Q(F, Q, method):
    """M2: normalized first Q column (u). M3: optimize u^T G v INSIDE plane.

    For any 2x2 real G, v=(cos theta,sin theta), u=(-sin theta,cos theta).
    The objective is C + A*cos(2theta) + B*sin(2theta). The maximum
    absolute value is |C|+sqrt(A^2+B^2). This is analytic, deterministic,
    and avoids a SECOND random optimization in the M3 plane.
    """
    if method == 'M2':
        vec = np.asarray(Q[:, 0], dtype=float)
    else:
        G = Q.T @ F @ Q
        c = 0.5 * (G[1, 0] - G[0, 1])
        a = 0.5 * (G[1, 0] + G[0, 1])
        b = 0.5 * (G[1, 1] - G[0, 0])
        orient = 1.0 if c >= 0 else -1.0
        theta = 0.5 * np.arctan2(orient*b, orient*a)
        u = np.array([-np.sin(theta), np.cos(theta)])
        vec = Q @ u
    n = np.linalg.norm(vec)
    return vec/n if n > 1e-12 and np.isfinite(n) else None

def entropy_reaction_support(vec):
    """As in the manuscript: exp(-sum p log p)/N, p_j = r_j^2/sum r_k^2."""
    if vec is None:
        return np.nan
    p = np.square(np.asarray(vec, dtype=float))
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        return np.nan
    p = p/total
    positive = p > 0
    return float(np.exp(-np.sum(p[positive]*np.log(p[positive])))/len(p))

def reduced_eigenvalue_splitting(G):
    """Original descriptive splitting abs((lambda1-lambda2)/(lambda1+lambda2))."""
    lam = np.linalg.eigvals(G)
    den = abs(lam[0]+lam[1])
    if den <= 1e-8 * max(1., np.max(abs(lam))):
        return np.nan
    return float(abs(lam[0]-lam[1])/den)

def geom_one_window(X, s6):
    F = s6.fit_v4_full_window(X)
    full_rho = float(np.max(abs(np.linalg.eigvals(F))))
    r = {}
    for method in ('M2', 'M3'):
        for suffix in ('R_A_check', 'R_sigma_check', 'support', 'Delta'):
            r[f'{method}_{suffix}'] = np.nan
    if not np.isfinite(full_rho) or full_rho >= 1.-s6.TOL:
        return r
    for method in ('M2', 'M3'):
        Q,_ = s6.method2(F) if method=='M2' else s6.method3(F)
        if Q is None: continue
        G = Q.T @ F @ Q
        metrics = s6.ratio_metrics(G, 1./s6.FS)
        r[f'{method}_R_A_check'] = metrics['R_A']
        r[f'{method}_R_sigma_check'] = metrics['R_sigma']
        if RA_AND_RSIGMA_REQUIRE_FINITE and not (np.isfinite(metrics['R_A']) and
                                                  np.isfinite(metrics['R_sigma'])):
            continue
        r[f'{method}_support'] = entropy_reaction_support(
            reaction_mode_from_Q(F, Q, method))
        r[f'{method}_Delta'] = reduced_eigenvalue_splitting(G)
    return r
