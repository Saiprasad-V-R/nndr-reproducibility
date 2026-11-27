"""Eq. (44) constant-spectrum non-normal similarity family.

The original code's Gaussian orthogonal matrices and fixed spectrum use
seed=1; this reproduces the original operator construction without the
incorrect old threshold.  The operator is intentionally non-symmetric and
is spectrally stable by similarity to diag(lambda).
"""
from __future__ import annotations
import numpy as np


def _orthogonal(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n,n)))
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q


def fixed_spectrum_family(n=200, rho=0.95, delta_sigma=0.05, seed=1):
    rng = np.random.default_rng(seed)
    U = _orthogonal(n,rng)
    V = _orthogonal(n,rng)
    scales = np.clip(1.0 + delta_sigma*rng.standard_normal(n-1), 0.1, 3.)
    eig = rng.uniform(0.2*rho, 0.95*rho, size=n)
    eig[0] = rho
    B = V.T @ np.diag(eig) @ V
    return {'U':U,'V':V,'B':B,'eigenvalues':eig,'scales':scales,'n':n,
            'seed':seed,'rho':rho,'delta_sigma':delta_sigma}


def operator_at_kappa(fam,kappa):
    kappa=float(kappa)
    if kappa <= 0: raise ValueError('kappa must be positive')
    scales=np.concatenate([fam['scales'],[1.0/kappa]])
    # F=U Sigma V.T Lambda V Sigma^-1 U.T
    F=fam['U'] @ ((scales[:,None] * fam['B']) / scales[None,:]) @ fam['U'].T
    return np.asarray(F,dtype=float)


def original_multiplier_ratio(G):
    """Legacy formula, only for selecting the fixed historical kappa values.

    NOT scientifically valid as the revised R.  Positive-real spectrum only.
    """
    z,V=np.linalg.eig(G)
    if np.max(abs(z.imag))>1e-10 or np.min(z.real)<=0:
        return float('nan')
    splitting=abs((z[0]-z[1])/(z[0]+z[1]))
    if splitting>=1: return float('nan')
    c=abs(np.vdot(V[:,0],V[:,1]))/(np.linalg.norm(V[:,0])*np.linalg.norm(V[:,1]))
    K=c/np.sqrt(max(1e-28,1-c*c))
    s=np.sqrt(1-splitting*splitting)
    Kc=np.sqrt(s/(1-s)) if splitting>0 else np.inf
    return float(K/Kc)
