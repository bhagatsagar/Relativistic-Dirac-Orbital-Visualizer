"""
physics.py — Exact analytic Dirac–Coulomb (point nucleus) bound-state core
+ exact angular algebra + exact E1 dipole matrix elements.

Scope (strict):
- V(r) = - Z * α / r   (point nucleus)
- Natural units: ħ = c = m = 1
- Bound stationary states only

Spinor convention (given):
Ψ(r,θ,φ) = (1/r) [ G(r) Ω_{κ,mj}(θ,φ),
                   i F(r) Ω_{-κ,mj}(θ,φ) ]^T

Radial normalization (given):
∫_0^∞ [G(r)^2 + F(r)^2] dr = 1

Angular conventions (Prompt 2):
- Spinor spherical harmonics Ω_{κ,m} built exactly from Clebsch–Gordan coefficients and Y_{lm}.
- E1 operator uses spherical components of position:
    r_q = r C^1_q,  where C^k_q = sqrt(4π/(2k+1)) Y_{kq}.
- Wigner–Eckart:
    ⟨j_f m_f | C^1_q | j_i m_i⟩
      = (-1)^(j_f-m_f) ( j_f 1 j_i ; -m_f q m_i ) ⟨j_f||C^1||j_i⟩
  with ⟨j_f||C^1||j_i⟩ computed exactly from (l, s=1/2) coupling using a 6j and an orbital 3j.

Radial Dirac equations used for validation (m=1), with total energy E (incl. rest mass):
  dG/dr + (κ/r) G = (1 + E - V) F
  dF/dr - (κ/r) F = (1 - E + V) G
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Literal, Tuple

import math
import numpy as np

try:
    from scipy import integrate, special
except Exception as e:  # pragma: no cover
    raise ImportError(
        "physics.py requires SciPy (scipy.special, scipy.integrate) for exact special functions."
    ) from e

try:
    # Used for radial dipole on merged grids (stable, no extrapolation required)
    from scipy.interpolate import PchipInterpolator
except Exception as e:  # pragma: no cover
    raise ImportError("physics.py requires scipy.interpolate (PchipInterpolator).") from e


# -----------------------------------------------------------------------------
# Constants / tolerances
# -----------------------------------------------------------------------------

# Fine-structure constant α (dimensionless). Value chosen as a precise float.
ALPHA_FS: float = 7.2973525693e-3

# Numeric tolerances for quantum number validation
_QN_TOL: float = 1e-12

# Tolerance for treating polarization spherical components as "present"
_POL_TOL_REL: float = 1e-12

# If selection rules forbid, dipole_e1 returns a ~0 vector. This threshold documents intent.
_FORBIDDEN_DIPOLE_MAG: float = 1e-12


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundQN:
    """Bound-state quantum numbers for Dirac–Coulomb.

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    kappa : int
        Relativistic angular quantum number (κ != 0).
    mj : float
        Magnetic projection, half-integer, |mj| <= j.
    Z : int
        Nuclear charge (Z >= 1).
    """
    n: int
    kappa: int
    mj: float
    Z: int


@dataclass(frozen=True)
class RadialSolution:
    """Radial solution container.

    Attributes
    ----------
    qn : BoundQN
        Quantum numbers.
    E : float
        Energy. If include_rest_mass=True, this is total energy E in (0,1).
        If include_rest_mass=False, this is binding energy E-1 (negative).
    r : np.ndarray
        Radial grid (monotone increasing, strictly positive).
    G : np.ndarray
        Large component radial function G(r).
    F : np.ndarray
        Small component radial function F(r).
    include_rest_mass : bool
        Whether E includes rest mass (m=1) or not.
    """
    qn: BoundQN
    E: float
    r: np.ndarray
    G: np.ndarray
    F: np.ndarray
    include_rest_mass: bool


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _phase_int(n: int) -> float:
    """Return (-1)^n exactly for integer n."""
    return -1.0 if (n & 1) else 1.0


def _is_half_integer(x: float, *, tol: float = _QN_TOL) -> bool:
    """Return True if x is (approximately) a half-integer."""
    two_x = 2.0 * x
    return abs(two_x - round(two_x)) <= tol


def _as_int_if_close(x: float, *, tol: float = _QN_TOL) -> int:
    """Round to nearest int if within tol, else raise."""
    xi = int(round(x))
    if abs(x - xi) > tol:
        raise ValueError(f"Value {x} is not integer within tolerance.")
    return xi


def _as_halfint_twice(x: float, *, tol: float = _QN_TOL) -> int:
    """Represent a half-integer x by an integer 2x (exact within tol)."""
    tx = int(round(2.0 * x))
    if abs(2.0 * x - tx) > tol:
        raise ValueError(f"Value {x} is not a half-integer within tolerance.")
    return tx


# -----------------------------------------------------------------------------
# Quantum number utilities + validation
# -----------------------------------------------------------------------------

def kappa_to_l_j(kappa: int) -> Tuple[int, float]:
    """Convert κ to (l, j).

    Explicit mapping:
    - if κ > 0: l = κ,     j = κ - 1/2
    - if κ < 0: l = -κ-1,  j = -κ - 1/2
    """
    if kappa == 0:
        raise ValueError("kappa must be nonzero.")
    if kappa > 0:
        l = kappa
        j = kappa - 0.5
    else:
        l = -kappa - 1
        j = -kappa - 0.5
    return int(l), float(j)


def validate_qn(qn: BoundQN) -> Dict[str, Any]:
    """Strict validation of Dirac–Coulomb bound-state quantum numbers.

    Raises ValueError on invalid input. Returns derived values on success.
    """
    if not isinstance(qn.n, int) or qn.n < 1:
        raise ValueError("n must be an integer >= 1.")
    if not isinstance(qn.kappa, int) or qn.kappa == 0:
        raise ValueError("kappa must be a nonzero integer.")
    if not isinstance(qn.Z, int) or qn.Z < 1:
        raise ValueError("Z must be an integer >= 1.")
    if not _is_half_integer(qn.mj):
        raise ValueError("mj must be a half-integer (…, -3/2, -1/2, +1/2, +3/2, …).")

    l, j = kappa_to_l_j(qn.kappa)
    if qn.n <= l:
        raise ValueError(f"Require n > l for bound states. Got n={qn.n}, l={l} (from kappa={qn.kappa}).")
    if abs(qn.mj) > j + _QN_TOL:
        raise ValueError(f"Require |mj| <= j. Got mj={qn.mj}, j={j}.")

    za = qn.Z * ALPHA_FS
    if za >= abs(qn.kappa):
        raise ValueError(
            f"Point-nucleus Dirac requires Z*alpha < |kappa| for real gamma. "
            f"Got Z*alpha={za:.6g}, |kappa|={abs(qn.kappa)}."
        )

    n_r = qn.n - abs(qn.kappa)
    if n_r < 0:
        raise ValueError("Require n - |kappa| >= 0 (nonnegative radial quantum number).")

    gamma = math.sqrt(qn.kappa * qn.kappa - za * za)

    return {
        "l": l,
        "j": j,
        "n_r": int(n_r),
        "gamma": float(gamma),
        "Zalpha": float(za),
    }


# -----------------------------------------------------------------------------
# Energy (exact closed form)
# -----------------------------------------------------------------------------

def energy_dirac_coulomb(qn: BoundQN, include_rest_mass: bool) -> float:
    """Exact point-nucleus Dirac–Coulomb bound-state energy.

    Uses:
      γ = sqrt(κ^2 - (Zα)^2)
      n_r = n - |κ|
      E_total = 1 / sqrt(1 + (Zα / (n_r + γ))^2)   (m=1)
    """
    dv = validate_qn(qn)
    za = dv["Zalpha"]
    gamma = dv["gamma"]
    n_r = dv["n_r"]

    denom = n_r + gamma
    if denom <= 0:
        raise ValueError("Invalid state: n_r + gamma must be positive.")
    E_total = 1.0 / math.sqrt(1.0 + (za / denom) ** 2)

    return E_total if include_rest_mass else (E_total - 1.0)


# -----------------------------------------------------------------------------
# Exact analytic radial functions (Laguerre closed form)
# -----------------------------------------------------------------------------

def _eval_genlaguerre_safe(n: int, a: float, x: np.ndarray) -> np.ndarray:
    """Safe generalized Laguerre evaluation. Returns zeros for n < 0."""
    if n < 0:
        return np.zeros_like(x, dtype=np.float64)
    return special.eval_genlaguerre(int(n), float(a), x).astype(np.float64)


def _default_grid_bounds(qn: BoundQN, *, E_total: float) -> Tuple[float, float]:
    """Heuristic default r_min, r_max.

    In these units, a0 = 1/(Zα), and characteristic radius ~ n^2 a0.
    Decay length ~ 1/p with p = sqrt(1 - E^2).
    """
    dv = validate_qn(qn)
    za = dv["Zalpha"]
    n = qn.n

    a0 = 1.0 / za
    r_char = (n * n) * a0
    p = math.sqrt(max(0.0, 1.0 - E_total * E_total))
    r_decay = (1.0 / p) if p > 0 else (10.0 * r_char)

    r_min = max(1e-12, 1e-10 * r_char)
    r_max = max(200.0 * r_char, 60.0 * r_decay)
    return float(r_min), float(r_max)


def solve_radial_analytic(
    qn: BoundQN,
    *,
    include_rest_mass: bool = True,
    r_min: float | None = None,
    r_max: float | None = None,
    n_points: int = 4000,
    grid: Literal["log", "linear"] = "log",
) -> RadialSolution:
    """Compute exact analytic Dirac–Coulomb bound-state radial functions G(r), F(r).

    Closed form (ρ = 2 p r, p = sqrt(1 - E^2), γ = sqrt(κ^2 - (Zα)^2), n_r = n - |κ|):
      pref(ρ) = ρ^γ * exp(-ρ/2)
      a  = n_r + 2γ
      ν  = n_r + γ
      ξ  = κ - ν

      G(r) = N0 * pref(ρ) * [ a * L_{n_r-1}^{2γ}(ρ) + ξ * L_{n_r}^{2γ}(ρ) ]
      F(r) = N0 * pref(ρ) * [ a * L_{n_r-1}^{2γ}(ρ) - ξ * L_{n_r}^{2γ}(ρ) ] * (p/(1+E))

    N0 is a stabilized analytic scale prefactor (log-gamma); final normalization is enforced numerically.
    """
    dv = validate_qn(qn)
    if n_points < 50:
        raise ValueError("n_points must be >= 50 for stable normalization/validation.")

    gamma = dv["gamma"]
    n_r = dv["n_r"]
    kappa = qn.kappa

    # Always compute E_total including rest mass for physics; return per include_rest_mass.
    E_total = energy_dirac_coulomb(qn, include_rest_mass=True)
    if not (0.0 < E_total < 1.0):
        raise ValueError("Expected positive-energy bound state with 0 < E_total < 1.")

    # Grid bounds
    if r_min is None or r_max is None:
        rmin_def, rmax_def = _default_grid_bounds(qn, E_total=E_total)
        r_min = rmin_def if r_min is None else r_min
        r_max = rmax_def if r_max is None else r_max

    r_min = float(r_min)
    r_max = float(r_max)
    if not (r_min > 0.0 and r_max > r_min):
        raise ValueError("Require 0 < r_min < r_max.")

    # Build grid
    if grid == "log":
        r = np.exp(np.linspace(np.log(r_min), np.log(r_max), int(n_points), dtype=np.float64))
    elif grid == "linear":
        r = np.linspace(r_min, r_max, int(n_points), dtype=np.float64)
        if np.any(r <= 0):
            raise ValueError("Linear grid produced non-positive r; choose r_min > 0.")
    else:
        raise ValueError("grid must be 'log' or 'linear'.")

    # Derived parameters
    p = math.sqrt(max(0.0, 1.0 - E_total * E_total))
    if p <= 0.0:
        raise ValueError("p = sqrt(1-E^2) must be positive for bound states.")
    rho = (2.0 * p) * r
    rho = np.maximum(rho, np.finfo(np.float64).tiny)

    # Laguerre polynomials
    Lnr = _eval_genlaguerre_safe(n_r, 2.0 * gamma, rho)
    Lnm1 = _eval_genlaguerre_safe(n_r - 1, 2.0 * gamma, rho)

    # ν = n_r + γ, ξ = κ - ν, a = n_r + 2γ
    nu = float(n_r) + float(gamma)
    xi = float(kappa) - nu
    a = float(n_r) + 2.0 * float(gamma)

    # pref = ρ^γ e^{-ρ/2} in log form
    pref = np.exp(gamma * np.log(rho) - 0.5 * rho)

    # Stabilized analytic scale prefactor N0 (final normalization is numerical).
    logN0 = 0.5 * (2.0 * gamma + 1.0) * math.log(2.0 * p)
    logN0 += 0.5 * (special.gammaln(n_r + 1.0) - special.gammaln(n_r + 2.0 * gamma + 1.0))
    logN0 += 0.5 * (math.log(1.0 + E_total) - math.log(2.0))
    N0 = float(math.exp(logN0))

    poly_plus = (a * Lnm1 + xi * Lnr)
    poly_minus = (a * Lnm1 - xi * Lnr)

    G = (N0 * pref * poly_plus).astype(np.float64)
    F = (N0 * pref * poly_minus * (p / (1.0 + E_total))).astype(np.float64)

    E_return = E_total if include_rest_mass else (E_total - 1.0)
    sol0 = RadialSolution(qn=qn, E=E_return, r=r, G=G, F=F, include_rest_mass=include_rest_mass)
    return normalize_radial(sol0)


# -----------------------------------------------------------------------------
# Normalization + residual validation
# -----------------------------------------------------------------------------

def _integrate_norm2(r: np.ndarray, G: np.ndarray, F: np.ndarray) -> float:
    """Compute ∫ (G^2 + F^2) dr with Simpson rule on nonuniform x-grid."""
    y = (G * G + F * F).astype(np.float64)
    return float(integrate.simpson(y, x=r))


def normalize_radial(sol: RadialSolution) -> RadialSolution:
    """Enforce ∫(G^2+F^2)dr = 1 on the provided grid."""
    norm2 = _integrate_norm2(sol.r, sol.G, sol.F)
    if not np.isfinite(norm2) or norm2 <= 0.0:
        raise ValueError("Normalization integral is non-finite or non-positive. Check grid/parameters.")
    norm = math.sqrt(norm2)
    G = (sol.G / norm).astype(np.float64)
    F = (sol.F / norm).astype(np.float64)
    return RadialSolution(qn=sol.qn, E=sol.E, r=sol.r, G=G, F=F, include_rest_mass=sol.include_rest_mass)


def validate_radial(sol: RadialSolution) -> Dict[str, Any]:
    """Validate normalization and radial Dirac equation residuals on the solution grid."""
    validate_qn(sol.qn)
    kappa = sol.qn.kappa

    # Convert reported E to total E used in equations
    E_total = sol.E if sol.include_rest_mass else (sol.E + 1.0)

    r = sol.r.astype(np.float64)
    G = sol.G.astype(np.float64)
    F = sol.F.astype(np.float64)

    norm2 = _integrate_norm2(r, G, F)
    norm_error = abs(math.sqrt(norm2) - 1.0)

    V = -(sol.qn.Z * ALPHA_FS) / r
    dG = np.gradient(G, r, edge_order=2)
    dF = np.gradient(F, r, edge_order=2)

    resG = dG + (kappa / r) * G - (1.0 + E_total - V) * F
    resF = dF - (kappa / r) * F - (1.0 - E_total + V) * G

    max_resG = float(np.max(np.abs(resG)))
    max_resF = float(np.max(np.abs(resF)))
    max_resT = float(max(max_resG, max_resF))

    return {
        "norm_error": float(norm_error),
        "max_residual_G": max_resG,
        "max_residual_F": max_resF,
        "max_residual_total": max_resT,
        "grid_info": {
            "r_min": float(r[0]),
            "r_max": float(r[-1]),
            "n_points": int(r.size),
        },
    }


# -----------------------------------------------------------------------------
# Exact Wigner symbols (half-integers supported)
# -----------------------------------------------------------------------------

# Prefer SciPy's wigner_3j / wigner_6j if present; fallback to SymPy exact routines.
_HAVE_SCIPY_WIGNER_3J = hasattr(special, "wigner_3j")
_HAVE_SCIPY_WIGNER_6J = hasattr(special, "wigner_6j")

if not (_HAVE_SCIPY_WIGNER_3J and _HAVE_SCIPY_WIGNER_6J):
    try:
        import sympy as _sp
        from sympy.physics.wigner import wigner_3j as _sympy_wigner_3j
        from sympy.physics.wigner import wigner_6j as _sympy_wigner_6j
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SciPy wigner_3j/6j not available and SymPy fallback could not be imported."
        ) from e
else:
    _sp = None
    _sympy_wigner_3j = None
    _sympy_wigner_6j = None


def _sympy_rational_halfint(x: float) -> Any:
    """Convert integer/half-integer float to SymPy Rational exactly."""
    tx = _as_halfint_twice(x)
    return _sp.Rational(tx, 2)


@lru_cache(maxsize=4096)
def wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Exact Wigner 3j symbol for integer/half-integer arguments."""
    if _HAVE_SCIPY_WIGNER_3J:
        return float(special.wigner_3j(j1, j2, j3, m1, m2, m3))
    # SymPy exact
    J1 = _sympy_rational_halfint(j1)
    J2 = _sympy_rational_halfint(j2)
    J3 = _sympy_rational_halfint(j3)
    M1 = _sympy_rational_halfint(m1)
    M2 = _sympy_rational_halfint(m2)
    M3 = _sympy_rational_halfint(m3)
    val = _sympy_wigner_3j(J1, J2, J3, M1, M2, M3)
    return float(_sp.N(val, 50))


@lru_cache(maxsize=4096)
def wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    """Exact Wigner 6j symbol for integer/half-integer arguments."""
    if _HAVE_SCIPY_WIGNER_6J:
        return float(special.wigner_6j(j1, j2, j3, j4, j5, j6))
    # SymPy exact
    J1 = _sympy_rational_halfint(j1)
    J2 = _sympy_rational_halfint(j2)
    J3 = _sympy_rational_halfint(j3)
    J4 = _sympy_rational_halfint(j4)
    J5 = _sympy_rational_halfint(j5)
    J6 = _sympy_rational_halfint(j6)
    val = _sympy_wigner_6j(J1, J2, J3, J4, J5, J6)
    return float(_sp.N(val, 50))


# -----------------------------------------------------------------------------
# Spinor spherical harmonics Ω_{κ,m} (exact CG algebra)
# -----------------------------------------------------------------------------

def _clebsch_gordan(l: int, ml: int, s: float, ms: float, j: float, m: float) -> float:
    """Compute ⟨l ml, s ms | j m⟩ exactly using Wigner 3j."""
    # <l ml s ms | j m> = (-1)^(l - s + m) sqrt(2j+1) ( l s j ; ml ms -m )
    exp_int = _as_int_if_close(l - s + m)
    return _phase_int(exp_int) * math.sqrt(2.0 * j + 1.0) * wigner_3j(l, s, j, ml, ms, -m)


def spinor_spherical_harmonic(
    kappa: int,
    mj: float,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Compute the 2-component spinor spherical harmonic Ω_{κ,mj}(θ,φ).

    Ω_{κ,m} = Σ_{m_l,m_s} ⟨l m_l, 1/2 m_s | j m⟩ Y_{l,m_l}(θ,φ) χ_{m_s}

    Returns
    -------
    np.ndarray
        Complex array with shape broadcast(theta,phi) + (2,),
        where [...,0] is the m_s=+1/2 component and [...,1] is the m_s=-1/2 component.
    """
    if kappa == 0:
        raise ValueError("kappa must be nonzero.")
    if not _is_half_integer(mj):
        raise ValueError("mj must be a half-integer.")

    l, j = kappa_to_l_j(kappa)
    if abs(mj) > j + _QN_TOL:
        raise ValueError(f"|mj| must be <= j. Got mj={mj}, j={j}.")

    th = np.asarray(theta, dtype=np.float64)
    ph = np.asarray(phi, dtype=np.float64)
    th, ph = np.broadcast_arrays(th, ph)

    # m_l for the two spin components:
    # m = m_l + m_s
    # component 0: m_s = +1/2 => m_l = m - 1/2
    # component 1: m_s = -1/2 => m_l = m + 1/2
    ml_up = _as_int_if_close(mj - 0.5)
    ml_dn = _as_int_if_close(mj + 0.5)

    out = np.zeros(th.shape + (2,), dtype=np.complex128)

    # SciPy sph_harm uses sph_harm(m, l, phi, theta)
    if -l <= ml_up <= l:
        cg_up = _clebsch_gordan(l, ml_up, 0.5, +0.5, j, mj)
        Y_up = special.sph_harm(ml_up, l, ph, th)
        out[..., 0] = cg_up * Y_up

    if -l <= ml_dn <= l:
        cg_dn = _clebsch_gordan(l, ml_dn, 0.5, -0.5, j, mj)
        Y_dn = special.sph_harm(ml_dn, l, ph, th)
        out[..., 1] = cg_dn * Y_dn

    return out


# -----------------------------------------------------------------------------
# E1 selection rules helper (exact, no approximations)
# -----------------------------------------------------------------------------

def _polarization_spherical(polarization_xyz: np.ndarray) -> Dict[int, complex]:
    """Convert Cartesian polarization vector (x,y,z) to spherical components (q=-1,0,+1)."""
    eps = np.asarray(polarization_xyz, dtype=np.complex128).reshape(3)
    ex, ey, ez = eps[0], eps[1], eps[2]
    # Vector spherical basis:
    # v_{+1} = -(v_x + i v_y)/sqrt(2)
    # v_{-1} =  (v_x - i v_y)/sqrt(2)
    # v_0    =  v_z
    vp1 = -(ex + 1j * ey) / math.sqrt(2.0)
    vm1 = (ex - 1j * ey) / math.sqrt(2.0)
    v0 = ez
    return {-1: vm1, 0: v0, +1: vp1}

def check_e1_selection_rules(
    qn_i: BoundQN,
    qn_f: BoundQN,
    polarization_xyz: np.ndarray | None,
) -> Dict[str, Any]:
    """Check exact E1 (k=1) selection rules for a specific (i -> f) transition.

    Convention:
    - Intrinsic E1 selection rules depend only on (qn_i, qn_f).
    - If polarization_xyz is provided, we additionally require that ε has the needed spherical q=Δm component.
      (This is a *field/polarization gating* layer, not part of the dipole definition.)

    Returns fields including:
      - allowed_intrinsic: passes intrinsic E1 rules (independent of ε)
      - allowed: passes intrinsic rules AND is allowed given ε
      - required_q: the single q = Δm needed (if Δm ∈ {-1,0,1}), else []
      - active_q: q components present in ε (or all if polarization_xyz is None)
      - used_q: required_q if allowed else []
    """
    dv_i = validate_qn(qn_i)
    dv_f = validate_qn(qn_f)

    if qn_i.Z != qn_f.Z:
        return {
            "allowed": False,
            "allowed_intrinsic": False,
            "reason": "Different Z for initial and final states (not the same Coulomb problem).",
        }

    li, ji = dv_i["l"], dv_i["j"]
    lf, jf = dv_f["l"], dv_f["j"]

    # Δm must be integer for dipole (difference of half-integers).
    dm = qn_f.mj - qn_i.mj
    dm2 = _as_halfint_twice(dm)
    if dm2 % 2 != 0:
        return {
            "allowed": False,
            "allowed_intrinsic": False,
            "reason": "Δm is not integer (unexpected).",
        }
    dm_int = dm2 // 2  # integer

    # Polarization determines which q are available (field gating).
    pol_provided = polarization_xyz is not None
    if polarization_xyz is None:
        pol_sph = {-1: 1.0 + 0.0j, 0: 1.0 + 0.0j, +1: 1.0 + 0.0j}
        active_q = {-1, 0, +1}
    else:
        pol_sph = _polarization_spherical(polarization_xyz)
        norm = math.sqrt(float(np.vdot(polarization_xyz, polarization_xyz).real)) if np.any(polarization_xyz) else 0.0
        tol = _POL_TOL_REL * (norm if norm > 0 else 1.0) + 1e-15
        active_q = {q for q, val in pol_sph.items() if abs(val) > tol}

    # Intrinsic E1 rules:
    dl = lf - li
    orb_ok = (dl == +1) or (dl == -1)  # E1 parity change
    jtri_ok = (abs(jf - ji) <= 1.0 + 1e-15) and (jf + ji >= 1.0 - 1e-15)  # k=1 triangle
    m_ok_intrinsic = (dm_int in {-1, 0, +1})

    allowed_intrinsic = bool(orb_ok and jtri_ok and m_ok_intrinsic)
    allowed = bool(allowed_intrinsic and (dm_int in active_q))

    required_q = [int(dm_int)] if m_ok_intrinsic else []
    used_q = [int(dm_int)] if allowed else []

    reason = "allowed" if allowed else "forbidden"
    if not orb_ok:
        reason = "forbidden: requires l_f = l_i ± 1 (E1 parity change)"
    elif not jtri_ok:
        reason = "forbidden: j triangle rule fails for k=1"
    elif not m_ok_intrinsic:
        reason = "forbidden: Δm not in {-1,0,+1} for E1"
    elif dm_int not in active_q:
        reason = "forbidden: polarization lacks required q = Δm component"

    return {
        "allowed": allowed,
        "allowed_intrinsic": allowed_intrinsic,
        "reason": reason,
        "delta_m": float(dm),
        "delta_m_int": int(dm_int),
        "required_q": required_q,
        "active_q": sorted(active_q),
        "used_q": used_q,
        "polarization_provided": bool(pol_provided),
        "polarization_spherical": {int(q): complex(pol_sph[q]) for q in (-1, 0, +1)},
        "li": int(li),
        "lf": int(lf),
        "ji": float(ji),
        "jf": float(jf),
        "delta_l": int(dl),
        "delta_j": float(jf - ji),
    }


# -----------------------------------------------------------------------------
# Reduced matrix element ⟨j_f||C^1||j_i⟩ and angular factors (exact)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def reduced_C1(kappa_f: int, kappa_i: int) -> float:
    """Compute ⟨j_f||C^1||j_i⟩ exactly for spinor spherical harmonics.

    Uses coupling (l,s=1/2)->j and an orbital reduced matrix element:

      ⟨ (l_f s) j_f || C^k || (l_i s) j_i ⟩
        = (-1)^(l_f + s + j_i + k) √[(2j_f+1)(2j_i+1)] { l_f j_f s ; j_i l_i k } ⟨l_f||C^k||l_i⟩

      ⟨l_f||C^k||l_i⟩ = (-1)^(l_f) √[(2l_f+1)(2l_i+1)] ( l_f k l_i ; 0 0 0 )

    Here k=1 and s=1/2. This form contains both a 6j and an orbital 3j (no shortcuts).

    Returns 0.0 automatically when selection rules imply zero (via Wigner symbols).
    """
    l_f, j_f = kappa_to_l_j(kappa_f)
    l_i, j_i = kappa_to_l_j(kappa_i)
    k = 1.0
    s = 0.5

    # Orbital reduced element
    three_l = wigner_3j(float(l_f), k, float(l_i), 0.0, 0.0, 0.0)
    l_red = _phase_int(l_f) * math.sqrt((2.0 * l_f + 1.0) * (2.0 * l_i + 1.0)) * three_l

    # Coupling 6j
    six = wigner_6j(float(l_f), float(j_f), s, float(j_i), float(l_i), k)

    exp_int = _as_int_if_close(l_f + s + j_i + k)  # guaranteed integer
    pref = _phase_int(exp_int) * math.sqrt((2.0 * j_f + 1.0) * (2.0 * j_i + 1.0))

    return float(pref * six * l_red)


def angular_C1(kappa_f: int, mj_f: float, kappa_i: int, mj_i: float, q: int) -> complex:
    """Compute ⟨j_f m_f | C^1_q | j_i m_i⟩ exactly using Wigner–Eckart."""
    if q not in (-1, 0, +1):
        raise ValueError("q must be in {-1,0,+1}.")
    if not (_is_half_integer(mj_f) and _is_half_integer(mj_i)):
        raise ValueError("mj values must be half-integers.")

    _, j_f = kappa_to_l_j(kappa_f)
    _, j_i = kappa_to_l_j(kappa_i)
    if abs(mj_f) > j_f + _QN_TOL or abs(mj_i) > j_i + _QN_TOL:
        return 0.0 + 0.0j

    # (-1)^(j_f - m_f)
    exp_int = _as_int_if_close(j_f - mj_f)
    phase = _phase_int(exp_int)

    tj = wigner_3j(float(j_f), 1.0, float(j_i), -float(mj_f), float(q), float(mj_i))
    red = reduced_C1(kappa_f, kappa_i)
    return complex(phase * tj * red)


# -----------------------------------------------------------------------------
# Radial dipole integral (exact model on merged grid) + E1 dipole API
# -----------------------------------------------------------------------------

def radial_dipole_integral(sol_i: RadialSolution, sol_f: RadialSolution) -> float:
    """Compute the E1 radial integral:
        R_if = ∫_0^∞ r [G_f(r) G_i(r) + F_f(r) F_i(r)] dr

    Implementation:
    - Uses a merged grid from both solutions (union of r arrays).
    - Interpolates each component using PCHIP (shape-preserving, stable near nodes).
    - Integrates with Simpson on the merged grid.
    - Stores a simple error estimate (|Simpson - trapezoid|) as:
          radial_dipole_integral.last_error

    Returns
    -------
    float
        The radial integral value (real for real radials).
    """
    if sol_i.qn.Z != sol_f.qn.Z:
        raise ValueError("Radial dipole integral requires the same Z for initial and final states.")

    ri = np.asarray(sol_i.r, dtype=np.float64)
    rf = np.asarray(sol_f.r, dtype=np.float64)

    r_lo = float(min(ri[0], rf[0]))
    r_hi = float(max(ri[-1], rf[-1]))

    # merged grid: union of both, clipped to [r_lo, r_hi]
    r_merge = np.unique(np.concatenate([ri, rf]))
    r_merge = r_merge[(r_merge >= r_lo) & (r_merge <= r_hi)]
    if r_merge.size < 50:
        raise ValueError("Merged grid too small for stable dipole integration.")

    # Interpolators on each solution's own domain
    Gi = PchipInterpolator(ri, sol_i.G, extrapolate=False)
    Fi = PchipInterpolator(ri, sol_i.F, extrapolate=False)
    Gf = PchipInterpolator(rf, sol_f.G, extrapolate=False)
    Ff = PchipInterpolator(rf, sol_f.F, extrapolate=False)

    def _eval_safe(itp: Any, x: np.ndarray) -> np.ndarray:
        y = itp(x)
        # outside the interpolator domain => nan when extrapolate=False; treat as 0 (tails)
        y = np.where(np.isfinite(y), y, 0.0)
        return y

    Gi_v = _eval_safe(Gi, r_merge)
    Fi_v = _eval_safe(Fi, r_merge)
    Gf_v = _eval_safe(Gf, r_merge)
    Ff_v = _eval_safe(Ff, r_merge)

    integrand = r_merge * (Gf_v * Gi_v + Ff_v * Fi_v)
    val_simpson = float(integrate.simpson(integrand, x=r_merge))
    val_trapz = float(np.trapz(integrand, x=r_merge))
    err_est = abs(val_simpson - val_trapz)

    radial_dipole_integral.last_error = err_est  # type: ignore[attr-defined]
    return val_simpson


def _cartesian_from_spherical(d_m1: complex, d_0: complex, d_p1: complex) -> np.ndarray:
    """Convert spherical vector components (q=-1,0,+1) to Cartesian (x,y,z)."""
    # Using:
    # v_{+1} = -(v_x + i v_y)/sqrt(2)
    # v_{-1} =  (v_x - i v_y)/sqrt(2)
    # v_0    =  v_z
    vx = (d_m1 - d_p1) / math.sqrt(2.0)
    vy = 1j * (d_m1 + d_p1) / math.sqrt(2.0)
    vz = d_0
    return np.array([vx, vy, vz], dtype=np.complex128)


def project_dipole(d_xyz: np.ndarray, eps_xyz: np.ndarray, *, conjugate_eps: bool = False) -> complex:
    """Project a Cartesian dipole vector onto a (possibly complex) Cartesian polarization.

    Returns the complex scalar:
        s = ε · d   (Cartesian contraction, no implied normalization)

    Notes on complex ε and Hermiticity:
    - The dipole operator r is Hermitian, so d_fi = conj(d_if) componentwise.
    - With this convention (no conjugation on ε), the consistent reverse relation is:
          (ε · d_if) = conj( (conj(ε) · d_fi) )
      which is why `conjugate_eps=True` is convenient for “reverse” checks.
    """
    d = np.asarray(d_xyz, dtype=np.complex128).reshape(3)
    eps = np.asarray(eps_xyz, dtype=np.complex128).reshape(3)
    if conjugate_eps:
        eps = np.conjugate(eps)
    return complex(eps[0] * d[0] + eps[1] * d[1] + eps[2] * d[2])


def dipole_e1(sol_i: RadialSolution, sol_f: RadialSolution) -> np.ndarray:
    """Compute the exact E1 dipole matrix element vector (Cartesian), independent of polarization.

    Option A convention:
      - Returns full Cartesian vector d_xyz = ⟨f| r⃗ |i⟩.
      - Polarization dependence enters only through projection: (ε · d).

    Implementation:
      - Checks intrinsic E1 selection rules (polarization=None).
      - Computes radial integral R_if = ∫ r (Gf Gi + Ff Fi) dr.
      - Computes spherical components d_q = R_if * ⟨j_f m_f | C^1_q | j_i m_i⟩ for q=-1,0,+1.
      - Converts spherical -> Cartesian and returns.

    Side info:
      - dipole_e1.last_selection: intrinsic selection dict (polarization=None).
      - dipole_e1.last_spherical: {q: d_q} for q in {-1,0,+1}.
    """
    sel0 = check_e1_selection_rules(sol_i.qn, sol_f.qn, polarization_xyz=None)
    dipole_e1.last_selection = sel0  # type: ignore[attr-defined]

    if not sel0.get("allowed_intrinsic", False):
        dipole_e1.last_spherical = {-1: 0.0 + 0.0j, 0: 0.0 + 0.0j, +1: 0.0 + 0.0j}  # type: ignore[attr-defined]
        return np.zeros(3, dtype=np.complex128)

    R = radial_dipole_integral(sol_i, sol_f)

    d_sph: Dict[int, complex] = {}
    for q in (-1, 0, +1):
        d_sph[q] = complex(R) * angular_C1(sol_f.qn.kappa, sol_f.qn.mj, sol_i.qn.kappa, sol_i.qn.mj, q)

    dipole_e1.last_spherical = {int(q): complex(d_sph[q]) for q in (-1, 0, +1)}  # type: ignore[attr-defined]
    return _cartesian_from_spherical(d_sph[-1], d_sph[0], d_sph[+1])


def dipole_matrix_element(sol_i: RadialSolution, sol_f: RadialSolution, polarization_xyz: np.ndarray) -> complex:
    """(Polarization-dependent) scalar dipole coupling: ⟨f| ε·r⃗ |i⟩ = ε·⟨f|r⃗|i⟩.

    This is intentionally a *scalar* to avoid ambiguity:
      - dipole_e1(...) gives the full vector ⟨f|r⃗|i⟩
      - project_dipole(...) does the polarization projection ε·d
    """
    d = dipole_e1(sol_i, sol_f)
    return project_dipole(d, polarization_xyz)


# -----------------------------------------------------------------------------
# __main__ sanity runs: radial + one allowed and one forbidden E1 transition
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Dirac–Coulomb bound state sanity check (radial)")
    qn_g = BoundQN(n=1, kappa=-1, mj=+0.5, Z=1)  # 1s1/2
    sol_g = solve_radial_analytic(qn_g, include_rest_mass=True, n_points=4000, grid="log")
    E_g = energy_dirac_coulomb(qn_g, include_rest_mass=True)
    met_g = validate_radial(sol_g)
    print(f"  qn = (Z={qn_g.Z}, n={qn_g.n}, kappa={qn_g.kappa}, mj={qn_g.mj:+g})")
    print(f"  E (total, m=1) = {E_g:.15f}")
    print(f"  norm_error        = {met_g['norm_error']:.3e}")
    print(f"  max_residual_total = {met_g['max_residual_total']:.3e}")

    print("\nE1 dipole sanity checks (hydrogen)")
    # Allowed intrinsically: 1s1/2 (κ=-1, m=+1/2) -> 2p1/2 (κ=+1, m=+1/2)
    qn_2p = BoundQN(n=2, kappa=+1, mj=+0.5, Z=1)   # 2p1/2
    sol_2p = solve_radial_analytic(qn_2p, include_rest_mass=True, n_points=5000, grid="log")

    # Forbidden intrinsically: 1s1/2 -> 2s1/2 (Δl=0)
    qn_2s = BoundQN(n=2, kappa=-1, mj=+0.5, Z=1)   # 2s1/2
    sol_2s = solve_radial_analytic(qn_2s, include_rest_mass=True, n_points=5000, grid="log")

    eps_z = np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    eps_x = np.array([1.0, 0.0, 0.0], dtype=np.complex128)

    # Full dipole vector is polarization-independent
    d_full = dipole_e1(sol_g, sol_2p)
    dmag_full = float(np.linalg.norm(d_full))

    sel_z = check_e1_selection_rules(qn_g, qn_2p, eps_z)
    sel_x = check_e1_selection_rules(qn_g, qn_2p, eps_x)

    g_z = project_dipole(d_full, eps_z)
    g_x = project_dipole(d_full, eps_x)

    print("  Transition 1s1/2 -> 2p1/2 (vector dipole, polarization-independent):")
    print(f"    d_full = {d_full}")
    print(f"    |d_full| = {dmag_full:.6e}   (radial_err≈{getattr(radial_dipole_integral, 'last_error', float('nan')):.3e})")

    print("  Polarization changes the *scalar coupling* |ε·d| (z vs x):")
    print(f"    selection (z-pol): allowed={sel_z['allowed']}, used_q={sel_z.get('used_q')}, active_q={sel_z.get('active_q')}")
    print(f"      |ε_z·d| = {abs(g_z):.6e}")
    print(f"    selection (x-pol): allowed={sel_x['allowed']}, used_q={sel_x.get('used_q')}, active_q={sel_x.get('active_q')}")
    print(f"      |ε_x·d| = {abs(g_x):.6e}")

    # Intrinsically forbidden case
    sel_forb = check_e1_selection_rules(qn_g, qn_2s, eps_z)
    d_forb = dipole_e1(sol_g, sol_2s)
    g_forb = project_dipole(d_forb, eps_z)
    print("  Forbidden transition 1s1/2 -> 2s1/2 (intrinsic E1 forbidden):")
    print(f"    selection: allowed={sel_forb['allowed']}, allowed_intrinsic={sel_forb.get('allowed_intrinsic')}, reason={sel_forb['reason']}")
    print(f"    |d_full| = {float(np.linalg.norm(d_forb)):.6e}   (should be ~0; threshold { _FORBIDDEN_DIPOLE_MAG:.1e})")
    print(f"    |ε·d|    = {abs(g_forb):.6e}")

    # Hermiticity checks
    d_if = dipole_e1(sol_g, sol_2p)
    d_fi = dipole_e1(sol_2p, sol_g)
    herm_vec_err = float(np.max(np.abs(d_if - np.conjugate(d_fi))))

    # For scalar projection with complex ε:
    eps_c = np.array([1.0, 1.0j, 0.5], dtype=np.complex128)
    eps_c /= math.sqrt(float(np.vdot(eps_c, eps_c).real))
    s_if = project_dipole(d_if, eps_c)
    s_fi = project_dipole(d_fi, eps_c, conjugate_eps=True)  # conj(ε)·d_fi
    herm_s_err = abs(s_if - np.conjugate(s_fi))

    print("  Hermiticity checks:")
    print(f"    max| d(i->f) - conj(d(f->i)) | = {herm_vec_err:.3e}")
    print(f"    | (ε·d_if) - conj((conj ε)·d_fi) | = {herm_s_err:.3e}")