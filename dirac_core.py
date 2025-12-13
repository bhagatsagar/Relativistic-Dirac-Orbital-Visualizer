# dirac_core.py - Relativistic Dirac-Coulomb Bound States and E1 Transitions
"""
Single-particle relativistic Dirac-Coulomb bound states with electric dipole (E1) transitions.

PHYSICS SCOPE:
- Analytic hydrogenic Dirac eigenstates (no PDE solving)
- Classical electromagnetic driving field
- No QED radiative corrections (Lamb shift, vacuum polarization, etc.)
- Truncated basis for time evolution (approximation for driven dynamics)

UNITS: Natural units with ℏ = c = m_electron = 1
CONVENTIONS: Heaviside-Lorentz electromagnetism
- Fine structure constant: α = e²/(4π) ≈ 1/137
- Electron charge: e = √(4πα) (positive magnitude)
- Electron has charge q = -e
- Coulomb potential for electron in nuclear field +Ze: V(r) = -Zα/r

PERFORMANCE OPTIMIZATIONS:
- Numba JIT compilation for numerical bottlenecks
- Parallel computation via ThreadPoolExecutor
- Vectorized NumPy operations with explicit threading control
- Pre-allocated arrays for reduced memory allocation
- Cached spinor stacks for repeated computations
"""
from __future__ import annotations
import os

# FIX CORE-NEW-001: Configure threading BEFORE importing numpy/scipy
# BLAS libraries read these at import time
_num_threads = os.cpu_count() or 4
os.environ.setdefault('OMP_NUM_THREADS', str(_num_threads))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_num_threads))
os.environ.setdefault('MKL_NUM_THREADS', str(_num_threads))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_num_threads))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict, Any
from math import factorial as math_factorial
import uuid
import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm, genlaguerre, eval_genlaguerre, gamma as scipy_gamma
from scipy.integrate import quad, solve_ivp
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Try to import numba for JIT compilation
try:
    from numba import jit, prange, set_num_threads
    NUMBA_AVAILABLE = True
    # Set numba thread count
    set_num_threads(_num_threads)
except Exception as e:
    # FIX BUG-001: Catch all exceptions (LLVM/llvmlite failures, etc.)
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    # Log the failure reason (logger not yet configured, use print)
    import sys
    print(f"Numba disabled: {type(e).__name__}: {e}", file=sys.stderr)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ArrayC = NDArray[np.complex128]
ArrayR = NDArray[np.float64]

# =============================================================================
# Physical Constants (Natural Units with Heaviside-Lorentz)
# =============================================================================

ALPHA_FS = 1.0 / 137.035999084  # Fine structure constant
E_CHARGE = np.sqrt(4.0 * np.pi * ALPHA_FS)  # Positive elementary charge magnitude
ELECTRON_Q = -E_CHARGE  # Electron charge (negative)

# Thread pool for parallel operations
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

def get_executor() -> ThreadPoolExecutor:
    """Get or create the global thread pool executor."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_num_threads)
    return _executor

def _shutdown_executor():
    """FIX BUG-008: Clean shutdown of thread pool on exit."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
            _executor = None

import atexit
atexit.register(_shutdown_executor)


# =============================================================================
# Numba-accelerated functions
# =============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_density_fast(spinor_real: ArrayR, spinor_imag: ArrayR) -> ArrayR:
    """
    Compute probability density from spinor components.
    
    JIT-compiled for maximum performance.
    """
    shape = spinor_real.shape[1:]
    result = np.zeros(shape, dtype=np.float64)
    
    for comp in range(4):
        for i in prange(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    re = spinor_real[comp, i, j, k]
                    im = spinor_imag[comp, i, j, k]
                    result[i, j, k] += re * re + im * im
    
    return result


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_superposition_fast(
    coeffs_real: ArrayR,
    coeffs_imag: ArrayR,
    spinors_real: ArrayR,
    spinors_imag: ArrayR,
    out_real: ArrayR,
    out_imag: ArrayR
) -> None:
    """
    Compute superposition of spinors: psi = sum_i c_i * spinor_i
    
    JIT-compiled with parallel loops.
    """
    n_states = len(coeffs_real)
    shape = spinors_real.shape[2:]  # (nx, ny, nz)
    
    # Zero output arrays
    out_real[:] = 0.0
    out_imag[:] = 0.0
    
    for s in range(n_states):
        cr = coeffs_real[s]
        ci = coeffs_imag[s]
        
        for comp in range(4):
            for i in prange(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        sr = spinors_real[s, comp, i, j, k]
                        si = spinors_imag[s, comp, i, j, k]
                        # Complex multiplication: (cr + i*ci) * (sr + i*si)
                        out_real[comp, i, j, k] += cr * sr - ci * si
                        out_imag[comp, i, j, k] += cr * si + ci * sr


@jit(nopython=True, cache=True)  # FIX DIRAC-001: Removed parallel=True to avoid race condition
def _radial_binning_fast(
    r_flat: ArrayR,
    density_flat: ArrayR,
    bin_edges: ArrayR,
    weights: ArrayR
) -> Tuple[ArrayR, ArrayR]:
    """
    Fast radial binning using numba.
    
    NOTE: parallel=True removed due to race condition when updating
    shared counts/sums arrays from multiple threads.
    """
    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins, dtype=np.float64)
    sums = np.zeros(n_bins, dtype=np.float64)
    
    n_points = len(r_flat)
    
    for i in range(n_points):  # FIX: Changed from prange to range
        r = r_flat[i]
        d = density_flat[i]
        w = weights[i]
        
        # Binary search for bin
        left = 0
        right = n_bins
        while left < right:
            mid = (left + right) // 2
            if bin_edges[mid + 1] <= r:
                left = mid + 1
            else:
                right = mid
        
        if left < n_bins and bin_edges[left] <= r < bin_edges[left + 1]:
            counts[left] += w
            sums[left] += d * w
    
    return sums, counts


@jit(nopython=True, cache=True, fastmath=True)
def _compute_color_volume_phase(psi_real_0: ArrayR, psi_imag_0: ArrayR) -> ArrayR:
    """Compute phase color volume from first spinor component."""
    shape = psi_real_0.shape
    result = np.empty(shape, dtype=np.float32)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                phase = np.arctan2(psi_imag_0[i, j, k], psi_real_0[i, j, k])
                result[i, j, k] = (phase + np.pi) / (2.0 * np.pi)
    
    return result


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_color_volume_spin(
    psi_real: ArrayR,
    psi_imag: ArrayR
) -> ArrayR:
    """Compute spin color volume: P_up - P_down, normalized to [0,1]."""
    shape = psi_real.shape[1:]  # (nx, ny, nz)
    result = np.empty(shape, dtype=np.float64)  # Use float64 for consistent typing
    
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # P_up = |ψ₀|² + |ψ₂|²
                p_up = (psi_real[0, i, j, k]**2 + psi_imag[0, i, j, k]**2 +
                        psi_real[2, i, j, k]**2 + psi_imag[2, i, j, k]**2)
                # P_down = |ψ₁|² + |ψ₃|²
                p_down = (psi_real[1, i, j, k]**2 + psi_imag[1, i, j, k]**2 +
                          psi_real[3, i, j, k]**2 + psi_imag[3, i, j, k]**2)
                result[i, j, k] = p_up - p_down
    
    # Normalize to [0, 1] using in-place operations to maintain type consistency
    mx = np.abs(result).max()
    if mx > 0:
        for i in prange(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    result[i, j, k] = result[i, j, k] / mx * 0.5 + 0.5
    else:
        for i in prange(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    result[i, j, k] = 0.5
    
    return result


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DiracGridConfig:
    """Configuration for 3D visualization grid (NOT used for physics integrals)."""
    nx: int = 64
    ny: int = 64
    nz: int = 64
    x_range: Tuple[float, float] = (-10000.0, 10000.0)
    y_range: Tuple[float, float] = (-10000.0, 10000.0)
    z_range: Tuple[float, float] = (-10000.0, 10000.0)

    def shape(self) -> Tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    @property
    def x_min(self) -> float:
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        return self.x_range[1]

    @property
    def y_min(self) -> float:
        return self.y_range[0]

    @property
    def y_max(self) -> float:
        return self.y_range[1]

    @property
    def z_min(self) -> float:
        return self.z_range[0]

    @property
    def z_max(self) -> float:
        return self.z_range[1]

    def coordinate_grids(self) -> Tuple[ArrayR, ArrayR, ArrayR]:
        x = np.linspace(self.x_min, self.x_max, self.nx, dtype=float)
        y = np.linspace(self.y_min, self.y_max, self.ny, dtype=float)
        z = np.linspace(self.z_min, self.z_max, self.nz, dtype=float)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z


@dataclass
class FieldConfig:
    """Configuration for electromagnetic fields."""
    Z: int = 1
    enable_coulomb: bool = True
    E_field: ArrayR = field(default_factory=lambda: np.zeros(3, dtype=float))
    B_field: ArrayR = field(default_factory=lambda: np.zeros(3, dtype=float))
    enable_external: bool = False


@dataclass
class TransitionConfig:
    """
    Configuration for electric dipole (E1) transitions.
    
    Driving field: E(t) = E₀ cos(ωt) in Cartesian components
    Interaction: H_int(t) = -q r·E(t) = +|e| r·E(t) for electron
    """
    state_i: int = 0  # Initial state index
    state_f: int = 1  # Final state index
    field_amplitude: float = 1.0  # E₀ magnitude
    field_polarization: ArrayR = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    field_frequency: Optional[float] = None  # ω; None = resonant
    
    # Computed quantities (set by DiracSolver)
    dipole_matrix_element_spherical: Optional[Dict[int, complex]] = None  # q -> <f|r_q|i>
    dipole_matrix_element: Optional[ArrayC] = None  # Cartesian components
    detuning: Optional[float] = None
    selection_rule_satisfied: Optional[bool] = None
    allowed_delta_m: Optional[List[int]] = None  # Which Δm are allowed given polarization


@dataclass
class HamiltonianConfig:
    """Configuration for the Dirac Hamiltonian and time evolution."""
    field: FieldConfig
    evolution_mode: Literal["stationary", "driven"] = "stationary"
    transition: Optional[TransitionConfig] = None
    include_rest_mass: bool = True
    include_coulomb: bool = True


@dataclass
class DiracState:
    """A single Dirac eigenstate (bound or free) with unique identifier."""
    spinor: ArrayC
    energy: float
    kind: Literal["bound", "free"]
    label: str
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    n: Optional[int] = None
    kappa: Optional[int] = None
    mj: Optional[float] = None
    momentum: Optional[ArrayR] = None
    spin_polarization: Optional[ArrayR] = None

    @property
    def large_components(self) -> ArrayC:
        return self.spinor[:2]

    @property
    def small_components(self) -> ArrayC:
        return self.spinor[2:]

    @property
    def density(self) -> ArrayR:
        return np.real(np.sum(np.abs(self.spinor) ** 2, axis=0))


# =============================================================================
# State Superposition Manager
# =============================================================================

class StateSuperposition:
    """Manages a superposition of Dirac states with complex coefficients."""
    
    def __init__(self) -> None:
        self.states: List[DiracState] = []
        self._coeffs: ArrayC = np.zeros(0, dtype=np.complex128)
        self._on_change_callback: Optional[callable] = None

    @property
    def coeffs(self) -> ArrayC:
        return self._coeffs

    def n_states(self) -> int:
        return len(self.states)

    def add_state(self, state: DiracState, amplitude: float = 1.0, phase: float = 0.0) -> int:
        self.states.append(state)
        c = amplitude * np.exp(1j * phase)
        self._coeffs = np.append(self._coeffs, c)
        if self._on_change_callback:
            self._on_change_callback()
        return len(self.states) - 1

    def remove_state(self, index: int) -> None:
        """Remove a state (caller must handle cache invalidation)."""
        if 0 <= index < len(self.states):
            del self.states[index]
            self._coeffs = np.delete(self._coeffs, index)
            if self._on_change_callback:
                self._on_change_callback()

    def set_coeff_polar(self, index: int, amplitude: float, phase: float) -> None:
        if 0 <= index < len(self.states):
            self._coeffs[index] = amplitude * np.exp(1j * phase)
            if self._on_change_callback:
                self._on_change_callback()

    def get_coeff_polar(self, index: int) -> Tuple[float, float]:
        if 0 <= index < len(self.states):
            c = self._coeffs[index]
            return float(np.abs(c)), float(np.angle(c))
        return 0.0, 0.0

    def normalize(self, eps: float = 1e-14) -> None:
        if self._coeffs.size == 0:
            return
        norm_sq = float(np.sum(np.abs(self._coeffs) ** 2))
        if norm_sq > eps:
            self._coeffs /= np.sqrt(norm_sq)
            if self._on_change_callback:
                self._on_change_callback()


# =============================================================================
# Physics Functions
# =============================================================================

def pauli_matrices() -> Tuple[ArrayC, ArrayC, ArrayC]:
    """Return the three Pauli spin matrices."""
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return sx, sy, sz


def dirac_alpha_beta() -> Tuple[ArrayC, ArrayC, ArrayC, ArrayC]:
    """Return the Dirac alpha and beta matrices in standard representation."""
    sx, sy, sz = pauli_matrices()
    zeros = np.zeros((2, 2), dtype=np.complex128)
    eye = np.eye(2, dtype=np.complex128)
    
    alpha_x = np.block([[zeros, sx], [sx, zeros]])
    alpha_y = np.block([[zeros, sy], [sy, zeros]])
    alpha_z = np.block([[zeros, sz], [sz, zeros]])
    beta = np.block([[eye, zeros], [zeros, -eye]])
    return alpha_x, alpha_y, alpha_z, beta


ALPHA_X, ALPHA_Y, ALPHA_Z, BETA = dirac_alpha_beta()


def kappa_to_l_j(kappa: int) -> Tuple[int, float]:
    """Convert Dirac quantum number κ to (l, j)."""
    if kappa == 0:
        raise ValueError("κ = 0 is not allowed")
    j = abs(kappa) - 0.5
    l = kappa if kappa > 0 else -kappa - 1
    return int(l), float(j)


def hydrogenic_dirac_energy(n: int, kappa: int, Z: int, include_rest_mass: bool = False) -> float:
    """
    Exact Dirac energy for hydrogenic bound state.
    
    E = mc² / sqrt(1 + (Zα)² / (n - |κ| + γ)²)
    where γ = sqrt(κ² - (Zα)²)
    
    In natural units with m=1:
    E_full = 1 / sqrt(1 + (Zα)² / (n - |κ| + γ)²)
    E_binding = E_full - 1
    """
    if n < 1 or kappa == 0:
        raise ValueError("Invalid quantum numbers")
    
    za = Z * ALPHA_FS
    if za >= abs(kappa):
        raise ValueError(f"Zα = {za} ≥ |κ| = {abs(kappa)}: no bound state")
    
    gamma = np.sqrt(kappa**2 - za**2)
    denom = n - abs(kappa) + gamma
    E_full = 1.0 / np.sqrt(1.0 + za**2 / denom**2)
    
    return float(E_full if include_rest_mass else E_full - 1.0)


def spinor_spherical_harmonic(kappa: int, mj: float, theta: ArrayR, phi: ArrayR) -> ArrayC:
    """
    Compute spinor spherical harmonic Ω_{κ,mj}(θ,φ).
    
    Returns a 2-component spinor field using Clebsch-Gordan coefficients.
    """
    l, j = kappa_to_l_j(kappa)
    m = float(mj)
    
    if abs(m) > j + 1e-9:
        raise ValueError(f"|mj| must be ≤ j, got mj={mj}, j={j}")
    
    m_l_up = int(np.round(m - 0.5))
    m_l_dn = int(np.round(m + 0.5))
    
    # Compute spherical harmonics only for valid m_l values
    Y_up = sph_harm(m_l_up, l, phi, theta) if -l <= m_l_up <= l else np.zeros_like(theta, dtype=np.complex128)
    Y_dn = sph_harm(m_l_dn, l, phi, theta) if -l <= m_l_dn <= l else np.zeros_like(theta, dtype=np.complex128)
    
    # Clebsch-Gordan coefficients
    if kappa < 0:  # j = l + 1/2
        denom = 2.0 * j
        c_up = np.sqrt(max(j + m, 0.0) / denom)
        c_dn = np.sqrt(max(j - m, 0.0) / denom)
        comp_up = c_up * Y_up
        comp_dn = c_dn * Y_dn
    else:  # j = l - 1/2
        denom = 2.0 * (j + 1.0)
        c_up = np.sqrt(max(j - m + 1.0, 0.0) / denom)
        c_dn = np.sqrt(max(j + m + 1.0, 0.0) / denom)
        comp_up = -c_up * Y_up
        comp_dn = c_dn * Y_dn
    
    spinor = np.zeros((2,) + theta.shape, dtype=np.complex128)
    spinor[0] = comp_up
    spinor[1] = comp_dn
    return spinor


def hydrogenic_dirac_radial(n: int, kappa: int, r: ArrayR, Z: int) -> Tuple[ArrayR, ArrayR]:
    """
    Compute hydrogenic Dirac radial functions G(r) and F(r).
    
    The full 4-spinor is: Ψ = (1/r)[G(r)Ω_κ, iF(r)Ω_{-κ}]
    
    Normalization: ∫₀^∞ (G² + F²) dr = 1
    """
    if n < 1 or kappa == 0 or Z <= 0:
        return np.zeros_like(r), np.zeros_like(r)
    
    za = Z * ALPHA_FS
    abs_k = abs(kappa)
    l, j = kappa_to_l_j(kappa)
    
    if za >= abs_k:
        raise ValueError(f"Zα = {za} ≥ |κ| = {abs_k}")
    
    # Non-relativistic radial quantum number
    n_r = n - l - 1
    
    if n_r < 0:
        raise ValueError(f"n_r = n - l - 1 = {n} - {l} - 1 < 0")
    
    # FIX DIRAC-002: Corrected Z-scaling
    # Use a0 = 1/α (Bohr radius for Z=1 in natural units)
    # Then rho = 2*Z*r/(n*a0) = 2*Z*α*r/n
    # This avoids double-counting Z that occurred with a0 = 1/(Z*α)
    a0 = 1.0 / ALPHA_FS  # Bohr radius for Z=1
    
    # Scaled radial coordinate
    rho = 2.0 * Z * r / (n * a0)
    
    # Non-relativistic radial function normalization
    # N_nl = sqrt((2Z/(n*a0))^3 * (n-l-1)! / (2n*(n+l)!))
    N_nl = np.sqrt((2*Z/(n*a0))**3 * math_factorial(n-l-1) / (2*n*math_factorial(n+l)))
    
    # Laguerre polynomial
    if n_r == 0:
        L = np.ones_like(rho)
    else:
        L = eval_genlaguerre(n_r, 2*l+1, rho)
    
    # Power term - safe calculation
    rho_power = np.zeros_like(rho)
    mask = rho > 0
    if l > 0:
        rho_power[mask] = np.power(rho[mask], l)
    else:
        rho_power = np.ones_like(rho)
    
    R_nl = N_nl * np.exp(-rho/2.0) * rho_power * L
    
    # Energy
    E = hydrogenic_dirac_energy(n, kappa, Z, include_rest_mass=True)
    
    # Dirac components
    G = r * R_nl
    # FIX: Use za (= Z*α) directly, not Z*ALPHA_FS again since za already defined
    F = za * r * R_nl * (kappa / abs_k) * np.sqrt((1.0 - E)/(1.0 + E))
    
    return np.asarray(G, dtype=float), np.asarray(F, dtype=float)


def check_dirac_radial_equations(n: int, kappa: int, Z: int, 
                                  n_points: int = 1000, 
                                  r_max_factor: float = 10.0) -> Dict[str, float]:
    """
    Validate radial Dirac equations by computing residuals on a fine grid.
    """
    r_char = n**2 / (Z * ALPHA_FS)
    r_max = r_max_factor * r_char
    r_grid = np.logspace(np.log10(1e-4 * r_char), np.log10(r_max), n_points)
    
    G, F = hydrogenic_dirac_radial(n, kappa, r_grid, Z)
    
    dG_dr = np.gradient(G, r_grid, edge_order=2)
    dF_dr = np.gradient(F, r_grid, edge_order=2)
    
    E = hydrogenic_dirac_energy(n, kappa, Z, include_rest_mass=True)
    V = -Z * ALPHA_FS / r_grid
    
    residual_G = dG_dr + (kappa / r_grid) * G - (E - V + 1.0) * F
    residual_F = dF_dr - (E - V - 1.0) * G - (kappa / r_grid) * F
    
    return {
        "max_residual_G": float(np.max(np.abs(residual_G))),
        "max_residual_F": float(np.max(np.abs(residual_F))),
        "max_residual_total": max(float(np.max(np.abs(residual_G))), float(np.max(np.abs(residual_F)))),
        "n": n, "kappa": kappa, "Z": Z
    }


def cartesian_to_spherical_components(E_xyz: ArrayR) -> Dict[int, complex]:
    """
    Convert Cartesian polarization vector to spherical tensor components.
    """
    Ex, Ey, Ez = E_xyz[0], E_xyz[1], E_xyz[2]
    
    E_plus1 = -(Ex + 1j*Ey) / np.sqrt(2.0)
    E_0 = Ez
    E_minus1 = (Ex - 1j*Ey) / np.sqrt(2.0)
    
    return {-1: complex(E_minus1), 0: complex(E_0), 1: complex(E_plus1)}


def check_e1_selection_rules(state_i: DiracState, state_f: DiracState,
                             polarization: Optional[ArrayR] = None) -> Tuple[bool, str, List[int]]:
    """
    Check if E1 transition is allowed: Δl=±1, Δj=0,±1, Δmj constrained by polarization.
    """
    if state_i.kind != "bound" or state_f.kind != "bound":
        return True, "Selection rules only checked for bound states", list(range(-1, 2))
    
    if state_i.kappa is None or state_f.kappa is None:
        return True, "Quantum numbers not available", list(range(-1, 2))
    
    l_i, j_i = kappa_to_l_j(state_i.kappa)
    l_f, j_f = kappa_to_l_j(state_f.kappa)
    mj_i = state_i.mj if state_i.mj is not None else 0.0
    mj_f = state_f.mj if state_f.mj is not None else 0.0
    
    violations = []
    
    # Parity / Δl rule
    if abs(l_f - l_i) != 1:
        violations.append(f"Δl={l_f-l_i} (must be ±1)")
    
    # Δj rule
    if abs(j_f - j_i) > 1:
        violations.append(f"Δj={j_f-j_i} (must be 0,±1)")
    if j_i == 0 and j_f == 0:
        violations.append("j=0→j=0 forbidden")
    
    # Δmj rule depends on polarization
    # FIX DIRAC-003: Convert Δm to int via rounding to avoid float precision issues
    # mj values are half-integers, so their difference should be an integer
    delta_m_float = mj_f - mj_i
    delta_m = int(round(delta_m_float))
    allowed_delta_m = []
    
    if polarization is not None:
        E_sph = cartesian_to_spherical_components(polarization)
        for q in [-1, 0, 1]:
            if abs(E_sph[q]) > 1e-12:
                allowed_delta_m.append(q)
    else:
        allowed_delta_m = [-1, 0, 1]
    
    if delta_m not in allowed_delta_m:
        violations.append(f"Δmj={delta_m} not allowed for given polarization")
    
    if violations:
        return False, "; ".join(violations), allowed_delta_m
    return True, "E1 allowed", allowed_delta_m


# =============================================================================
# Main Solver Class
# =============================================================================

class DiracSolver:
    """
    Solver for visualizing analytic hydrogenic Dirac wavefunctions and time evolution.
    
    PERFORMANCE OPTIMIZATIONS:
    - Cached spinor stack for repeated superposition computations
    - Pre-allocated output arrays
    - Numba JIT-compiled density and superposition calculations
    - Parallel computation via thread pools
    """
    
    def __init__(
        self,
        grid_config: Optional[DiracGridConfig] = None,
        field_config: Optional[FieldConfig] = None,
        include_rest_mass: bool = False,
    ) -> None:
        self.grid = grid_config or DiracGridConfig()
        self.field = field_config or FieldConfig()
        self.hamiltonian = HamiltonianConfig(
            field=self.field,
            evolution_mode="stationary",
            include_rest_mass=include_rest_mass,
            include_coulomb=self.field.enable_coulomb,
        )
        self.mass = 1.0
        self.superposition = StateSuperposition()
        self._time = 0.0
        
        # Current coefficients (updated incrementally for performance)
        self._c_current: Optional[ArrayC] = None
        
        # Connect callback to invalidate cached coefficients
        self.superposition._on_change_callback = self._invalidate_current_coeffs
        
        # Caching
        self._dipole_cache: Dict[Tuple[str, str, str, str], Any] = {}
        self._sph_harm_cache: Dict[Tuple[int, int, str], ArrayC] = {}
        self._grid_id = str(uuid.uuid4())
        
        # OPTIMIZATION: Pre-allocated arrays and cached spinor stack
        self._spinor_stack_real: Optional[ArrayR] = None
        self._spinor_stack_imag: Optional[ArrayR] = None
        self._spinor_stack_valid = False
        self._output_real: Optional[ArrayR] = None
        self._output_imag: Optional[ArrayR] = None
        
        self._init_grid()
    
    def _invalidate_current_coeffs(self) -> None:
        """Callback when superposition changes - reset caches."""
        self._c_current = None
        self._time = 0.0
        self._spinor_stack_valid = False  # Invalidate spinor stack cache

    def _init_grid(self) -> None:
        """Initialize coordinate grids for rendering."""
        self.X, self.Y, self.Z = self.grid.coordinate_grids()
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        # Spherical coordinates
        self.THETA = np.zeros_like(self.R)
        self.PHI = np.zeros_like(self.R)
        mask = self.R > 1e-12
        self.THETA[mask] = np.arccos(np.clip(self.Z[mask] / self.R[mask], -1, 1))
        self.PHI[mask] = np.arctan2(self.Y[mask], self.X[mask])
        
        # Pre-allocate output arrays
        shape = self.grid.shape()
        self._output_real = np.zeros((4,) + shape, dtype=np.float64)
        self._output_imag = np.zeros((4,) + shape, dtype=np.float64)

    def _volume_element(self) -> float:
        dx = (self.grid.x_max - self.grid.x_min) / max(self.grid.nx - 1, 1)
        dy = (self.grid.y_max - self.grid.y_min) / max(self.grid.ny - 1, 1)
        dz = (self.grid.z_max - self.grid.z_min) / max(self.grid.nz - 1, 1)
        return float(dx * dy * dz)

    def _update_spinor_stack(self) -> None:
        """
        Update cached spinor stack for fast superposition computation.
        
        Called lazily when spinor stack is needed and invalid.
        """
        if self._spinor_stack_valid and self._spinor_stack_real is not None:
            return
        
        n_states = self.superposition.n_states()
        if n_states == 0:
            self._spinor_stack_real = None
            self._spinor_stack_imag = None
            return
        
        shape = self.grid.shape()
        
        # Pre-allocate contiguous arrays
        self._spinor_stack_real = np.zeros((n_states, 4) + shape, dtype=np.float64)
        self._spinor_stack_imag = np.zeros((n_states, 4) + shape, dtype=np.float64)
        
        # Copy spinor data
        for i, st in enumerate(self.superposition.states):
            self._spinor_stack_real[i] = np.real(st.spinor)
            self._spinor_stack_imag[i] = np.imag(st.spinor)
        
        self._spinor_stack_valid = True

    # -------------------------------------------------------------------------
    # Grid Management
    # -------------------------------------------------------------------------
    
    def update_grid(self, new_grid: DiracGridConfig) -> None:
        self.grid = new_grid
        self._grid_id = str(uuid.uuid4())
        self._init_grid()
        self._dipole_cache.clear()
        self._sph_harm_cache.clear()
        self._spinor_stack_valid = False
        self._rebuild_states()
        # FIX NEW-002: Refresh transition if in driven mode
        self._refresh_transition_if_driven()

    def ensure_grid_for_bound_states(self, safety_factor: float = 5.0) -> None:
        """Expand grid to contain all bound states if needed."""
        max_n = max((st.n or 0 for st in self.superposition.states if st.kind == "bound"), default=0)
        if max_n <= 0 or not self.field.enable_coulomb or self.field.Z <= 0:
            return
        
        r_char = (max_n**2) / (self.field.Z * ALPHA_FS)
        half_target = safety_factor * r_char
        
        current_half = max(abs(self.grid.x_min), abs(self.grid.x_max),
                          abs(self.grid.y_min), abs(self.grid.y_max),
                          abs(self.grid.z_min), abs(self.grid.z_max))
        
        if current_half < half_target:
            logger.info(f"Expanding grid to {half_target:.1f} (was {current_half:.1f})")
            self.update_grid(DiracGridConfig(
                nx=self.grid.nx, ny=self.grid.ny, nz=self.grid.nz,
                x_range=(-half_target, half_target),
                y_range=(-half_target, half_target),
                z_range=(-half_target, half_target),
            ))

    def _get_cached_sph_harm(self, m: int, l: int) -> ArrayC:
        """Get spherical harmonic from cache or compute."""
        cache_key = (m, l, self._grid_id)
        
        if cache_key not in self._sph_harm_cache:
            Y = sph_harm(m, l, self.PHI, self.THETA)
            self._sph_harm_cache[cache_key] = Y
        
        return self._sph_harm_cache[cache_key]
    
    def _spinor_spherical_harmonic_cached(self, kappa: int, mj: float) -> ArrayC:
        """Optimized spinor spherical harmonic using cached Y_l^m."""
        l, j = kappa_to_l_j(kappa)
        m = float(mj)
        
        if abs(m) > j + 1e-9:
            raise ValueError(f"|mj| must be ≤ j, got mj={mj}, j={j}")
        
        m_l_up = int(np.round(m - 0.5))
        m_l_dn = int(np.round(m + 0.5))
        
        if -l <= m_l_up <= l:
            Y_up = self._get_cached_sph_harm(m_l_up, l)
        else:
            Y_up = np.zeros_like(self.THETA, dtype=np.complex128)
            
        if -l <= m_l_dn <= l:
            Y_dn = self._get_cached_sph_harm(m_l_dn, l)
        else:
            Y_dn = np.zeros_like(self.THETA, dtype=np.complex128)
        
        if kappa < 0:  # j = l + 1/2
            denom = 2.0 * j
            c_up = np.sqrt(max(j + m, 0.0) / denom)
            c_dn = np.sqrt(max(j - m, 0.0) / denom)
            comp_up = c_up * Y_up
            comp_dn = c_dn * Y_dn
        else:  # j = l - 1/2
            denom = 2.0 * (j + 1.0)
            c_up = np.sqrt(max(j - m + 1.0, 0.0) / denom)
            c_dn = np.sqrt(max(j + m + 1.0, 0.0) / denom)
            comp_up = -c_up * Y_up
            comp_dn = c_dn * Y_dn
        
        spinor = np.zeros((2,) + self.THETA.shape, dtype=np.complex128)
        spinor[0] = comp_up
        spinor[1] = comp_dn
        return spinor

    # -------------------------------------------------------------------------
    # Field Configuration
    # -------------------------------------------------------------------------
    
    def set_nuclear_charge(self, Z: int) -> None:
        if Z <= 0:
            raise ValueError("Z must be positive")
        self.field.Z = Z
        self.field.enable_coulomb = True
        self.hamiltonian.include_coulomb = True
        self._dipole_cache.clear()
        self._spinor_stack_valid = False
        self._rebuild_states()
        # FIX NEW-002: Refresh transition if in driven mode
        self._refresh_transition_if_driven()

    def update_field(self, new_field: FieldConfig) -> None:
        """Update field configuration with proper cache invalidation."""
        old_Z = self.field.Z
        self.field = new_field
        self.hamiltonian.field = new_field
        # FIX BUG-004: Sync include_coulomb and invalidate caches
        self.hamiltonian.include_coulomb = new_field.enable_coulomb
        self._dipole_cache.clear()
        self._spinor_stack_valid = False
        
        # FIX NEW-003: If Z changed, rebuild bound states
        if new_field.Z != old_Z and new_field.enable_coulomb:
            self._rebuild_states()
            self._refresh_transition_if_driven()

    # -------------------------------------------------------------------------
    # Evolution Mode
    # -------------------------------------------------------------------------
    
    def set_evolution_mode(
        self,
        mode: Literal["stationary", "driven"],
        transition_config: Optional[TransitionConfig] = None,
    ) -> None:
        """Set time evolution mode."""
        self.hamiltonian.evolution_mode = mode
        
        if mode == "driven":
            if transition_config is None:
                transition_config = TransitionConfig(state_i=0, state_f=1)
            self.hamiltonian.transition = transition_config
            self._setup_transition()
        else:
            # FIX BUG-003: Clear transition when switching to stationary
            self.hamiltonian.transition = None

    def _refresh_transition_if_driven(self) -> None:
        """FIX NEW-002: Recompute transition parameters if in driven mode after state rebuild."""
        if self.hamiltonian.evolution_mode == "driven" and self.hamiltonian.transition is not None:
            trans = self.hamiltonian.transition
            n_states = self.superposition.n_states()
            # Validate indices still valid
            if trans.state_i < n_states and trans.state_f < n_states:
                self._setup_transition()
            else:
                # Indices invalid after rebuild, switch to stationary
                logger.warning("Transition state indices invalid after rebuild, switching to stationary")
                self.set_evolution_mode("stationary")

    def _setup_transition(self) -> None:
        """Compute dipole matrix elements and setup transition parameters."""
        trans = self.hamiltonian.transition
        if trans is None:
            return
        
        n_states = self.superposition.n_states()
        if trans.state_i >= n_states or trans.state_f >= n_states:
            logger.warning("Transition indices out of range")
            return
        
        state_i = self.superposition.states[trans.state_i]
        state_f = self.superposition.states[trans.state_f]
        
        allowed, reason, allowed_dm = check_e1_selection_rules(
            state_i, state_f, trans.field_polarization
        )
        trans.selection_rule_satisfied = allowed
        trans.allowed_delta_m = allowed_dm
        
        if not allowed:
            logger.warning(f"E1 selection rules violated: {reason}")
        
        dipole_sph, dipole_cart = self._compute_dipole_element_accurate(state_i, state_f)
        trans.dipole_matrix_element_spherical = dipole_sph
        trans.dipole_matrix_element = dipole_cart
        
        omega_0 = abs(state_f.energy - state_i.energy)
        if trans.field_frequency is None:
            trans.field_frequency = omega_0
            trans.detuning = 0.0
        else:
            trans.detuning = trans.field_frequency - omega_0
        
        logger.info(f"Transition setup: ω={trans.field_frequency:.4e}, Δ={trans.detuning:.4e}")

    def _compute_dipole_element_accurate(
        self, state_i: DiracState, state_f: DiracState
    ) -> Tuple[Dict[int, complex], ArrayC]:
        """Compute dipole matrix element using accurate 1D radial integrals."""
        cache_key = (state_i.uid, state_f.uid, "dipole", self._grid_id)
        if cache_key in self._dipole_cache:
            return self._dipole_cache[cache_key]
        
        if state_i.kind == "bound" and state_f.kind == "bound":
            dipole_sph, dipole_cart = self._dipole_radial_integral(state_i, state_f)
        else:
            dipole_cart = self._dipole_grid_integral(state_i, state_f)
            dipole_sph = {-1: 0.0, 0: 0.0, 1: 0.0}
        
        self._dipole_cache[cache_key] = (dipole_sph, dipole_cart)
        return dipole_sph, dipole_cart

    def _dipole_radial_integral(
        self, state_i: DiracState, state_f: DiracState
    ) -> Tuple[Dict[int, complex], ArrayC]:
        """Compute <f|r|i> via separation into angular and radial parts."""
        n_i, k_i, mj_i = state_i.n, state_i.kappa, state_i.mj
        n_f, k_f, mj_f = state_f.n, state_f.kappa, state_f.mj
        Z = self.field.Z
        
        if None in [n_i, k_i, mj_i, n_f, k_f, mj_f]:
            return {-1: 0.0, 0: 0.0, 1: 0.0}, self._dipole_grid_integral(state_i, state_f)
        
        l_i, j_i = kappa_to_l_j(k_i)
        l_f, j_f = kappa_to_l_j(k_f)
        
        if abs(l_f - l_i) != 1:
            return {-1: 0.0, 0: 0.0, 1: 0.0}, np.zeros(3, dtype=np.complex128)
        
        # FIX DIRAC-003: Use int(round()) for comparison to avoid float precision issues
        delta_m = int(round(mj_f - mj_i))
        
        dipole_sph = {}
        for q in [-1, 0, 1]:
            if delta_m != q:
                dipole_sph[q] = 0.0
            else:
                R_val = self._radial_dipole_integral(n_i, k_i, n_f, k_f, Z)
                ang_coeff = self._angular_dipole_coefficient(k_i, mj_i, k_f, mj_f, q)
                dipole_sph[q] = complex(R_val * ang_coeff)
        
        d_x = -(dipole_sph[1] + dipole_sph[-1]) / np.sqrt(2.0)
        d_y = 1j * (dipole_sph[1] - dipole_sph[-1]) / np.sqrt(2.0)
        d_z = dipole_sph[0]
        
        dipole_cart = np.array([d_x, d_y, d_z], dtype=np.complex128)
        return dipole_sph, dipole_cart

    def _radial_dipole_integral(self, n_i: int, k_i: int, n_f: int, k_f: int, Z: int) -> float:
        """Compute radial integral: ∫₀^∞ r [G_f G_i + F_f F_i] dr"""
        r_char = max(n_i, n_f)**2 / (Z * ALPHA_FS)
        r_max = 20.0 * r_char
        
        def integrand(r: float) -> float:
            if r < 1e-15:
                return 0.0
            G_i, F_i = hydrogenic_dirac_radial(n_i, k_i, np.array([r]), Z)
            G_f, F_f = hydrogenic_dirac_radial(n_f, k_f, np.array([r]), Z)
            return r * (G_f[0] * G_i[0] + F_f[0] * F_i[0])
        
        result, error = quad(integrand, 0, r_max, epsabs=1e-10, epsrel=1e-8, limit=200)
        return float(result)

    def _angular_dipole_coefficient(self, k_i: int, mj_i: float, k_f: int, mj_f: float, q: int) -> float:
        """Angular coefficient for E1 matrix element."""
        # FIX DIRAC-003: Use int(round()) for comparison
        delta_m = int(round(mj_f - mj_i))
        if delta_m != q:
            return 0.0
        
        l_i, j_i = kappa_to_l_j(k_i)
        l_f, j_f = kappa_to_l_j(k_f)
        
        return 1.0 / np.sqrt(2*l_f + 1)

    def _dipole_grid_integral(self, state_i: DiracState, state_f: DiracState) -> ArrayC:
        """Fallback: compute dipole via 3D grid integration."""
        dv = self._volume_element()
        overlap = np.sum(np.conjugate(state_f.spinor) * state_i.spinor, axis=0)
        
        d_x = np.sum(self.X * overlap) * dv
        d_y = np.sum(self.Y * overlap) * dv
        d_z = np.sum(self.Z * overlap) * dv
        
        return np.array([d_x, d_y, d_z], dtype=np.complex128)

    def get_transition_info(self) -> Optional[Dict[str, Any]]:
        """Get current transition parameters."""
        if self.hamiltonian.evolution_mode != "driven":
            return None
        
        trans = self.hamiltonian.transition
        if trans is None:
            return None
        
        return {
            "state_i": trans.state_i,
            "state_f": trans.state_f,
            "detuning": trans.detuning,
            "field_frequency": trans.field_frequency,
            "dipole_magnitude": (
                float(np.linalg.norm(trans.dipole_matrix_element))
                if trans.dipole_matrix_element is not None else None
            ),
            "dipole_spherical": trans.dipole_matrix_element_spherical,
            "selection_rules_satisfied": trans.selection_rule_satisfied,
            "allowed_delta_m": trans.allowed_delta_m,
            "field_amplitude": trans.field_amplitude,
        }

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------
    
    def add_bound_state(self, n: int, kappa: int, mj: float, 
                       amplitude: float = 1.0, phase: float = 0.0) -> int:
        """Add a hydrogenic bound state."""
        self._validate_bound_qn(n, kappa, mj)
        
        if not self.field.enable_coulomb or self.field.Z <= 0:
            raise ValueError("Coulomb field required for bound states")
        
        state = self._build_bound_state(n, kappa, mj)
        self._dipole_cache.clear()
        self._spinor_stack_valid = False
        return self.superposition.add_state(state, amplitude, phase)

    def remove_state(self, index: int) -> None:
        """Remove a state and invalidate caches."""
        if 0 <= index < self.superposition.n_states():
            self.superposition.remove_state(index)
            self._dipole_cache.clear()
            self._spinor_stack_valid = False
            
            if self.hamiltonian.transition:
                trans = self.hamiltonian.transition
                if trans.state_i == index or trans.state_f == index:
                    self.hamiltonian.evolution_mode = "stationary"
                    self.hamiltonian.transition = None
                else:
                    if trans.state_i > index:
                        trans.state_i -= 1
                    if trans.state_f > index:
                        trans.state_f -= 1

    def add_free_state(self, momentum: ArrayR, spin_polarization: Optional[ArrayR] = None,
                      amplitude: float = 1.0, phase: float = 0.0, 
                      positive_energy: bool = True) -> int:
        """Add a free particle state."""
        state = self._build_free_state(momentum, spin_polarization, positive_energy)
        self._dipole_cache.clear()
        self._spinor_stack_valid = False
        return self.superposition.add_state(state, amplitude, phase)

    def _validate_bound_qn(self, n: int, kappa: int, mj: float) -> None:
        if n < 1:
            raise ValueError("n must be ≥ 1")
        if kappa == 0:
            raise ValueError("κ cannot be 0")
        l, j = kappa_to_l_j(kappa)
        if n <= l:
            raise ValueError(f"n must be > l={l} for κ={kappa}")
        if abs(mj) > j + 1e-8:
            raise ValueError(f"|mj| must be ≤ j={j}")
        # FIX CORE-NEW-002: Validate mj is a half-integer (2*mj must be an integer)
        twice_mj = 2 * mj
        if abs(twice_mj - round(twice_mj)) > 1e-8:
            raise ValueError(f"mj must be a half-integer, got {mj}")

    def _build_bound_state(self, n: int, kappa: int, mj: float) -> DiracState:
        """Build a bound state with correct radial functions and normalization."""
        Z = self.field.Z
        l, j = kappa_to_l_j(kappa)
        
        G_r, F_r = hydrogenic_dirac_radial(n, kappa, self.R, Z)
        
        Omega_k = self._spinor_spherical_harmonic_cached(kappa, mj)
        Omega_mk = self._spinor_spherical_harmonic_cached(-kappa, mj)
        
        r_safe = np.where(self.R > 1e-12, self.R, 1e-12)
        radial_big = G_r / r_safe
        radial_small = F_r / r_safe
        
        spinor = np.zeros((4,) + self.grid.shape(), dtype=np.complex128)
        spinor[0] = radial_big * Omega_k[0]
        spinor[1] = radial_big * Omega_k[1]
        spinor[2] = 1j * radial_small * Omega_mk[0]
        spinor[3] = 1j * radial_small * Omega_mk[1]
        
        spinor = self._normalize_spinor(spinor)
        energy = hydrogenic_dirac_energy(n, kappa, Z, self.hamiltonian.include_rest_mass)
        
        return DiracState(
            spinor=spinor, energy=energy, kind="bound",
            label=f"n={n}, κ={kappa} (l={l}, j={j:.1f}), mj={mj:+.1f}",
            n=n, kappa=kappa, mj=mj
        )

    def _build_free_state(self, momentum: ArrayR, spin_pol: Optional[ArrayR],
                         positive_energy: bool) -> DiracState:
        """Build a free particle state."""
        p_vec = np.asarray(momentum, dtype=float)
        if p_vec.shape != (3,):
            raise ValueError("momentum must be 3-vector")
        
        if spin_pol is None:
            s_hat = np.array([0.0, 0.0, 1.0])
        else:
            s_hat = np.asarray(spin_pol, dtype=float)
            norm = np.linalg.norm(s_hat)
            s_hat = s_hat / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        
        theta_s = np.arccos(np.clip(s_hat[2], -1.0, 1.0))
        phi_s = np.arctan2(s_hat[1], s_hat[0])
        chi = np.array([np.cos(theta_s/2), np.exp(1j*phi_s)*np.sin(theta_s/2)], dtype=np.complex128)
        
        p = np.linalg.norm(p_vec)
        E_full = np.sqrt(self.mass**2 + p**2)
        sx, sy, sz = pauli_matrices()
        sigma_dot_p = p_vec[0]*sx + p_vec[1]*sy + p_vec[2]*sz
        
        norm_factor = np.sqrt(E_full + self.mass)
        if positive_energy:
            upper = norm_factor * chi
            lower = (sigma_dot_p @ chi) / norm_factor
        else:
            upper = -(sigma_dot_p @ chi) / norm_factor
            lower = norm_factor * chi
        
        phase = p_vec[0]*self.X + p_vec[1]*self.Y + p_vec[2]*self.Z
        plane = np.exp(1j * phase)
        
        spinor = np.zeros((4,) + self.grid.shape(), dtype=np.complex128)
        spinor[0] = plane * upper[0]
        spinor[1] = plane * upper[1]
        spinor[2] = plane * lower[0]
        spinor[3] = plane * lower[1]
        spinor = self._normalize_spinor(spinor)
        
        energy = E_full if positive_energy else -E_full
        if not self.hamiltonian.include_rest_mass:
            energy = (E_full - self.mass) if positive_energy else -(E_full - self.mass)
        
        sign = '+' if positive_energy else '-'
        return DiracState(
            spinor=spinor, energy=energy, kind="free",
            label=f"free |p|={p:.3f}, E={sign}{abs(energy):.3f}",
            momentum=p_vec.copy(), spin_polarization=s_hat.copy()
        )

    def _normalize_spinor(self, spinor: ArrayC, eps: float = 1e-14) -> ArrayC:
        """Normalize spinor on the grid."""
        dv = self._volume_element()
        norm_sq = float(np.sum(np.abs(spinor)**2) * dv)
        if norm_sq > eps:
            spinor /= np.sqrt(norm_sq)
        return spinor

    def _rebuild_states(self) -> None:
        """Rebuild all states on current grid."""
        if not self.superposition.states:
            return
        
        old_states = list(self.superposition.states)
        old_coeffs = self.superposition.coeffs.copy()
        
        self.superposition.states = []
        self.superposition._coeffs = np.zeros(0, dtype=np.complex128)
        
        for coeff, st in zip(old_coeffs, old_states):
            if abs(coeff) < 1e-14:
                continue
            amp, ph = float(np.abs(coeff)), float(np.angle(coeff))
            
            if st.kind == "bound" and st.n and st.kappa:
                mj_val = st.mj if st.mj is not None else 0.5
                self.add_bound_state(st.n, st.kappa, mj_val, amp, ph)
            elif st.kind == "free" and st.momentum is not None:
                self.add_free_state(st.momentum, st.spin_polarization, amp, ph, st.energy >= 0)
        
        self._spinor_stack_valid = False

    # -------------------------------------------------------------------------
    # Time Evolution (General ODE Integration)
    # -------------------------------------------------------------------------
    
    @property
    def time(self) -> float:
        return self._time

    def reset_time(self) -> None:
        self._time = 0.0
        self._c_current = None

    def step(self, dt: float) -> float:
        """Advance time by dt using INCREMENTAL evolution."""
        if dt == 0:
            return self._time
        
        if self._c_current is None:
            self._c_current = self.superposition.coeffs.copy()
        
        if not self.superposition.states:
            self._time += dt
            return self._time
        
        energies = np.array([st.energy for st in self.superposition.states])
        mode = self.hamiltonian.evolution_mode
        
        if mode == "stationary":
            self._c_current *= np.exp(-1j * energies * dt)
        elif mode == "driven" and self.hamiltonian.transition is not None:
            self._c_current = self._evolve_driven_incremental(
                self._c_current, energies, self._time, dt
            )
        else:
            self._c_current *= np.exp(-1j * energies * dt)
        
        self._time += dt
        return self._time
    
    def get_current_coefficients(self) -> ArrayC:
        """Get current coefficients (fast access, no recomputation)."""
        if self._c_current is None:
            return self.superposition.coeffs.copy()
        return self._c_current.copy()

    def _coefficients_at_time(self, t: float) -> ArrayC:
        """Get superposition coefficients at time t."""
        if not self.superposition.states:
            return np.zeros(0, dtype=np.complex128)
        
        c0 = self.superposition.coeffs
        energies = np.array([st.energy for st in self.superposition.states])
        mode = self.hamiltonian.evolution_mode
        
        if mode == "stationary" or t == 0.0:
            return c0 * np.exp(-1j * energies * t)
        
        if mode == "driven" and self.hamiltonian.transition is not None:
            return self._evolve_driven(c0, energies, t)
        
        return c0 * np.exp(-1j * energies * t)

    def _evolve_driven_incremental(
        self, c_current: ArrayC, energies: ArrayR, t_current: float, dt: float
    ) -> ArrayC:
        """Evolve driven system INCREMENTALLY from t_current to t_current + dt."""
        trans = self.hamiltonian.transition
        if trans is None or trans.dipole_matrix_element is None:
            return c_current * np.exp(-1j * energies * dt)
        
        n_states = len(c_current)
        i, f = trans.state_i, trans.state_f
        
        if i >= n_states or f >= n_states:
            return c_current * np.exp(-1j * energies * dt)
        
        E0 = trans.field_amplitude
        pol = trans.field_polarization
        omega = trans.field_frequency or abs(energies[f] - energies[i])
        dipole = trans.dipole_matrix_element
        V_if = E_CHARGE * np.dot(dipole, E0 * pol)
        
        if np.abs(V_if) < 1e-15:
            return c_current * np.exp(-1j * energies * dt)
        
        def rhs(t_val: float, c: ArrayC) -> ArrayC:
            dcdt = -1j * energies * c
            field_t = E0 * pol * np.cos(omega * t_val)
            coupling = E_CHARGE * np.dot(dipole, field_t)
            dcdt[i] += -1j * coupling * c[f]
            dcdt[f] += -1j * np.conj(coupling) * c[i]
            return dcdt
        
        if dt < 1e-12:
            return c_current
        
        sol = solve_ivp(
            lambda t_val, c: rhs(t_val, c),
            (t_current, t_current + dt),
            c_current,
            method='DOP853',
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            logger.warning(f"ODE integration failed: {sol.message}")
            return c_current * np.exp(-1j * energies * dt)

    def _evolve_driven(self, c0: ArrayC, energies: ArrayR, t: float) -> ArrayC:
        """Evolve driven system via ODE integration."""
        trans = self.hamiltonian.transition
        if trans is None or trans.dipole_matrix_element is None:
            return c0 * np.exp(-1j * energies * t)
        
        n_states = len(c0)
        i, f = trans.state_i, trans.state_f
        
        if i >= n_states or f >= n_states:
            return c0 * np.exp(-1j * energies * t)
        
        E0 = trans.field_amplitude
        pol = trans.field_polarization
        omega = trans.field_frequency or abs(energies[f] - energies[i])
        
        dipole = trans.dipole_matrix_element
        V_if = E_CHARGE * np.dot(dipole, E0 * pol)
        
        def rhs(t_val: float, c: ArrayC) -> ArrayC:
            dcdt = -1j * energies * c
            field_t = E0 * pol * np.cos(omega * t_val)
            if np.abs(V_if) > 1e-15:
                coupling = E_CHARGE * np.dot(dipole, field_t)
                dcdt[i] += -1j * coupling * c[f]
                dcdt[f] += -1j * np.conj(coupling) * c[i]
            return dcdt
        
        if t < 1e-12:
            return c0
        
        sol = solve_ivp(
            lambda t_val, c: rhs(t_val, c),
            (0, t),
            c0,
            method='DOP853',
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            logger.warning(f"ODE integration failed: {sol.message}")
            return c0 * np.exp(-1j * energies * t)

    # -------------------------------------------------------------------------
    # Observables (OPTIMIZED)
    # -------------------------------------------------------------------------
    
    def total_spinor(self, t: Optional[float] = None) -> ArrayC:
        """Get total spinor wavefunction at time t."""
        if not self.superposition.states:
            return np.zeros((4,) + self.grid.shape(), dtype=np.complex128)
        
        t = t if t is not None else self._time
        coeffs = self._coefficients_at_time(t)
        
        # Use optimized computation
        return self._compute_superposition_optimized(coeffs)
    
    def total_spinor_current(self) -> ArrayC:
        """
        Get total spinor at CURRENT time using CURRENT coefficients.
        
        OPTIMIZED: Uses Numba JIT compilation when available.
        """
        if not self.superposition.states:
            return np.zeros((4,) + self.grid.shape(), dtype=np.complex128)
        
        coeffs = self.get_current_coefficients()
        return self._compute_superposition_optimized(coeffs)
    
    def _compute_superposition_optimized(self, coeffs: ArrayC) -> ArrayC:
        """
        Compute superposition using optimized methods.
        
        Uses Numba JIT compilation when available, falls back to einsum otherwise.
        """
        # Ensure spinor stack is up to date
        self._update_spinor_stack()
        
        if self._spinor_stack_real is None:
            return np.zeros((4,) + self.grid.shape(), dtype=np.complex128)
        
        if NUMBA_AVAILABLE:
            # Use Numba-accelerated computation
            coeffs_real = np.real(coeffs).astype(np.float64)
            coeffs_imag = np.imag(coeffs).astype(np.float64)
            
            _compute_superposition_fast(
                coeffs_real, coeffs_imag,
                self._spinor_stack_real, self._spinor_stack_imag,
                self._output_real, self._output_imag
            )
            
            return self._output_real + 1j * self._output_imag
        else:
            # Fallback to NumPy einsum (still fast but not parallel)
            spinors_stack = np.stack([st.spinor for st in self.superposition.states], axis=0)
            return np.einsum('s,sabcd->abcd', coeffs, spinors_stack, optimize=True)

    def density_3d(self, t: Optional[float] = None) -> ArrayR:
        """Compute probability density at time t."""
        psi = self.total_spinor(t)
        return self._compute_density_optimized(psi)
    
    def density_3d_current(self) -> ArrayR:
        """
        Compute probability density at CURRENT time.
        
        OPTIMIZED: Uses Numba JIT compilation when available.
        """
        psi = self.total_spinor_current()
        return self._compute_density_optimized(psi)
    
    def _compute_density_optimized(self, psi: ArrayC) -> ArrayR:
        """Compute density using optimized methods."""
        if NUMBA_AVAILABLE:
            psi_real = np.real(psi).astype(np.float64)
            psi_imag = np.imag(psi).astype(np.float64)
            return _compute_density_fast(psi_real, psi_imag)
        else:
            return np.real(np.sum(np.abs(psi)**2, axis=0))

    def density_slice(self, quantity: str = "density", plane: str = "xy",
                     t: Optional[float] = None) -> ArrayR:
        psi = self.total_spinor(t)
        
        if quantity == "density":
            data = np.real(np.sum(np.abs(psi)**2, axis=0))
        elif quantity == "large":
            data = np.real(np.sum(np.abs(psi[:2])**2, axis=0))
        elif quantity == "small":
            data = np.real(np.sum(np.abs(psi[2:])**2, axis=0))
        else:
            data = np.real(np.sum(np.abs(psi)**2, axis=0))
        
        mid = (self.grid.nx//2, self.grid.ny//2, self.grid.nz//2)
        if plane == "xy":
            return data[:, :, mid[2]]
        elif plane == "xz":
            return data[:, mid[1], :]
        else:
            return data[mid[0], :, :]

    def line_profile(self, axis: str = "x", t: Optional[float] = None) -> Tuple[ArrayR, ArrayR]:
        density = self.density_3d(t)
        mid = (self.grid.nx//2, self.grid.ny//2, self.grid.nz//2)
        
        if axis == "x":
            coord = np.linspace(self.grid.x_min, self.grid.x_max, self.grid.nx)
            prof = density[:, mid[1], mid[2]]
        elif axis == "y":
            coord = np.linspace(self.grid.y_min, self.grid.y_max, self.grid.ny)
            prof = density[mid[0], :, mid[2]]
        else:
            coord = np.linspace(self.grid.z_min, self.grid.z_max, self.grid.nz)
            prof = density[mid[0], mid[1], :]
        
        return coord, prof

    def radial_distribution(self, t: Optional[float] = None, n_bins: int = 128) -> Tuple[ArrayR, ArrayR]:
        """
        Compute radial distribution.
        
        OPTIMIZED: Uses Numba-accelerated binning when available.
        """
        density = self.density_3d(t)
        dv = self._volume_element()
        
        r_flat = self.R.ravel().astype(np.float64)
        rho_flat = density.ravel().astype(np.float64)
        weights = np.ones_like(r_flat) * dv
        
        r_max = float(np.max(r_flat))
        bin_edges = np.linspace(0, r_max, n_bins + 1)
        
        if NUMBA_AVAILABLE:
            sums, counts = _radial_binning_fast(r_flat, rho_flat, bin_edges, weights)
            hist = np.where(counts > 0, sums, 0)
        else:
            hist, _ = np.histogram(r_flat, bins=n_bins, range=(0, r_max), 
                                   weights=rho_flat * dv)
        
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return centers, hist

    def current_3d(self, t: Optional[float] = None) -> Tuple[ArrayR, ArrayR, ArrayR]:
        """Probability current j = ψ†αψ."""
        psi = self.total_spinor(t)
        psi_flat = psi.reshape(4, -1)
        psi_dag = np.conjugate(psi_flat)
        
        jx = np.real(np.sum(psi_dag * (ALPHA_X @ psi_flat), axis=0)).reshape(self.grid.shape())
        jy = np.real(np.sum(psi_dag * (ALPHA_Y @ psi_flat), axis=0)).reshape(self.grid.shape())
        jz = np.real(np.sum(psi_dag * (ALPHA_Z @ psi_flat), axis=0)).reshape(self.grid.shape())
        
        return jx, jy, jz

    def spin_densities(self, t: Optional[float] = None) -> Tuple[ArrayR, ArrayR]:
        """Spin-up and spin-down probability densities."""
        psi = self.total_spinor(t)
        P_up = np.real(np.abs(psi[0])**2 + np.abs(psi[2])**2)
        P_down = np.real(np.abs(psi[1])**2 + np.abs(psi[3])**2)
        return P_up, P_down

    def spin_expectation(self, t: Optional[float] = None) -> ArrayR:
        """Expectation value ⟨S⟩ = (ℏ/2)⟨Σ⟩."""
        psi = self.total_spinor(t)
        dv = self._volume_element()
        psi_flat = psi.reshape(4, -1)
        psi_dag = np.conjugate(psi_flat)
        
        sx, sy, sz = pauli_matrices()
        Sigma = [np.block([[s, np.zeros((2,2))], [np.zeros((2,2)), s]]) for s in [sx, sy, sz]]
        
        S = np.zeros(3)
        for i, Sig in enumerate(Sigma):
            S[i] = 0.5 * np.real(np.sum(psi_dag * (Sig @ psi_flat))) * dv
        
        prob = float(np.sum(np.abs(psi_flat)**2) * dv)
        if prob > 0:
            S /= prob
        
        return S

    def expectation_values(self, t: Optional[float] = None) -> Dict[str, float]:
        """Compute various expectation values."""
        t = t if t is not None else self._time
        psi = self.total_spinor(t)
        dv = self._volume_element()
        
        density = np.real(np.sum(np.abs(psi)**2, axis=0))
        prob = float(np.sum(density) * dv)
        
        if prob > 0:
            x_mean = float(np.sum(self.X * density) * dv / prob)
            y_mean = float(np.sum(self.Y * density) * dv / prob)
            z_mean = float(np.sum(self.Z * density) * dv / prob)
            r_mean = float(np.sum(self.R * density) * dv / prob)
        else:
            x_mean = y_mean = z_mean = r_mean = 0.0
        
        S = self.spin_expectation(t)
        
        return {
            "t": t, "probability": prob,
            "x_mean": x_mean, "y_mean": y_mean, "z_mean": z_mean, "r_mean": r_mean,
            "Sx": S[0], "Sy": S[1], "Sz": S[2],
        }

    def level_summary(self) -> List[str]:
        """Get summary of all states."""
        lines = []
        for j, st in enumerate(self.superposition.states):
            amp, phase = self.superposition.get_coeff_polar(j)
            lines.append(f"[{j}] {st.label} E={st.energy:.6f}, |c|={amp:.3f}")
        return lines

    # -------------------------------------------------------------------------
    # Color Volume Computation (OPTIMIZED)
    # -------------------------------------------------------------------------
    
    def compute_color_volume(self, psi: ArrayC, mode: str = "phase") -> Optional[ArrayR]:
        """
        Compute color volume for isosurface coloring.
        
        OPTIMIZED: Uses Numba JIT compilation when available.
        """
        if mode == "amplitude":
            density = self._compute_density_optimized(psi)
            vol = density.astype(np.float32)
            mx = vol.max()
            return vol / mx if mx > 0 else vol
        
        elif mode == "phase":
            if NUMBA_AVAILABLE:
                return _compute_color_volume_phase(
                    np.real(psi[0]).astype(np.float64),
                    np.imag(psi[0]).astype(np.float64)
                )
            else:
                phase = np.angle(psi[0])
                return ((phase + np.pi) / (2 * np.pi)).astype(np.float32)
        
        elif mode == "spin":
            if NUMBA_AVAILABLE:
                return _compute_color_volume_spin(
                    np.real(psi).astype(np.float64),
                    np.imag(psi).astype(np.float64)
                )
            else:
                P_up = np.abs(psi[0])**2 + np.abs(psi[2])**2
                P_down = np.abs(psi[1])**2 + np.abs(psi[3])**2
                diff = (P_up - P_down).astype(np.float32)
                mx = np.abs(diff).max()
                return (diff / mx * 0.5 + 0.5) if mx > 0 else diff * 0 + 0.5
        
        return None

    # -------------------------------------------------------------------------
    # Save/Load
    # -------------------------------------------------------------------------
    
    def save_configuration(self, filename: str) -> None:
        """Save complete configuration including states, coefficients, and transition settings."""
        spinors = (np.stack([st.spinor for st in self.superposition.states], axis=0) 
                  if self.superposition.states else np.zeros((0, 4) + self.grid.shape()))
        
        trans_data = None
        if self.hamiltonian.transition:
            trans = self.hamiltonian.transition
            trans_data = {
                "state_i": trans.state_i,
                "state_f": trans.state_f,
                "field_amplitude": trans.field_amplitude,
                "field_polarization": trans.field_polarization.tolist(),
                "field_frequency": trans.field_frequency,
            }
        
        metadata = {
            "grid": {"nx": self.grid.nx, "ny": self.grid.ny, "nz": self.grid.nz,
                    "x_range": list(self.grid.x_range), "y_range": list(self.grid.y_range), 
                    "z_range": list(self.grid.z_range)},
            "field": {"Z": self.field.Z, "enable_coulomb": self.field.enable_coulomb},
            "hamiltonian": {"evolution_mode": self.hamiltonian.evolution_mode,
                           "include_rest_mass": self.hamiltonian.include_rest_mass},
            "transition": trans_data,
            "time": self._time,
            "states": [{"kind": st.kind, "label": st.label, "energy": st.energy,
                       "n": st.n, "kappa": st.kappa, "mj": st.mj} 
                      for st in self.superposition.states],
        }
        
        np.savez_compressed(filename, spinors=spinors, coeffs=self.superposition.coeffs,
                           metadata=np.array([metadata], dtype=object))

    def load_configuration(self, filename: str) -> None:
        """
        Load complete configuration.
        
        FIX IO-001: Corrected metadata parsing and added proper cache invalidation.
        """
        data = np.load(filename, allow_pickle=True)
        spinors = data["spinors"]
        coeffs = data["coeffs"]
        
        # FIX IO-001: Handle both array formats for metadata
        # Old format saved as np.array([metadata], dtype=object) -> shape (1,)
        # The [0] gives us the dict, but it might not need .item()
        metadata_raw = data["metadata"]
        if metadata_raw.ndim == 0:
            # 0-d object array
            metadata = metadata_raw.item()
        elif metadata_raw.ndim == 1 and len(metadata_raw) > 0:
            # 1-d array with dict at index 0
            metadata = metadata_raw[0]
            # Only call .item() if it's an ndarray, not if it's already a dict
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()
        else:
            metadata = {}
        
        g = metadata.get("grid", {})
        self.grid = DiracGridConfig(
            nx=g.get("nx", 64), ny=g.get("ny", 64), nz=g.get("nz", 64),
            x_range=tuple(g.get("x_range", [-10000, 10000])),
            y_range=tuple(g.get("y_range", [-10000, 10000])),
            z_range=tuple(g.get("z_range", [-10000, 10000])),
        )
        self._grid_id = str(uuid.uuid4())
        self._init_grid()
        
        f = metadata.get("field", {})
        self.field.Z = f.get("Z", 1)
        self.field.enable_coulomb = f.get("enable_coulomb", True)
        # FIX BUG-004: Sync hamiltonian.include_coulomb with field
        self.hamiltonian.include_coulomb = self.field.enable_coulomb
        
        h = metadata.get("hamiltonian", {})
        self.hamiltonian.evolution_mode = h.get("evolution_mode", "stationary")
        self.hamiltonian.include_rest_mass = h.get("include_rest_mass", False)
        
        states_meta = metadata.get("states", [])
        self.superposition.states = []
        for j, spinor in enumerate(spinors):
            st_meta = states_meta[j] if j < len(states_meta) else {}
            self.superposition.states.append(DiracState(
                spinor=spinor, energy=st_meta.get("energy", 0), kind=st_meta.get("kind", "bound"),
                label=st_meta.get("label", f"state {j}"), n=st_meta.get("n"), 
                kappa=st_meta.get("kappa"), mj=st_meta.get("mj")
            ))
        
        self.superposition._coeffs = np.asarray(coeffs, dtype=np.complex128)
        self._time = metadata.get("time", 0.0)
        
        # FIX IO-001: Properly invalidate ALL caches after load
        self._spinor_stack_valid = False
        self._dipole_cache.clear()
        self._sph_harm_cache.clear()
        
        # FIX NEW-001: Compute _c_current at the loaded time for consistency
        # Must be done after spinor_stack invalidation but before transition setup
        if self._time != 0.0 and self.superposition.n_states() > 0:
            self._c_current = self._coefficients_at_time(self._time)
        else:
            self._c_current = self.superposition.coeffs.copy()
        
        trans_data = metadata.get("transition")
        if trans_data and self.hamiltonian.evolution_mode == "driven":
            trans_config = TransitionConfig(
                state_i=trans_data["state_i"],
                state_f=trans_data["state_f"],
                field_amplitude=trans_data["field_amplitude"],
                field_polarization=np.array(trans_data["field_polarization"]),
                field_frequency=trans_data.get("field_frequency"),
            )
            self.set_evolution_mode("driven", trans_config)
        
        self._dipole_cache.clear()


# =============================================================================
# Performance Info Function
# =============================================================================

def get_performance_info() -> Dict[str, Any]:
    """Get information about available performance optimizations."""
    return {
        "numba_available": NUMBA_AVAILABLE,
        "num_threads": _num_threads,
        "omp_threads": os.environ.get('OMP_NUM_THREADS', 'not set'),
        "openblas_threads": os.environ.get('OPENBLAS_NUM_THREADS', 'not set'),
        "mkl_threads": os.environ.get('MKL_NUM_THREADS', 'not set'),
    }
