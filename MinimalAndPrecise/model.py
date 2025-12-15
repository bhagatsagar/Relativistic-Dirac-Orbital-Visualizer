"""
model.py — GUI-friendly DiracSolver-like adapter.

Key rules:
- Exact physics (energies, selection rules, dipoles) comes from physics.py only.
- This module may perform **render-only** interpolation/sampling to a 3D grid via rendering.py.
- Time evolution is stationary: coefficients acquire phases exp(-i E t).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Literal

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

import physics
import rendering


@dataclass
class _State:
    qn: physics.BoundQN
    sol: physics.RadialSolution
    coeff: complex
    psi_cache: Optional[np.ndarray] = None  # shape (4,nx,ny,nz)


class DiracSolver:
    """Compact state manager used by the GUI.

    Stores exact bound states (RadialSolution) and caches render-grid samples (psi_cache).
    """

    def __init__(self, grid: rendering.RenderGrid, Z: int, include_rest_mass: bool = False):
        self.grid: rendering.RenderGrid = grid
        self.fields: Dict[str, Any] = rendering.make_grid_fields(grid)
        self.Z: int = int(Z)
        self.include_rest_mass: bool = bool(include_rest_mass)

        self.states: List[_State] = []
        self.t: float = 0.0

        # Transition configuration
        self._trans_pair: Optional[Tuple[int, int]] = None
        self._trans_pol: Optional[np.ndarray] = None
        self._trans_field_amp: float = 1.0
        # Render profiling (optional)
        self.render_profile_enabled: bool = False
        self.last_render_profile: Dict[str, Any] = {}


    def set_render_profiling(self, enabled: bool) -> None:
        """Enable/disable render profiling (sampling only)."""
        self.render_profile_enabled = bool(enabled)

    def get_last_render_profile(self) -> Dict[str, Any]:
        """Return last recorded render sampling timings (copy)."""
        return dict(self.last_render_profile)

    # -----------------------
    # Grid & nuclear charge
    # -----------------------

    def set_nuclear_charge(self, Z: int) -> None:
        """Set Z and recompute exact radial solutions for all states (invalidates caches)."""
        self.Z = int(Z)
        new_states: List[_State] = []
        for st in self.states:
            qn = physics.BoundQN(n=st.qn.n, kappa=st.qn.kappa, mj=st.qn.mj, Z=self.Z)
            sol = physics.solve_radial_analytic(qn, include_rest_mass=self.include_rest_mass)
            new_states.append(_State(qn=qn, sol=sol, coeff=st.coeff, psi_cache=None))
        self.states = new_states

    def update_grid(self, new_grid: rendering.RenderGrid) -> None:
        """Update render grid and invalidate all cached samples + per-grid caches (render-only)."""
        self.grid = new_grid
        self.fields = rendering.make_grid_fields(new_grid)  # new fields => new per-grid caches
        for st in self.states:
            st.psi_cache = None
        self.last_render_profile = {}

    # -----------------------
    # State management
    # -----------------------

    def add_bound_state(self, n: int, kappa: int, mj: float, amplitude: float = 1.0, phase: float = 0.0) -> None:
        """Add an exact bound state and initialize its complex coefficient."""
        qn = physics.BoundQN(n=int(n), kappa=int(kappa), mj=float(mj), Z=int(self.Z))
        sol = physics.solve_radial_analytic(qn, include_rest_mass=self.include_rest_mass)
        coeff = complex(float(amplitude)) * complex(math.cos(float(phase)), math.sin(float(phase)))
        self.states.append(_State(qn=qn, sol=sol, coeff=coeff, psi_cache=None))

    def remove_state(self, i: int) -> None:
        """Remove state by index."""
        self.states.pop(int(i))

    def level_summary(self) -> List[Dict[str, Any]]:
        """Return a summary list suitable for GUI display."""
        out: List[Dict[str, Any]] = []
        for idx, st in enumerate(self.states):
            E = float(st.sol.E)
            out.append(
                {
                    "index": idx,
                    "Z": st.qn.Z,
                    "n": st.qn.n,
                    "kappa": st.qn.kappa,
                    "mj": st.qn.mj,
                    "E": E,
                    "coeff": st.coeff,
                    "amplitude": abs(st.coeff),
                    "phase": float(np.angle(st.coeff)),
                }
            )
        return out

    # -----------------------
    # Sampling / superposition
    # -----------------------

    def _ensure_sampled(self, i: int) -> np.ndarray:
        """Ensure psi_cache exists for state i; sampling is render-only approximation."""
        st = self.states[int(i)]
        if st.psi_cache is None:
            st.psi_cache = rendering.sample_bound_spinor(st.sol, self.fields)
        return st.psi_cache

    def total_spinor_current(self) -> np.ndarray:
        """Return the current superposed 4-spinor on the render grid (render-only sampling)."""
        if not self.states:
            nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
            return np.zeros((4, nx, ny, nz), dtype=np.complex128)

        # Collect sampling profiles only when we actually sample (psi_cache miss)
        sampled_profiles: List[Dict[str, Any]] = []
        total_profile: Dict[str, float] = {"angular_s": 0.0, "interp_s": 0.0, "assemble_s": 0.0}

        psi_tot = None
        for idx, st in enumerate(self.states):
            if st.psi_cache is None:
                prof = {} if self.render_profile_enabled else None
                st.psi_cache = rendering.sample_bound_spinor(st.sol, self.fields, profile=prof)

                if prof is not None:
                    # Aggregate
                    total_profile["angular_s"] += float(prof.get("angular_s", 0.0))
                    total_profile["interp_s"] += float(prof.get("interp_s", 0.0))
                    total_profile["assemble_s"] += float(prof.get("assemble_s", 0.0))

                    sampled_profiles.append(
                        {
                            "state_index": idx,
                            "n": st.qn.n,
                            "kappa": st.qn.kappa,
                            "mj": st.qn.mj,
                            **prof,
                        }
                    )

            if psi_tot is None:
                psi_tot = st.coeff * st.psi_cache
            else:
                psi_tot = psi_tot + st.coeff * st.psi_cache

        if self.render_profile_enabled:
            self.last_render_profile = {
                "states_sampled": sampled_profiles,
                "total_angular_s": total_profile["angular_s"],
                "total_interp_s": total_profile["interp_s"],
                "total_assemble_s": total_profile["assemble_s"],
            }

        return np.asarray(psi_tot, dtype=np.complex128)

    # -----------------------
    # Stationary time stepping
    # -----------------------

    def step(self, dt: float) -> None:
        """Advance time by dt with stationary phases: c_k *= exp(-i E_k dt)."""
        dt = float(dt)
        self.t += dt
        for st in self.states:
            E = float(st.sol.E)  # energy representation matches include_rest_mass choice
            st.coeff *= complex(math.cos(-E * dt), math.sin(-E * dt))

    # -----------------------
    # Transitions (exact physics from physics.py)
    # -----------------------

    def set_transition_pair(
        self,
        i: int,
        f: int,
        polarization_xyz: np.ndarray,
        field_amplitude: float = 1.0,
    ) -> None:
        """Set which states define the (i->f) transition for reporting (no driven dynamics)."""
        self._trans_pair = (int(i), int(f))
        self._trans_pol = np.asarray(polarization_xyz, dtype=np.complex128).reshape(3)
        self._trans_field_amp = float(field_amplitude)

    def get_transition_info(self) -> Dict[str, Any]:
        """Report ω0, selection rules (optionally gated by ε), full dipole vector, and polarization-dependent coupling."""
        if self._trans_pair is None or self._trans_pol is None:
            return {"configured": False}

        i, f = self._trans_pair
        if not (0 <= i < len(self.states) and 0 <= f < len(self.states)):
            return {"configured": False, "reason": "transition indices out of range"}

        sol_i = self.states[i].sol
        sol_f = self.states[f].sol

        Ei = float(sol_i.E)
        Ef = float(sol_f.E)
        omega0 = abs(Ef - Ei)

        # Selection rules *including* polarization gating
        sel = physics.check_e1_selection_rules(sol_i.qn, sol_f.qn, self._trans_pol)

        # Full dipole vector (independent of polarization)
        d_full = physics.dipole_e1(sol_i, sol_f)
        dmag_full = float(np.linalg.norm(d_full))

        # Polarization-dependent scalar coupling: ε·d (then multiplied by field amplitude)
        epsdotd = physics.project_dipole(d_full, self._trans_pol)
        if not sel.get("allowed", False):
            # Enforce exact "forbidden given ε" behavior at the reporting layer
            epsdotd = 0.0 + 0.0j

        coupling = float(self._trans_field_amp * abs(epsdotd))

        return {
            "configured": True,
            "i": i,
            "f": f,
            "omega0": float(omega0),
            "detuning": 0.0,
            "selection": sel,
            "q_used": sel.get("used_q", []),

            # New unambiguous reporting
            "dipole_vector_full": d_full,
            "dipole_magnitude_full": dmag_full,
            "eps_dot_d": epsdotd,
            "field_amplitude": float(self._trans_field_amp),
            "coupling": coupling,

            # Backward-friendly aliases (optional but harmless)
            "dipole_vector": d_full,
            "dipole_magnitude": dmag_full,
            "coupling_proxy": coupling,

            "exact_source": "physics.py",
            "render_source": "rendering.py (render-only; not used here)",

        }


    # -----------------------
    # Expectation values (render-only integration)
    # -----------------------

    def expectation_values(self) -> Dict[str, Any]:
        """Render-only expectation values from the sampled grid.

        Guardrails:
        - Raises if dv==0 (degenerate grid: any axis has size 1).
        - Returns explicit render_estimate_* keys to prevent treating these as exact physics.
        - Adds a warning if the render probability is suspiciously low (default <0.9),
        which usually indicates the grid is too small to capture normalization.
        """
        # Hard guardrail: grid-based integration requires dv>0
        rendering.assert_valid_volume_element(self.fields, where="DiracSolver.expectation_values")

        psi = self.total_spinor_current()
        dv = float(self.fields["dv"])
        R = np.asarray(self.fields["R"], dtype=np.float64)

        # Probability estimate + warning (render-only)
        prob_info = rendering.probability_render_estimate(psi, self.fields, warn_if_below=0.9)
        prob = float(prob_info["render_estimate_probability"])
        warn_msg = str(prob_info.get("render_estimate_warning", "")) if "render_estimate_warning" in prob_info else ""

        # If probability is unusable, return NaNs but keep it clearly labeled
        if prob <= 0.0 or (not np.isfinite(prob)):
            out: Dict[str, Any] = {
                "render_estimate_probability": prob,
                "render_estimate_r_mean": float("nan"),
                "render_estimate_Sz_mean": float("nan"),
                "render_estimate_note": "render-only estimate (not exact); check grid extent/resolution",
            }
            if warn_msg:
                out["render_estimate_warning"] = warn_msg
            return out

        rho = rendering.density(psi)

        # ⟨r⟩ render estimate
        r_mean = float(np.sum(R * rho) * dv / prob)

        # ⟨S_z⟩ render estimate: S_z = 1/2 diag(σ_z, σ_z)
        a0 = np.abs(psi[0]) ** 2
        a1 = np.abs(psi[1]) ** 2
        a2 = np.abs(psi[2]) ** 2
        a3 = np.abs(psi[3]) ** 2
        sz_density = 0.5 * ((a0 - a1) + (a2 - a3))
        Sz_mean = float(np.sum(sz_density) * dv / prob)

        out2: Dict[str, Any] = {
            "render_estimate_probability": prob,
            "render_estimate_r_mean": r_mean,
            "render_estimate_Sz_mean": Sz_mean,
            "render_estimate_note": "render-only estimate (not exact); increase grid extent/resolution for better normalization",
        }
        if warn_msg:
            out2["render_estimate_warning"] = warn_msg
        return out2


    # -----------------------
    # Visualization helpers (proxies)
    # -----------------------

    def compute_color_volume(self, psi: np.ndarray, mode: str) -> np.ndarray:
        """Proxy to rendering.color_volume."""
        return rendering.color_volume(psi, mode=mode)  # type: ignore[arg-type]

    def radial_distribution_from_density(self, dens: np.ndarray, n_bins: int = 128) -> Dict[str, Any]:
        """Proxy to rendering.radial_distribution_from_density using current grid fields."""
        R = np.asarray(self.fields["R"], dtype=np.float64)
        dv = float(self.fields["dv"])
        return rendering.radial_distribution_from_density(R, dens, dv, n_bins=int(n_bins))
    
# =============================================================================
# Transition enumeration layer (exact physics via physics.py; no render shortcuts)
# =============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundStateInfo:
    qn: physics.BoundQN
    E: float
    l: int
    j: float


@dataclass(frozen=True)
class TransitionInfo:
    qn_i: physics.BoundQN
    qn_f: physics.BoundQN
    Ei: float
    Ef: float
    delta_E: float
    omega0: float
    coupling: float               # field_amp * |eps · d|
    eps_dot_d: complex
    allowed_intrinsic: bool
    allowed_with_eps: bool
    selection_intrinsic: Dict[str, Any]
    selection_with_eps: Dict[str, Any]
    radial_integral: float
    radial_err_est: float


def _mj_list_for_j(j: float) -> List[float]:
    """Return mj values from -j..+j step 1, as floats (half-integers)."""
    j2 = int(round(2.0 * float(j)))
    m2s = list(range(-j2, j2 + 1, 2))
    return [0.5 * m2 for m2 in m2s]


def enumerate_bound_states(
    *,
    Z: int,
    n_max: int,
    include_rest_mass: bool = False,
) -> List[BoundStateInfo]:
    """Enumerate bound states up to n_max for fixed Z with allowed (kappa, mj).

    Uses physics.py as source of truth:
      - kappa->(l,j) via physics.kappa_to_l_j
      - energy via physics.energy_dirac_coulomb
      - validity via physics.validate_qn (implicitly inside energy_dirac_coulomb)
    """
    Z = int(Z)
    n_max = int(n_max)
    if n_max < 1:
        raise ValueError("n_max must be >= 1")

    out: List[BoundStateInfo] = []

    for n in range(1, n_max + 1):
        # Allowed kappa: negative κ = -1..-n, positive κ = +1..+(n-1)
        kappas = list(range(-n, 0)) + list(range(1, n))
        for kappa in kappas:
            l, j = physics.kappa_to_l_j(int(kappa))

            # Energy depends on (n,kappa,Z) only
            # (skip invalid states e.g. too-large Z*alpha for |kappa|)
            try:
                qn_rep = physics.BoundQN(n=n, kappa=int(kappa), mj=+0.5, Z=Z)
                E = float(physics.energy_dirac_coulomb(qn_rep, include_rest_mass=bool(include_rest_mass)))
            except Exception:
                continue

            for mj in _mj_list_for_j(j):
                qn = physics.BoundQN(n=n, kappa=int(kappa), mj=float(mj), Z=Z)
                # validate explicitly (mj bound, Zα<|κ|, n>l, etc.)
                try:
                    physics.validate_qn(qn)
                except Exception:
                    continue
                out.append(BoundStateInfo(qn=qn, E=E, l=int(l), j=float(j)))

    return out


def _normalize_eps(eps_xyz: np.ndarray) -> np.ndarray:
    eps = np.asarray(eps_xyz, dtype=np.complex128).reshape(3)
    nrm = float(np.vdot(eps, eps).real) ** 0.5
    return eps if nrm == 0.0 else (eps / nrm)


def _cartesian_from_spherical(d_m1: complex, d_0: complex, d_p1: complex) -> np.ndarray:
    """Same convention as physics._cartesian_from_spherical, redefined locally to avoid private access."""
    import math
    vx = (d_m1 - d_p1) / math.sqrt(2.0)
    vy = 1j * (d_m1 + d_p1) / math.sqrt(2.0)
    vz = d_0
    return np.array([vx, vy, vz], dtype=np.complex128)


def enumerate_e1_transitions(
    *,
    Z: int,
    n_max: int,
    eps_xyz: np.ndarray,
    include_rest_mass: bool = False,
    field_amplitude: float = 1.0,
    direction: Literal["absorption", "emission", "both"] = "absorption",
    top_k: Optional[int] = 50,
    radial_n_points: int = 4000,
    radial_grid: Literal["log", "linear"] = "log",
) -> List[TransitionInfo]:
    """Enumerate E1-allowed transitions among bound states up to n_max.

    Filtering:
      - intrinsic E1 allowed via physics.check_e1_selection_rules(..., polarization_xyz=None)
    Coupling:
      - compute exact vector dipole via (radial integral) * (angular_C1), then eps·d
      - if epsilon-gated selection forbids, coupling is forced to 0 exactly

    Exact sources:
      - selection rules: physics.check_e1_selection_rules
      - radial integral: physics.radial_dipole_integral
      - angular factor: physics.angular_C1
      - projection: physics.project_dipole
    """
    Z = int(Z)
    n_max = int(n_max)
    eps = _normalize_eps(np.asarray(eps_xyz, dtype=np.complex128))
    field_amplitude = float(field_amplitude)

    # Build level table grouped by (n,kappa) so we can reuse radial work (mj does not affect radial).
    states = enumerate_bound_states(Z=Z, n_max=n_max, include_rest_mass=include_rest_mass)
    level_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for st in states:
        key = (st.qn.n, st.qn.kappa)
        ent = level_map.get(key)
        if ent is None:
            ent = {
                "n": st.qn.n,
                "kappa": st.qn.kappa,
                "l": st.l,
                "j": st.j,
                "E": st.E,
                "mjs": [],
            }
            level_map[key] = ent
        ent["mjs"].append(st.qn.mj)

    # Representative exact radial solution per (n,kappa)
    sol_cache: Dict[Tuple[int, int], physics.RadialSolution] = {}
    for key, ent in level_map.items():
        n, kappa = key
        # mj chosen only to satisfy validation; radial does not depend on mj
        qn_rep = physics.BoundQN(n=n, kappa=kappa, mj=+0.5, Z=Z)
        sol_cache[key] = physics.solve_radial_analytic(
            qn_rep,
            include_rest_mass=bool(include_rest_mass),
            n_points=int(radial_n_points),
            grid=str(radial_grid),  # type: ignore[arg-type]
        )

    # Cache radial integrals by level-pair (directional key)
    radint_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[float, float]] = {}

    def _radial_pair(key_i: Tuple[int, int], key_f: Tuple[int, int]) -> Tuple[float, float]:
        k = (key_i, key_f)
        hit = radint_cache.get(k)
        if hit is not None:
            return hit
        Ri = sol_cache[key_i]
        Rf = sol_cache[key_f]
        val = float(physics.radial_dipole_integral(Ri, Rf))
        err = float(getattr(physics.radial_dipole_integral, "last_error", float("nan")))
        radint_cache[k] = (val, err)
        return val, err

    # Enumerate transitions
    keys = list(level_map.keys())
    out: List[TransitionInfo] = []

    for key_i in keys:
        ent_i = level_map[key_i]
        Ei = float(ent_i["E"])
        l_i = int(ent_i["l"])
        j_i = float(ent_i["j"])
        mjs_i = list(ent_i["mjs"])

        for key_f in keys:
            if key_f == key_i:
                continue

            ent_f = level_map[key_f]
            Ef = float(ent_f["E"])
            l_f = int(ent_f["l"])
            j_f = float(ent_f["j"])
            mjs_f = list(ent_f["mjs"])

            # Direction filter
            if direction == "absorption" and not (Ef > Ei):
                continue
            if direction == "emission" and not (Ef < Ei):
                continue

            # Quick orbital/j triangle prefilter (mj-independent)
            dl = l_f - l_i
            if not (dl == +1 or dl == -1):
                continue
            if not (abs(j_f - j_i) <= 1.0 + 1e-15 and (j_f + j_i) >= 1.0 - 1e-15):
                continue

            # Radial integral once per (n,kappa)->(n,kappa)
            R_if, Rerr = _radial_pair(key_i, key_f)

            n_i, kappa_i = key_i
            n_f, kappa_f = key_f

            for mj_i in mjs_i:
                qn_i = physics.BoundQN(n=n_i, kappa=kappa_i, mj=float(mj_i), Z=Z)

                for mj_f in mjs_f:
                    qn_f = physics.BoundQN(n=n_f, kappa=kappa_f, mj=float(mj_f), Z=Z)

                    # Intrinsic E1 selection (no epsilon gating)
                    sel0 = physics.check_e1_selection_rules(qn_i, qn_f, polarization_xyz=None)
                    if not sel0.get("allowed_intrinsic", False):
                        continue

                    # Epsilon-gated selection (exact gating layer)
                    sel_eps = physics.check_e1_selection_rules(qn_i, qn_f, polarization_xyz=eps)
                    allowed_eps = bool(sel_eps.get("allowed", False))

                    # Exact angular factor (Wigner–Eckart) and exact radial integral (physics model)
                    d_m1 = complex(R_if) * physics.angular_C1(kappa_f, mj_f, kappa_i, mj_i, -1)
                    d_0  = complex(R_if) * physics.angular_C1(kappa_f, mj_f, kappa_i, mj_i,  0)
                    d_p1 = complex(R_if) * physics.angular_C1(kappa_f, mj_f, kappa_i, mj_i, +1)

                    d_xyz = _cartesian_from_spherical(d_m1, d_0, d_p1)
                    epsdotd = physics.project_dipole(d_xyz, eps)

                    if not allowed_eps:
                        epsdotd = 0.0 + 0.0j

                    coupling = field_amplitude * float(abs(epsdotd))
                    delta_E = float(Ef - Ei)
                    omega0 = float(abs(delta_E))

                    out.append(
                        TransitionInfo(
                            qn_i=qn_i,
                            qn_f=qn_f,
                            Ei=Ei,
                            Ef=Ef,
                            delta_E=delta_E,
                            omega0=omega0,
                            coupling=float(coupling),
                            eps_dot_d=complex(epsdotd),
                            allowed_intrinsic=True,
                            allowed_with_eps=allowed_eps,
                            selection_intrinsic=sel0,
                            selection_with_eps=sel_eps,
                            radial_integral=float(R_if),
                            radial_err_est=float(Rerr),
                        )
                    )

    # Sort by coupling (desc), then omega0 (desc) for "top lines"
    out.sort(key=lambda tr: (tr.coupling, tr.omega0), reverse=True)

    if top_k is not None:
        return out[: int(top_k)]
    return out


# Optional convenience methods on the GUI solver (doesn't affect correctness)
def _diracsolver_scan_transitions(self: "DiracSolver", n_max: int, eps_xyz: np.ndarray, **kwargs: Any) -> List[TransitionInfo]:
    return enumerate_e1_transitions(
        Z=self.Z,
        n_max=int(n_max),
        eps_xyz=np.asarray(eps_xyz, dtype=np.complex128),
        include_rest_mass=self.include_rest_mass,
        **kwargs,
    )


# Monkey-patch style attach (keeps API small and avoids core redesign)
setattr(DiracSolver, "scan_transitions_e1", _diracsolver_scan_transitions)


if __name__ == "__main__":
    # CLI-like demo: print top transitions (by coupling) for a chosen epsilon
    import argparse

    p = argparse.ArgumentParser(description="Enumerate Dirac–Coulomb E1 transitions (exact physics from physics.py).")
    p.add_argument("--Z", type=int, default=1)
    p.add_argument("--nmax", type=int, default=3)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--include-rest-mass", action="store_true")
    p.add_argument("--direction", choices=["absorption", "emission", "both"], default="absorption")
    p.add_argument("--eps", type=str, default="0,0,1", help="polarization vector 'ex,ey,ez' (real or complex tokens)")
    p.add_argument("--field-amp", type=float, default=1.0)
    args = p.parse_args()

    def _parse_eps(s: str) -> np.ndarray:
        parts = [t.strip() for t in s.split(",")]
        if len(parts) != 3:
            raise ValueError("eps must have 3 comma-separated components")
        # allow tokens like '1', '0.5', '1j', '0.2+0.1j'
        vals = [complex(tok) for tok in parts]
        return np.array(vals, dtype=np.complex128)

    eps = _parse_eps(args.eps)

    lines = enumerate_e1_transitions(
        Z=args.Z,
        n_max=args.nmax,
        eps_xyz=eps,
        include_rest_mass=args.include_rest_mass,
        field_amplitude=args.field_amp,
        direction=args.direction,
        top_k=args.top,
    )

    def _fmt(qn: physics.BoundQN) -> str:
        return f"(n={qn.n}, κ={qn.kappa:+d}, mj={qn.mj:+.1f})"

    print(f"Top {len(lines)} E1 transitions for Z={args.Z}, n_max={args.nmax}, eps={_normalize_eps(eps)}")
    print("Sorted by coupling (desc). Units: ω0 and E in natural units (ħ=c=m=1).")
    print("-" * 110)
    for k, tr in enumerate(lines, 1):
        tag = "OK" if tr.allowed_with_eps else "ε-forbidden"
        print(
            f"{k:2d}. {_fmt(tr.qn_i)} -> {_fmt(tr.qn_f)}  "
            f"ω0={tr.omega0:.6e}  |ε·d|*E0={tr.coupling:.6e}  ({tag})  "
            f"rad_err≈{tr.radial_err_est:.2e}"
        )
