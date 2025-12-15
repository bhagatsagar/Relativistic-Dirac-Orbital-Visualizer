import numpy as np
import physics
import rendering
import model

# -------------------------
# A) Exact physics: energy + dipole between two specific states
# -------------------------
qn_i = physics.BoundQN(n=1, kappa=-1, mj=+0.5, Z=1)   # 1s1/2
qn_f = physics.BoundQN(n=2, kappa=+1, mj=+0.5, Z=1)   # 2p1/2

sol_i = physics.solve_radial_analytic(qn_i, include_rest_mass=False)
sol_f = physics.solve_radial_analytic(qn_f, include_rest_mass=False)

E_i = sol_i.E
E_f = sol_f.E
print("Exact energies (binding):", E_i, E_f, "ΔE =", (E_f - E_i))

# Full (polarization-independent) dipole vector:
d_xyz = physics.dipole_e1(sol_i, sol_f)
print("Exact dipole vector d =", d_xyz, "|d| =", np.linalg.norm(d_xyz))

eps = np.array([0.0, 0.0, 1.0], dtype=np.complex128)  # z-polarized light
sel = physics.check_e1_selection_rules(qn_i, qn_f, polarization_xyz=eps)
coupling_scalar = physics.project_dipole(d_xyz, eps)
print("Selection:", sel["reason"], "allowed =", sel["allowed"])
print("eps·d =", coupling_scalar, "|eps·d| =", abs(coupling_scalar))


# -------------------------
# B) Render-only: sample a state on a 3D grid and get a color volume
# -------------------------
grid = rendering.RenderGrid(
    nx=64, ny=64, nz=64,
    x_range=(-200.0, 200.0),
    y_range=(-200.0, 200.0),
    z_range=(-200.0, 200.0),
)
fields = rendering.make_grid_fields(grid)

psi = rendering.sample_bound_spinor(sol_i, fields)
info = rendering.probability_render_estimate(psi, fields)
print("Render-only probability estimate:", info)

rgb = rendering.color_volume(psi, mode="phase")
print("RGB volume shape:", rgb.shape)   # (nx, ny, nz, 3)


# -------------------------
# C) “GUI adapter” style: DiracSolver superpositions + profiling
# -------------------------
solver = model.DiracSolver(grid=grid, Z=1, include_rest_mass=False)
solver.set_render_profiling(True)

solver.add_bound_state(n=1, kappa=-1, mj=+0.5, amplitude=1.0, phase=0.0)
solver.add_bound_state(n=2, kappa=+1, mj=+0.5, amplitude=0.4, phase=1.0)

psi_tot = solver.total_spinor_current()
print("Superposed psi shape:", psi_tot.shape)

print("Render profiling from last sampling:")
print(solver.get_last_render_profile())

expvals = solver.expectation_values()
print("Render-only expectation values:", expvals)


# -------------------------
# D) “Scan transitions / list lines” using the new layer
# -------------------------
lines = model.enumerate_e1_transitions(
    Z=1,
    n_max=3,
    eps_xyz=np.array([0, 0, 1], dtype=np.complex128),
    include_rest_mass=False,
    field_amplitude=1.0,
    direction="absorption",
    top_k=10,
)
print("\nTop transitions (by coupling):")
for t in lines:
    print(
        f"{t.qn_i} -> {t.qn_f}  omega0={t.omega0:.6e}  coupling={t.coupling:.6e}  "
        f"allowed_eps={t.allowed_with_eps}"
    )
