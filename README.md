# Relativistic Dirac Orbital Visualizer

A high-performance Python application for visualizing relativistic hydrogen-like wavefunctions and dipole transitions using the Dirac equation.

## Overview

This application provides real-time 3D visualization of analytic hydrogenic Dirac eigenstates and their time evolution under electric dipole (E1) transitions. It features:

- **Relativistic Quantum Mechanics**: Full 4-component Dirac spinor wavefunctions
- **3D Isosurface Rendering**: VTK-based visualization with phase/spin coloring
- **Time Evolution**: Both stationary and driven dynamics
- **Selection Rules**: Automatic E1 transition validation
- **High Performance**: Numba JIT compilation and parallel processing

## Physics Background

### Dirac Equation

The code solves the single-particle Dirac-Coulomb problem for hydrogenic atoms:

```
H = α·p + βm + V(r)
```

where `V(r) = -Zα/r` is the Coulomb potential in natural units (ℏ = c = mₑ = 1).

### Supported Features

- Exact analytic Dirac eigenstates for hydrogen-like atoms
- Electric dipole (E1) matrix elements via radial integration
- Time evolution with classical driving fields
- Quantum numbers: n, κ (kappa), mⱼ

### Limitations

- No QED radiative corrections (Lamb shift, vacuum polarization)
- Classical electromagnetic fields only
- Truncated basis for driven dynamics

## Installation

### Prerequisites

- Python 3.9 or higher
- A display capable of OpenGL rendering

### Required Dependencies

```bash
pip install numpy scipy PySide6 pyqtgraph vtk
```

### Optional (Recommended for Performance)

```bash
pip install numba
```

Installing Numba enables JIT compilation which provides **2-10x speedup** for numerical computations.

### Full Installation

```bash
# Clone the repository
git clone https://github.com/M3C3I/Relativistic-Dirac-Orbital-Visualizer.git
cd Relativistic-Dirac-Orbital-Visualizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Usage

### Quick Start

```bash
python gui_app.py
```

### Adding States

1. Open the **States** tab in the control panel
2. Set the quantum numbers:
   - **n**: Principal quantum number (1, 2, 3, ...)
   - **κ**: Dirac quantum number (±1, ±2, ...; κ=0 is forbidden)
   - **mⱼ**: Magnetic quantum number (-j to +j in half-integer steps)
3. Click **Add Bound State**

### Understanding κ (Kappa)

The Dirac quantum number κ encodes both orbital and total angular momentum:

| κ | l | j | Spectroscopic |
|---|---|---|---------------|
| -1 | 0 | 1/2 | s₁/₂ |
| +1 | 1 | 1/2 | p₁/₂ |
| -2 | 1 | 3/2 | p₃/₂ |
| +2 | 2 | 3/2 | d₃/₂ |
| -3 | 2 | 5/2 | d₅/₂ |

### Setting Up Transitions

1. Add at least 2 states
2. Open the **Transitions** tab
3. Select initial and final states
4. Choose polarization (z, x, or y)
5. Set field amplitude E₀
6. Click **Apply Transition**
7. Switch evolution mode to "Driven Transition"
8. Press **Play** to see time evolution

### Selection Rules

E1 transitions require:
- Δl = ±1 (parity change)
- Δj = 0, ±1 (but j=0 → j=0 forbidden)
- Δmⱼ = q, where q depends on polarization:
  - z-polarized: Δmⱼ = 0
  - x-polarized: Δmⱼ = ±1
  - y-polarized: Δmⱼ = ±1

The GUI automatically validates these rules and displays allowed Δm values.

### Visualization Controls

#### 3D View
- **Iso level slider**: Adjust isosurface threshold (percentile-based by default)
- **Color mode**: Phase, Amplitude, or Spin
- **Reset Camera**: Return to default view
- Click and drag to rotate, scroll to zoom

#### 2D Slice
- Select viewing plane (xy, xz, or yz)
- Choose quantity (total density, large/small components)

#### Grid Resolution
- Available sizes: 32³, 48³, 64³, 96³, 128³, 256³
- Higher resolution = better quality but slower rendering

## Performance Optimization

### Hardware Utilization

The optimized code utilizes available CPU cores through:

1. **Numba JIT Compilation**: Parallel loops with `prange`
2. **NumPy Threading**: Automatic BLAS parallelization
3. **Vectorized Operations**: Eliminates Python loops

### Performance Tips

1. **Install Numba**: Provides 2-10x speedup
   ```bash
   pip install numba
   ```

2. **Use appropriate grid size**:
   - 64³ for exploration (default)
   - 96³-128³ for high-quality renders
   - 32³ for fast iteration

3. **Check performance status**: The View tab shows:
   - Numba JIT status (✓ Enabled / ✗ Not available)
   - Number of threads being used

### Threading Configuration

The application automatically detects available CPU cores. To manually configure:

```python
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['NUMBA_NUM_THREADS'] = '8'
```

Set these **before** importing the module.

## API Reference

### DiracSolver

Main computation class:

```python
from dirac_core import DiracSolver, DiracGridConfig, FieldConfig

# Create solver
grid = DiracGridConfig(nx=64, ny=64, nz=64)
solver = DiracSolver(grid, FieldConfig(Z=1))

# Add states
solver.add_bound_state(n=1, kappa=-1, mj=0.5)  # 1s₁/₂
solver.add_bound_state(n=2, kappa=+1, mj=0.5)  # 2p₁/₂

# Time evolution
solver.step(dt=1.0)

# Get density
density = solver.density_3d_current()
```

### Key Functions

```python
# Dirac energy levels
from dirac_core import hydrogenic_dirac_energy
E = hydrogenic_dirac_energy(n=2, kappa=-1, Z=1)

# Check selection rules
from dirac_core import check_e1_selection_rules
allowed, reason, dm_list = check_e1_selection_rules(state_i, state_f, polarization)

# Performance info
from dirac_core import get_performance_info
info = get_performance_info()
print(f"Numba available: {info['numba_available']}")
print(f"Threads: {info['num_threads']}")
```

## File Format

Configurations can be saved/loaded as `.npz` files containing:
- Grid parameters
- Nuclear charge Z
- State quantum numbers and spinors
- Superposition coefficients
- Transition settings

## Troubleshooting

### "VTK not found" Error

```bash
pip install vtk
```

On Linux, you may need OpenGL libraries:
```bash
sudo apt-get install libgl1-mesa-dev
```

### Slow Performance

1. Install Numba: `pip install numba`
2. Reduce grid resolution in View tab
3. Check thread count in Performance panel

### Black/Empty 3D View

- Add at least one state
- Adjust iso level slider (try 5-15%)
- Click "Reset Camera"

### Memory Issues with Large Grids

- 256³ grid requires ~4GB RAM
- Use 64³ or 96³ for most applications

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request


## Citation

If you use this software in research, please cite:

```bibtex
@software{relativistic_dirac_orbital_visualizer,
  title = {Relativistic Dirac Orbital Visualizer},
  year = {2025},
  url = {https://github.com/M3C3I/Relativistic-Dirac-Orbital-Visualizer.git}
}
```

## Acknowledgments

- Dirac equation formulation follows Bjorken & Drell conventions
- Spherical harmonics from SciPy
- 3D rendering via VTK
- GUI framework: PySide6 (Qt for Python)

## References

1. Bjorken, J.D. & Drell, S.D. "Relativistic Quantum Mechanics" (1964)
2. Berestetskii, V.B. et al. "Quantum Electrodynamics" (1982)
3. Grant, I.P. "Relativistic Quantum Theory of Atoms and Molecules" (2007)
