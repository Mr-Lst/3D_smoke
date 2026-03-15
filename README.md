# 3D Smoke Simulation with Pulsating Heat Source

**Author:** [Your Name]  
**Date:** March 2026  
**Language:** Python 3.x  

---

## Introduction

This project presents a **numerical simulation of three-dimensional smoke flow** within a confined domain, featuring a **pulsating heat source**. The program employs a **Bernstein collocation method** to approximate the solutions of the governing partial differential equations for fluid flow and thermal dynamics, taking into account **viscosity, thermal diffusivity, buoyancy, and gravity**.  

The primary goal of this framework is to provide a **flexible and precise computational tool** for studying heat transfer and vortex evolution in a 3D spatial domain over time, while respecting **well-posed boundary conditions** for walls and the pulsating source.

---

## Requirements

- Python 3.9 or higher  
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `numba`

---

## Key Features

1. **Bernstein Basis Functions**
   - Supports arbitrary polynomial degrees (`Nx`, `Ny`, `Nz`, `Nt`) for each spatial and temporal dimension.  
   - Efficient computation of first and second derivatives via **Numba JIT acceleration**.  

2. **PDE System**
   - Solves for velocity components `(vx, vy, vz)`, pressure `(p)`, and temperature `(T)`.  
   - Sparse matrix formulation using **Kronecker products** to reduce memory usage.  
   - Linear system solved iteratively using `GMRES` or directly with `spsolve`.  

3. **Dynamic Boundary Conditions**
   - Wall boundaries enforced at domain edges.  
   - Central pulsating heat source with adjustable frequency.  
   - Pressure constraint applied to ensure numerical stability.  

4. **Flexibility in Space-Time Discretization**
   - Adjustable polynomial degree for high-resolution approximation.  
   - Modifiable physical properties: viscosity (`mu`), thermal expansion (`beta`), thermal diffusivity (`alpha`), and gravity (`g`).  

5. **Result Visualization**
   - Temperature evolution at the domain center over time.  
   - 2D temperature slices at the final time step along selected planes.

---

## How to Use

1. Run the program:

\`\`\`bash
python smoke_simulation.py
\`\`\`

2. Enter the prompted input values, such as:

- Polynomial degrees: `Nx`, `Ny`, `Nz`, `Nt`  
- Physical properties: `mu`, `alpha`, `beta`, `g`  
- Temperatures: `T_cold`, `T_hot`  
- Pulsation frequency: `freq`  

3. Observe the **Newton-Raphson iteration** progress until convergence:

\`\`\`text
Iter 1: norm(F) = 3.12e-01

Iter 2: norm(F) = 1.04e-02
> Converged.
\`\`\`

4. After convergence, the program generates:

- **Temperature evolution curve** at the center of the domain.  
- **2D temperature contour slice** at a mid-plane at final time.

---

## Code Structure

- `comb_numba`: Efficient computation of combinatorial coefficients using Numba.  
- `bernstein`, `bernstein_deriv`, `bernstein_deriv2`: Bernstein basis functions and derivatives.  
- `build_collocation_matrix`: Constructs collocation matrices for function approximation.  
- `apply_boundary_conditions`: Enforces boundary conditions at walls.  
- `apply_dynamic_heat_source`: Applies a time-dependent heat source in the domain center.  
- `apply_pressure_constraint`: Stabilizes pressure by constraining the global mean.  
- `build_system`: Constructs the complete PDE system as a sparse linear system.  
- `main`: Orchestrates Newton iterations, linear solves, and visualization.

---

## Scientific Notes

- The method leverages **high-order polynomial approximations** for smooth solutions.  
- **Sparse matrices** and **Kronecker products** allow for high-dimensional discretization without prohibitive memory usage.  
- The pulsating heat source provides a mechanism to study **frequency-dependent thermal response**.  
- Designed to be **modular and extensible** for research purposes in fluid dynamics and heat transfer.

---

## Contribution and Extensions

- GPU acceleration using **CuPy** for faster computation.  
- Implementation of **complex boundary conditions**, e.g., moving walls.  
- Expansion of temporal discretization to explore long-term flow behavior.  

---

> 💡 **Note:** This code is intended as a reliable scientific tool, combining **mathematical rigor, computational efficiency, and numerical stability**. It is suitable for academic research and exploratory studies in fluid mechanics and thermal dynamics.

EOF
