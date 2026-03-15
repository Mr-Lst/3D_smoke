import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.sparse import kron, eye, csr_matrix, bmat, diags
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from numba import njit
import time

@njit
def comb_numba(n, k):
    if k < 0 or k > n:
        return 0
    res = 1
    for i in range(1, min(k, n - k) + 1):
        res = res * (n - i + 1) // i
    return res

@njit
def bernstein(i, n, x):
    return comb_numba(n, i) * (x**i) * ((1.0 - x)**(n - i))

@njit
def bernstein_deriv(i, n, x):
    if i < 0 or i > n:
        return 0.0
    val = 0.0
    if i - 1 >= 0:
        val += bernstein(i - 1, n - 1, x)
    if i <= n - 1:
        val -= bernstein(i, n - 1, x)
    return n * val

@njit
def bernstein_deriv2(i, n, x):
    if i < 0 or i > n:
        return 0.0
    val = 0.0
    if i - 2 >= 0:
        val += bernstein(i - 2, n - 2, x)
    if i - 1 >= 0 and i - 1 <= n - 2:
        val -= 2 * bernstein(i - 1, n - 2, x)
    if i <= n - 2:
        val += bernstein(i, n - 2, x)
    return n * (n - 1) * val

def build_collocation_matrix(pts, N, deriv=0):
    n_pts = len(pts)
    B = np.zeros((n_pts, N+1))
    for i, x in enumerate(pts):
        for k in range(N+1):
            if deriv == 0:
                B[i, k] = bernstein(k, N, x)
            elif deriv == 1:
                B[i, k] = bernstein_deriv(k, N, x)
            else:
                B[i, k] = bernstein_deriv2(k, N, x)
    return B

def apply_boundary_conditions(F, J, Bx, By, Bz, Bt, Nx, Ny, Nz, Nt, n_vars):
    nx, ny, nz, nt = Bx.shape[0], By.shape[0], Bz.shape[0], Bt.shape[0]
    offsets = [0, n_vars, 2*n_vars]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for it in range(nt):
                    if ix == 0 or ix == nx-1 or iy == 0 or iy == ny-1 or iz == 0 or iz == nz-1:
                        for comp in range(3):
                            row = offsets[comp] + (ix * ny * nz * nt + iy * nz * nt + iz * nt + it)
                            J[row, :] = 0
                            for i in range(Nx+1):
                                for j in range(Ny+1):
                                    for k in range(Nz+1):
                                        for l in range(Nt+1):
                                            col = offsets[comp] + (i * (Ny+1)*(Nz+1)*(Nt+1) +
                                                                   j * (Nz+1)*(Nt+1) +
                                                                   k * (Nt+1) + l)
                                            J[row, col] = Bx[ix, i] * By[iy, j] * Bz[iz, k] * Bt[it, l]
                            F[row] = 0.0
    return F, J

def apply_dynamic_heat_source(F, J, Bx, By, Bz, Bt, Nx, Ny, Nz, Nt, T_cold, T_hot, n_vars, freq=5.0):
    nx, ny, nz, nt = Bx.shape[0], By.shape[0], Bz.shape[0], Bt.shape[0]
    ix = np.argmin(np.abs(np.linspace(0, 1, nx) - 0.5))
    iy = np.argmin(np.abs(np.linspace(0, 1, ny) - 0.5))
    iz = 0
    offsets_T = 4 * n_vars
    for it in range(nt):
        t_val = it / (nt - 1)
        dynamic_T = T_cold + (T_hot - T_cold) * 0.5 * (1 + np.sin(2 * np.pi * freq * t_val))
        row = offsets_T + (ix * ny * nz * nt + iy * nz * nt + iz * nt + it)
        J[row, :] = 0
        for i in range(Nx+1):
            for j in range(Ny+1):
                for k in range(Nz+1):
                    for l in range(Nt+1):
                        col = offsets_T + (i * (Ny+1)*(Nz+1)*(Nt+1) +
                                           j * (Nz+1)*(Nt+1) +
                                           k * (Nt+1) + l)
                        J[row, col] = Bx[ix, i] * By[iy, j] * Bz[iz, k] * Bt[it, l]
        F[row] = dynamic_T
    return F, J

def apply_pressure_constraint(J, n_vars):
    offsets_p = 3 * n_vars
    last_row = offsets_p + n_vars - 1
    J[last_row, :] = 0
    J[last_row, offsets_p : offsets_p + n_vars] = 1.0 / n_vars
    return J

def build_system(U, Bx, By, Bz, Bt, Bx1, By1, Bz1, Bt1, Bx2, By2, Bz2,
                 mu, rho, g, beta, T_cold, alpha, Nx, Ny, Nz, Nt):
    n_vars = (Nx+1)*(Ny+1)*(Nz+1)*(Nt+1)
    Bx_s = csr_matrix(Bx); By_s = csr_matrix(By); Bz_s = csr_matrix(Bz); Bt_s = csr_matrix(Bt)
    Bx1_s = csr_matrix(Bx1); By1_s = csr_matrix(By1); Bz1_s = csr_matrix(Bz1); Bt1_s = csr_matrix(Bt1)
    Bx2_s = csr_matrix(Bx2); By2_s = csr_matrix(By2); Bz2_s = csr_matrix(Bz2)
    Ix = eye(Nx+1); Iy = eye(Ny+1); Iz = eye(Nz+1); It = eye(Nt+1)
    Dt = rho * kron(Bt1_s, kron(Iz, kron(Iy, Ix)))
    Dx = kron(Bt_s, kron(Iz, kron(Iy, Bx1_s)))
    Dy = kron(Bt_s, kron(Iz, kron(By1_s, Ix)))
    Dz = kron(Bt_s, kron(Bz1_s, kron(Iy, Ix)))
    Lap = kron(Bt_s, kron(Iz, kron(Iy, Bx2_s))) + \
          kron(Bt_s, kron(Iz, kron(By2_s, Ix))) + \
          kron(Bt_s, kron(Bz2_s, kron(Iy, Ix)))
    vx = U[0:n_vars]; vy = U[n_vars:2*n_vars]; vz = U[2*n_vars:3*n_vars]
    T = U[4*n_vars:5*n_vars]
    Vx = diags(vx); Vy = diags(vy); Vz = diags(vz)
    Convection = rho * (Vx @ Dx + Vy @ Dy + Vz @ Dz)
    A = Dt + Convection - mu * Lap
    A_T = Dt / rho + (Vx @ Dx + Vy @ Dy + Vz @ Dz) - alpha * Lap
    T_weight = -rho * g * beta * kron(Bt_s, kron(Iz, kron(Iy, Ix)))
    J = bmat([
        [A,   None, None, Dx,   None],
        [None, A,   None, Dy,   None],
        [None, None, A,   Dz,   T_weight],
        [Dx,   Dy,   Dz,  None, None],
        [None, None, None, None, A_T]
    ], format='csr')
    F = J @ U
    return F, J

if __name__ == "__main__":
    print("=== 3D Smoke Simulation with Pulsating Source (Final) ===")
    Nx = int(input("Polynomial degree in x (e.g., 4): "))
    Ny = int(input("Polynomial degree in y (e.g., 4): "))
    Nz = int(input("Polynomial degree in z (e.g., 4): "))
    Nt = int(input("Polynomial degree in time (e.g., 4): "))
    mu = float(input("Viscosity mu (e.g., 0.01): "))
    alpha = float(input("Thermal diffusivity alpha (e.g., 0.001): "))
    beta = float(input("Thermal expansion beta (e.g., 0.01): "))
    g = float(input("Gravity g (e.g., 9.81): "))
    T_cold = float(input("Wall temperature T_cold (e.g., 300): "))
    T_hot = float(input("Source temperature T_hot (e.g., 350): "))
    freq = float(input("Pulse frequency (e.g., 5.0): "))
    rho = 1.0
    x_pts = 0.5 * (1 - np.cos(np.linspace(0, np.pi, Nx+1)))
    y_pts = 0.5 * (1 - np.cos(np.linspace(0, np.pi, Ny+1)))
    z_pts = 0.5 * (1 - np.cos(np.linspace(0, np.pi, Nz+1)))
    t_pts = np.linspace(0, 1, Nt+1)
    print("Building basis matrices...")
    Bx  = build_collocation_matrix(x_pts, Nx, 0)
    Bx1 = build_collocation_matrix(x_pts, Nx, 1)
    Bx2 = build_collocation_matrix(x_pts, Nx, 2)
    By  = build_collocation_matrix(y_pts, Ny, 0)
    By1 = build_collocation_matrix(y_pts, Ny, 1)
    By2 = build_collocation_matrix(y_pts, Ny, 2)
    Bz  = build_collocation_matrix(z_pts, Nz, 0)
    Bz1 = build_collocation_matrix(z_pts, Nz, 1)
    Bz2 = build_collocation_matrix(z_pts, Nz, 2)
    Bt  = build_collocation_matrix(t_pts, Nt, 0)
    Bt1 = build_collocation_matrix(t_pts, Nt, 1)
    n_vars = (Nx+1)*(Ny+1)*(Nz+1)*(Nt+1)
    U = np.zeros(5 * n_vars)
    print("Starting Newton iterations...")
    max_iter = 10
    tol = 1e-6
    for it in range(max_iter):
        F, J = build_system(U, Bx, By, Bz, Bt, Bx1, By1, Bz1, Bt1, Bx2, By2, Bz2,
                            mu, rho, g, beta, T_cold, alpha, Nx, Ny, Nz, Nt)
        F, J = apply_boundary_conditions(F, J, Bx, By, Bz, Bt, Nx, Ny, Nz, Nt, n_vars)
        F, J = apply_dynamic_heat_source(F, J, Bx, By, Bz, Bt, Nx, Ny, Nz, Nt, T_cold, T_hot, n_vars, freq)
        J = apply_pressure_constraint(J, n_vars)
        offsets_p = 3 * n_vars
        last_row = offsets_p + n_vars - 1
        F[last_row] = np.mean(U[offsets_p : offsets_p + n_vars])
        normF = np.linalg.norm(F)
        print(f"Iter {it+1}: norm(F) = {normF:.2e}")
        if normF < tol:
            print("✅ Converged.")
            break
        try:
            diag_J = J.diagonal().copy()
            diag_J[diag_J == 0] = 1.0
            M = diags(1.0 / diag_J)
            def preconditioner(r):
                return M @ r
            P = LinearOperator(J.shape, matvec=preconditioner)
            dU, info = gmres(J, -F, M=P, tol=1e-6, maxiter=500)
            if info != 0:
                print("⚠️ GMRES didn't converge, using spsolve...")
                dU = spsolve(J, -F)
        except Exception as e:
            print("❌ Linear solve failed:", e)
            break
        U += dU
        print(f"   norm(update) = {np.linalg.norm(dU):.2e}")
    vx_c = U[0:n_vars].reshape((Nx+1, Ny+1, Nz+1, Nt+1))
    vy_c = U[n_vars:2*n_vars].reshape((Nx+1, Ny+1, Nz+1, Nt+1))
    vz_c = U[2*n_vars:3*n_vars].reshape((Nx+1, Ny+1, Nz+1, Nt+1))
    p_c  = U[3*n_vars:4*n_vars].reshape((Nx+1, Ny+1, Nz+1, Nt+1))
    T_c  = U[4*n_vars:5*n_vars].reshape((Nx+1, Ny+1, Nz+1, Nt+1))
    t_vals = np.linspace(0, 1, 100)
    T_mid = []
    for t in t_vals:
        val = 0.0
        for i in range(Nx+1):
            for j in range(Ny+1):
                for k in range(Nz+1):
                    for l in range(Nt+1):
                        val += T_c[i,j,k,l] * bernstein(i, Nx, 0.5) * bernstein(j, Ny, 0.5) * bernstein(k, Nz, 0.5) * bernstein(l, Nt, t)
        T_mid.append(val)
    plt.figure(figsize=(10,5))
    plt.plot(t_vals, T_mid, 'r-', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Temperature at center')
    plt.title('Temperature evolution at center')
    plt.grid(True)
    plt.show()
    T_slice = T_c[:, Ny//2, :, Nt]
    x_vals = np.linspace(0, 1, Nx+1)
    z_vals = np.linspace(0, 1, Nz+1)
    X, Z = np.meshgrid(x_vals, z_vals)
    plt.figure(figsize=(8,6))
    plt.contourf(X, Z, T_slice.T, levels=20, cmap='magma')
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(f'Temperature slice at y=0.5, t=1')
    plt.show()
    print("🏁 Done.")
