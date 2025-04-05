import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# Define helper functions
def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    return b

def pressure_poisson_periodic(p, dx, dy):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        p[-1, :] = p[-2, :]
        p[0, :] = p[1, :]
    return p

def plot_snapshot(u, v, X, Y):
    fig = plt.figure(figsize=(8, 5), dpi=100)
    plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()

def animate_velocity(U, V, X, Y, interval=50):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    q = ax.quiver(X[::5, ::5], Y[::5, ::5], U[1][::5, ::5], V[1][::5, ::5],scale=20)
    
    def update(i):
        q.set_UVC(U[i][::5, ::5], V[i][::5, ::5])
        ax.set_title(f"Time: {i}")
        return q,

    ani = animation.FuncAnimation(fig, update, frames=len(U), blit=False, interval=interval)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()


# === Main simulation ===

nx = 41
ny = 41
nt = 10
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1
nu = 0.1
F = 1
dt = 0.01

u = np.zeros((ny, nx))
un = np.zeros((ny, nx))
U = []

v = np.zeros((ny, nx))
vn = np.zeros((ny, nx))
V = []

p = np.ones((ny, nx))
b = np.zeros((ny, nx))

udiff = 1
stepcount = 0

while udiff > 0.001:
    U.append(u.copy())
    V.append(v.copy())
    un = u.copy()
    vn = v.copy()

    b = build_up_b(rho, dt, dx, dy, u, v)
    p = pressure_poisson_periodic(p, dx, dy)

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                    dt / (2 * rho * dx) *
                    (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                    nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt)

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) *
                    (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    keys = {"zero": 0, "one": -1}
    for m in keys:
        k = keys[m]
        u[1:-1, k] = (un[1:-1, 0+k] - un[1:-1, 0+k] * dt / dx *
                     (un[1:-1, 0+k] - un[1:-1, -1+k]) -
                      vn[1:-1, 0+k] * dt / dy *
                     (un[1:-1, 0+k] - un[0:-2, 0+k]) -
                      dt / (2 * rho * dx) *
                     (p[1:-1, 1+k] - p[1:-1, -1+k]) +
                      nu * (dt / dx**2 *
                     (un[1:-1, 1+k] - 2 * un[1:-1, 0+k] + un[1:-1, -1+k]) +
                      dt / dy**2 *
                     (un[2:, 0+k] - 2 * un[1:-1, 0+k] + un[0:-2, 0+k])) + F * dt)

        v[1:-1, k] = (vn[1:-1, 0+k] - un[1:-1, 0+k] * dt / dx *
                     (vn[1:-1, 0+k] - vn[1:-1, -1+k]) -
                      vn[1:-1, 0+k] * dt / dy *
                     (vn[1:-1, 0+k] - vn[0:-2, 0+k]) -
                      dt / (2 * rho * dy) *
                     (p[2:, 0+k] - p[0:-2, 0+k]) +
                      nu * (dt / dx**2 *
                     (vn[1:-1, 1+k] - 2 * vn[1:-1, 0+k] + vn[1:-1, -1+k]) +
                      dt / dy**2 *
                     (vn[2:, 0+k] - 2 * vn[1:-1, 0+k] + vn[0:-2, 0+k])))

        u[k, :] = 0
        v[k, :] = 0

    udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
    stepcount += 1

# plot and animation
plot_snapshot(U[-1], V[-1], X, Y)
animate_velocity(U, V, X, Y)


