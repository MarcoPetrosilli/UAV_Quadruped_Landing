"""
build_reachable_polytope.py
===========================
OFFLINE computation of the N-step controllable set (backward-reachable set)
for the hybrid MPC landing controller, following the polytopic recursion of
Persson (KTH) / Borrelli-Bemporad-Morari.

Run ONCE before simulation:
    python build_reachable_polytope.py
It writes  reachable_polytope.npz  which the controller loads at startup.

-------------------------------------------------------------------------------
WHY PER-AXIS 2D SETS
-------------------------------------------------------------------------------
The horizontal model [ex,ey,evx,evy] with tilt input is, per axis, a decoupled
double integrator (ex,evx) driven by pitch and (ey,evy) driven by roll. The
4-D controllable set is therefore the Cartesian PRODUCT of two identical 2-D
per-axis sets. Working in 2-D:
  * is exact for the decoupled model (no approximation),
  * avoids the numerical degeneracies of 4-D vertex projection,
  * matches Persson's horizontal/vertical decomposition.

Membership of the full state is then simply: the (ex,evx) pair is in the 2-D
set AND the (ey,evy) pair is in the same 2-D set AND (ez,evz) is in the
vertical set.

-------------------------------------------------------------------------------
RELATIVE FRAME
-------------------------------------------------------------------------------
Everything is built in the platform-relative frame: state = drone-minus-target
position/velocity error, target = origin. The set is therefore independent of
where the (moving) platform is, and is computed once. Online you transform the
current state into this frame and test H x <= h.
-------------------------------------------------------------------------------
"""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def ordered_vertices_2d(H, h):
    """Vertici di un politopo 2D ordinati in senso antiorario (per il poligono)."""
    V = vertices(H, h)
    c = V.mean(axis=0)
    ang = np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0])
    return V[np.argsort(ang)]


def controllable_set_history(A, B, XH, Xh, UH, Uh, SH, Sh, N, tag=""):
    """Come controllable_set, ma restituisce la LISTA [K_0, K_1, ..., K_N]."""
    H_K, h_K = SH.copy(), Sh.copy()
    history = [(H_K.copy(), h_K.copy())]
    for j in range(1, N + 1):
        Hp, hp = pre_set(H_K, h_K, A, B, UH, Uh)
        H_K, h_K = intersect(Hp, hp, XH, Xh)
        history.append((H_K.copy(), h_K.copy()))
    print(f"  [{tag}] storia: {len(history)} set (K_0..K_{N})")
    return history


def plot_nested_sets(history, xlabel, ylabel, title, fname, step=1):
    """Disegna i set annidati; 'step' salta iterazioni se N e' grande."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.cm.viridis
    idxs = list(range(0, len(history), step))
    if idxs[-1] != len(history) - 1:
        idxs.append(len(history) - 1)
    for j in idxs:
        H, h = history[j]
        V = ordered_vertices_2d(H, h)
        Vc = np.vstack([V, V[0]])          # chiudi il poligono
        col = cmap(j / (len(history) - 1))
        ax.plot(Vc[:, 0], Vc[:, 1], color=col, lw=1.2,
                label=f"K_{j}" if j in (0, len(history) - 1) else None)
    # evidenzia il set finale
    Hf, hf = history[-1]
    Vf = ordered_vertices_2d(Hf, hf); Vf = np.vstack([Vf, Vf[0]])
    ax.fill(Vf[:, 0], Vf[:, 1], color=cmap(1.0), alpha=0.10)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, len(history) - 1))
    fig.colorbar(sm, ax=ax, label="iterazione j (passo backward)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    print(f"  salvato {fname}")
    
# ----------------------------------------------------------------------
#  Polytope helpers  (H-representation:  H x <= h)
# ----------------------------------------------------------------------
def box(halfwidths):
    """Axis-aligned box |x_i| <= halfwidths[i]."""
    n = len(halfwidths)
    H = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([halfwidths, halfwidths]).astype(float)
    return H, h


def _interior_point(H, h):
    """Chebyshev centre of {x : H x <= h} (also proves non-emptiness)."""
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    A = np.hstack([H, norms])
    c = np.zeros(H.shape[1] + 1); c[-1] = -1.0
    res = linprog(c, A_ub=A, b_ub=h,
                  bounds=[(None, None)] * H.shape[1] + [(0, None)], method="highs")
    if not res.success or res.x[-1] <= 1e-9:
        raise RuntimeError("polytope empty or lower-dimensional")
    return res.x[:H.shape[1]]


def vertices(H, h):
    interior = _interior_point(H, h)
    hs = HalfspaceIntersection(np.hstack([H, -h.reshape(-1, 1)]), interior)
    return hs.intersections


def hrep(V):
    hull = ConvexHull(V)
    H = hull.equations[:, :-1]
    h = -hull.equations[:, -1]
    nrm = np.linalg.norm(H, axis=1, keepdims=True)
    return H / nrm, h / nrm.flatten()


def pre_set(H_K, h_K, A, B, H_U, h_U):
    """Pre(K) = {x : exists u in U with A x + B u in K}, projected onto x."""
    n, m = A.shape[1], B.shape[1]
    top = np.hstack([H_K @ A, H_K @ B])
    bot = np.hstack([np.zeros((H_U.shape[0], n)), H_U])
    Hz = np.vstack([top, bot])
    hz = np.concatenate([h_K, h_U])
    Vz = vertices(Hz, hz)
    return hrep(Vz[:, :n])          # drop the u-coordinates = projection


def remove_redundant(H, h):
    keep = []
    for i in range(H.shape[0]):
        res = linprog(-H[i], A_ub=np.delete(H, i, 0), b_ub=np.delete(h, i),
                      bounds=[(None, None)] * H.shape[1], method="highs")
        if (not res.success) or (-res.fun > h[i] + 1e-7):
            keep.append(i)
    return H[keep], h[keep]


def intersect(H1, h1, H2, h2):
    return remove_redundant(np.vstack([H1, h1 and H2]), np.concatenate([h1, h2])) \
        if False else remove_redundant(np.vstack([H1, H2]), np.concatenate([h1, h2]))


def controllable_set(A, B, XH, Xh, UH, Uh, SH, Sh, N, tag=""):
    H_K, h_K = SH.copy(), Sh.copy()
    for j in range(1, N + 1):
        Hp, hp = pre_set(H_K, h_K, A, B, UH, Uh)
        H_K, h_K = intersect(Hp, hp, XH, Xh)
    print(f"  [{tag}] N={N}: {H_K.shape[0]} facets")
    return H_K, h_K


# ----------------------------------------------------------------------
#  Build
# ----------------------------------------------------------------------
def main():
    dt = 0.02
    g = 9.8

    # ---------- per-axis HORIZONTAL double integrator ----------
    # state [e, ev] ; input = tilt ; accel = g * tilt
    A_ax = np.array([[1, dt], [0, 1]], float)
    B_ax = np.array([[0], [g * dt]], float)

    a_xy_lim = 0.17           # tilt limit [rad]  (MUST match the MPC constraint)
    v_max = 1.0               # max relative speed used as a state bound
    pos_max = 8.0

    XH, Xh = box([pos_max, v_max])
    UH, Uh = box([a_xy_lim])
    SH, Sh = box([0.10, 0.10])     # terminal "captured" box (rel. pos & vel)
    N_hrz = 80

    print("Horizontal per-axis controllable set:")
    #H_ax, h_ax = controllable_set(A_ax, B_ax, XH, Xh, UH, Uh, SH, Sh, N_hrz, "hrz")
    
    hist_h = controllable_set_history(A_ax, B_ax, XH, Xh, UH, Uh, SH, Sh, N_hrz, "hrz")
    plot_nested_sets(hist_h,
                     "distanza relativa e_x [m]", "velocita relativa e_vx [m/s]",
                     "Set controllabile N-step — asse X (= asse Y)",
                     "reachable_x.png", step=max(1, N_hrz // 15))
    plot_nested_sets(hist_h,
                     "distanza relativa e_y [m]", "velocita relativa e_vy [m/s]",
                     "Set controllabile N-step — asse Y",
                     "reachable_y.png", step=max(1, N_hrz // 15))
                     
    H_ax, h_ax = hist_h[-1]

    # ---------- VERTICAL double integrator ----------
    # state [ez, evz] ; input = az directly
    A_vz = np.array([[1, dt], [0, 1]], float)
    B_vz = np.array([[0], [dt]], float)
    az_lim = 9.0              # MUST match the MPC vertical constraint |az|<=9
    XHv, Xhv = box([pos_max, v_max])
    UHv, Uhv = box([az_lim])
    SHv, Shv = box([0.10, 0.10])
    N_vrt = 80

    print("Vertical controllable set:")
    #H_vz, h_vz = controllable_set(A_vz, B_vz, XHv, Xhv, UHv, Uhv, SHv, Shv, N_vrt, "vrt")
    
    hist_v = controllable_set_history(A_vz, B_vz, XHv, Xhv, UHv, Uhv, SHv, Shv, N_vrt, "vrt")
    plot_nested_sets(hist_v,
                     "distanza relativa e_z [m]", "velocita relativa e_vz [m/s]",
                     "Set controllabile N-step — asse Z",
                     "reachable_z.png", step=max(1, N_vrt // 15))
                     
    H_vz, h_vz = hist_v[-1]

    np.savez("reachable_polytope.npz",
             H_ax=H_ax, h_ax=h_ax,          # per-axis horizontal set (2D)
             H_vz=H_vz, h_vz=h_vz,          # vertical set (2D)
             dt=dt, a_xy_lim=a_xy_lim, az_lim=az_lim, v_max=v_max,
             N_hrz=N_hrz, N_vrt=N_vrt)
    print("\nSaved reachable_polytope.npz")
    print(f"  horizontal per-axis: {H_ax.shape[0]} facets (2D)")
    print(f"  vertical:            {H_vz.shape[0]} facets (2D)")


if __name__ == "__main__":
    main()

