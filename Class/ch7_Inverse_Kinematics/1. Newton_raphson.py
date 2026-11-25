"""
Newton-Raphson iteration for 1-DOF inverse kinematics (Modern Robotics, Fig. 6.7)

Goal: find theta_d such that  x_d - f(theta_d) = 0
Update:  theta_{k+1} = theta_k + (df/dtheta(theta_k))^{-1} * (x_d - f(theta_k))
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Problem setup (edit freely) ----------------
# Pick one of the presets below:
#   "gauss"    : the Fig. 6.7 curve, converges in ~4 iterations (quadratic convergence)
#   "easy"     : f = theta^2, starts near the positive root and converges in ~3 iterations
#   "zigzag"   : f = atan(theta); Newton overshoots back and forth, ~10 iterations
#   "singular" : 1-link arm x = cos(theta) reaching full extension x_d = 1.
#                df/dtheta -> 0 at the solution (singularity), so convergence
#                drops to linear: each step only halves theta, ~10 iterations
#   "hard"     : alias for "zigzag"
#
# Select from the command line, e.g.
#   uv run .\Newton_raphson.py PROBLEM="easy"
#   uv run .\Newton_raphson.py PROBLEM="hard"
import sys

PROBLEM = "easy"     # default when no PROBLEM=... argument is given
for arg in sys.argv[1:]:
    key, sep, val = arg.partition("=")
    if sep and key.strip().upper() == "PROBLEM":
        PROBLEM = val.strip().strip("\"'").lower()

PRESETS = {
    "gauss": dict(f=lambda th: 1.8 - 1.8 * np.exp(-(th - 0.2) ** 2 / 4),
                  x_d=1.5, theta0=1.5, xlim=(-3.5, 5.0), ylim=(-1.0, 1.9)),
    "easy": dict(f=lambda th: th ** 2,
                 x_d=1.0, theta0=0.8, xlim=(-2.0, 2.0), ylim=(-3.0, 1.5)),
    "zigzag": dict(f=lambda th: np.arctan(th),
                   x_d=0.0, theta0=1.39, xlim=(-3.0, 3.0), ylim=(-1.6, 1.6)),
    "singular": dict(f=lambda th: np.cos(th),
                     x_d=1.0, theta0=1.5, xlim=(-1.0, 2.5), ylim=(-0.8, 1.6)),
}
PRESETS["hard"] = PRESETS["zigzag"]
if PROBLEM not in PRESETS:
    sys.exit(f"Unknown PROBLEM={PROBLEM!r}. Choose one of: {', '.join(PRESETS)}")
print(f"PROBLEM = {PROBLEM}")
P = PRESETS[PROBLEM]
x_d = P["x_d"]       # desired end-effector position
theta0 = P["theta0"] # initial guess
tol = 1e-6           # convergence tolerance on |x_d - f(theta)|
max_iter = 30
SAVE_GIF = False     # set True to save "newton_raphson.gif" (needs pillow)
STEP = True          # True: wait for a key/click in the figure before each update


def f(theta):
    """Forward kinematics (any smooth 1-D function works)."""
    return P["f"](theta)


def df(theta, h=1e-6):
    """df/dtheta via central difference (replace with analytic if you like)."""
    return (f(theta + h) - f(theta - h)) / (2 * h)


# ---------------- Newton-Raphson iterations ----------------
thetas = [theta0]
print(f"{'k':>3} {'theta_k':>12} {'x_d - f':>14} {'df/dtheta':>12} {'dtheta':>12}")
for k in range(max_iter):
    th = thetas[-1]
    err = x_d - f(th)
    slope = df(th)
    if abs(err) < tol:
        print(f"{k:3d} {th:12.6f} {err:14.3e}   converged")
        break
    if abs(slope) < 1e-12:
        print("Derivative ~ 0: Newton step undefined, stopping.")
        break
    dth = err / slope                     # Δθ = (∂f/∂θ)^-1 (x_d - f(θ))
    print(f"{k:3d} {th:12.6f} {err:14.3e} {slope:12.6f} {dth:12.6f}")
    thetas.append(th + dth)

# ---------------- Visualization ----------------
grid = np.linspace(*P["xlim"], 600)
g = x_d - f(grid)                         # curve plotted in the figure

fig, ax = plt.subplots(figsize=(11, 6.5))
theta_d = thetas[-1]                      # converged root (for labeling)
ZOOM_WHEN_STEP_BELOW = 0.35               # show zoom inset when |Δθ| is small


def draw_step(a, i, labels=True, fs=11, short=False):
    """Draw curve + tangent + θ_k, θ_{k+1}, slope and Δθ annotations on axes a."""
    th = thetas[i]
    gi = x_d - f(th)
    s = df(th)                            # ∂f/∂θ(θ_k)
    has_next = i + 1 < len(thetas)
    th_next = thetas[i + 1] if has_next else th

    xx = np.linspace(*a.get_xlim(), 800)  # resample so zoomed curves stay accurate
    a.plot(xx, x_d - f(xx), "k", lw=2)
    a.axhline(0, color="k", lw=1)

    # θ_d on the axis
    a.plot(theta_d, 0, "o", mfc="white", mec="green", ms=8, zorder=5)

    # vertical dashed line θ_k -> curve
    a.plot([th, th], [0, gi], "b--", lw=1)
    a.plot(th, gi, "bo", ms=7, zorder=6)

    if has_next:
        # tangent line of (x_d - f) at θ_k: slope = -∂f/∂θ
        span = abs(th_next - th)
        d = np.sign(th_next - th) or 1.0  # step direction (left or right)
        pad = 0.1 * (a.get_xlim()[1] - a.get_xlim()[0]) / 8
        t = np.linspace(th - d * (0.6 * span + 2 * pad), th_next + d * (0.3 * span + pad), 50)
        a.plot(t, gi - s * (t - th), color="red", lw=1.3)
        a.plot(th_next, 0, "rs", ms=7, zorder=6)

    if not labels:
        return th, th_next, gi, s

    # labels for θ_k, θ_{k+1}, θ_d below the axis
    ylo = a.get_ylim()[0] if a.get_ylim()[0] < 0 else -0.6
    dy = 0.09 * (a.get_ylim()[1] - a.get_ylim()[0])
    a.text(th, -dy, rf"$\theta_{{{i}}}$", color="blue", ha="center", va="top", fontsize=fs + 2)
    if has_next:
        close = abs(th_next - th) < 0.35 * (a.get_xlim()[1] - a.get_xlim()[0]) / 8
        a.text(th_next, -dy * (2.0 if close else 1.0), rf"$\theta_{{{i+1}}}$",
               color="red", ha="center", va="top", fontsize=fs + 2)
    if abs(theta_d - th_next) > 0.15 or not has_next:
        a.text(theta_d, dy * 0.6, r"$\theta_d$", color="green", ha="center",
               va="bottom", fontsize=fs + 1)

    # slope label next to the point
    left = th < 0 and a is ax            # point left of the θ=0 axis: label on its left
    a.annotate(rf"slope $= -\partial f/\partial\theta\,(\theta_{{{i}}}) = {-s:.3f}$",
               xy=(th, gi), xytext=(-15 if left else 15, 12), textcoords="offset points",
               ha="right" if left else "left", fontsize=fs, color="red",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Δθ arrow and formula
    if has_next:
        side = 1 if (gi < 0 and not short) else -1   # put arrow on the side away from the point
        ya = side * dy * 2.2
        a.annotate("", xy=(th_next, ya), xytext=(th, ya),
                   arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))
        dtxt = (rf"$\Delta\theta = {th_next - th:.4f}$" if short else
               rf"$\Delta\theta = (\partial f/\partial\theta\,(\theta_{{{i}}}))^{{-1}}"
               rf"\,(x_d - f(\theta_{{{i}}})) = {gi:.3f}\,/\,{s:.3f} = {th_next - th:.4f}$")
        a.text((th + th_next) / 2, ya + side * dy * 0.4, dtxt, color="purple", ha="center",
               va="bottom" if side > 0 else "top", fontsize=fs)
    return th, th_next, gi, s


def draw(i):
    ax.clear()
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(*P["ylim"])
    ax.axvline(0, color="k", lw=1)

    # past iterates in gray
    for j, thj in enumerate(thetas[:i]):
        ax.plot(thj, x_d - f(thj), "o", color="lightgray", ms=5)

    zoom = i + 1 < len(thetas) and abs(thetas[i + 1] - thetas[i]) < ZOOM_WHEN_STEP_BELOW
    th, th_next, gi, s = draw_step(ax, i, labels=not zoom)
    if zoom:   # keep only the full Δθ formula on the main plot
        ax.text(0.99, 0.04, rf"$\Delta\theta = (\partial f/\partial\theta\,(\theta_{{{i}}}))^{{-1}}"
                rf"\,(x_d - f(\theta_{{{i}}})) = {gi:.2e}\,/\,{s:.3f} = {th_next - th:.2e}$",
                color="purple", ha="right", va="bottom", fontsize=11,
                transform=ax.transAxes)

    # horizontal dashed line to the vertical axis with x_d - f(θ_k) label
    ax.plot([0, th], [gi, gi], "b--", lw=1)
    ax.text(0.08 if th < 0 else -0.08, gi + 0.03, rf"$x_d - f(\theta_{{{i}}}) = {gi:.3g}$",
            color="blue", ha="left" if th < 0 else "right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # zoom inset when the step becomes too small to read
    if zoom:
        ins = ax.inset_axes([0.66, 0.45, 0.32, 0.5])
        w = max(abs(th_next - th), 1e-4)
        xs = (min(th, th_next) - 0.6 * w, max(th, th_next) + 0.6 * w)
        ins.set_xlim(*xs)
        ys = sorted([0, gi])
        hgt = max(abs(gi), 1e-6)
        ins.set_ylim(ys[0] - 0.8 * hgt, ys[1] + 0.4 * hgt)
        draw_step(ins, i, labels=True, fs=8, short=True)
        ins.set_title("zoom", fontsize=9)
        ins.tick_params(labelsize=7)
        ax.indicate_inset_zoom(ins, edgecolor="gray")

    if i + 1 < len(thetas):
        ax.set_title(rf"Iteration {i}:  $\theta_{{{i}}}$ = {th:.4f}  →  "
                     rf"$\theta_{{{i+1}}}$ = {th_next:.4f}", fontsize=13)
    else:
        ax.set_title(rf"Converged:  $\theta_d$ ≈ {th:.6f},  "
                     rf"$|x_d - f(\theta)|$ = {abs(gi):.1e}", fontsize=13)

    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$x_d - f(\theta)$", fontsize=12)
    ax.grid(alpha=0.3)


if SAVE_GIF:
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, draw, frames=len(thetas), interval=1200)
    anim.save("newton_raphson.gif", writer="pillow")
    print("Saved newton_raphson.gif")
else:
    for i in range(len(thetas)):
        draw(i)
        if STEP:
            fig.canvas.draw_idle()
            print(f"[iteration {i}] press any key (or click) in the figure to continue")
            plt.waitforbuttonpress()      # stop here until key/click
            if not plt.fignum_exists(fig.number):
                break                     # window closed -> quit
        else:
            plt.pause(1.2)                # auto-play
    plt.show()