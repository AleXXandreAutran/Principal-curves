"""Create an HTML animation of principal-curve iterations on a 3D dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from principal_curves_demo.curves import principal_curve
from principal_curves_demo.datasets import generate_datasets_3d


def main() -> None:
    dataset_name = "helix"  # Try: double_helix, sine_3d, u_shape_3d, two_gaussians_3d
    datasets = generate_datasets_3d(n_samples=1200, random_state=0)
    X, y = datasets[dataset_name]

    n_nodes = 55
    n_iter = 50
    bandwidth = 3.0
    smoothness = 0.35

    result = principal_curve(X, n_nodes=n_nodes, n_iter=n_iter, bandwidth=bandwidth, smoothness=smoothness)
    history = result.history

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=6, cmap="tab10", alpha=0.35)
    line, = ax.plot([], [], [], "k-", linewidth=2.5, label="principal curve")
    nodes = ax.scatter([], [], [], c="red", s=22, marker="x", label="nodes")

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = maxs - mins
    pad = 0.08
    ax.set_xlim(mins[0] - pad * span[0], maxs[0] + pad * span[0])
    ax.set_ylim(mins[1] - pad * span[1], maxs[1] + pad * span[1])
    ax.set_zlim(mins[2] - pad * span[2], maxs[2] + pad * span[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(loc="upper right")

    def update(frame: int):
        curve = history[frame]
        line.set_data(curve[:, 0], curve[:, 1])
        line.set_3d_properties(curve[:, 2])
        nodes._offsets3d = (curve[:, 0], curve[:, 1], curve[:, 2])
        ax.set_title(f"{dataset_name} — iteration {frame}/{len(history)-1}")
        return line, nodes

    animation = FuncAnimation(fig, update, frames=len(history), interval=180, blit=False)
    output = Path("outputs/principal_curve_3d_animation.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(animation.to_jshtml(), encoding="utf-8")
    print(f"Animation written to {output}")


if __name__ == "__main__":
    main()
