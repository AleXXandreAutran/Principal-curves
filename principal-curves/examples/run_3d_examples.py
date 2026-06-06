"""Run the 3D principal-curve demonstration."""

from principal_curves_demo.datasets import generate_datasets_3d
from principal_curves_demo.plotting import plot_principal_curves_3d


def main() -> None:
    datasets = generate_datasets_3d(n_samples=1200, random_state=0)
    fig = plot_principal_curves_3d(
        datasets,
        n_nodes=55,
        n_iter=60,
        bandwidth=3.0,
        smoothness=0.35,
        output_path="outputs/principal_curves_3d.png",
    )
    fig.show()


if __name__ == "__main__":
    main()
