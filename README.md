# Principal Curves Demo

This repository contains a small implementation of a **simple principal-curve approximation** and several examples in 2D and 3D.

The goal is to visualize the idea that a principal curve can act as a **high-density skeleton** of a point cloud. 
The implementation is intentionally simple and is meant for experimentation, visualization, and discussion.

## What the algorithm does

Given a point cloud `X`, the algorithm estimates one global smooth curve through the central part of the data:

1. **PCA initialization**: initialize ordered curve nodes along the first principal component.
2. **Assignment**: assign each data point to its closest curve node.
3. **Local averaging**: move every node toward a weighted local average of the data points.
4. **Smoothing**: smooth the polyline by moving internal nodes toward the average of their neighbors.
5. **Iteration**: repeat the assignment, local averaging, and smoothing steps until the curve stabilizes.

This gives a simple approximation of a principal curve.

## Repository structure

```text
principal-curves-demo/
├── README.md
├── requirements.txt
├── src/
│   └── principal_curves_demo/
│       ├── __init__.py
│       ├── curves.py          # principal-curve algorithm
│       ├── datasets.py        # 2D and 3D synthetic datasets
│       └── plotting.py        # plotting utilities
├── examples/
│   ├── run_2d_examples.py     # grid of 2D examples
│   ├── run_3d_examples.py     # grid of 3D examples
│   └── animate_3d_curve.py    # HTML animation of the iterations
└── outputs/
    └── .gitkeep
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/principal-curves-demo.git
cd principal-curves-demo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Because the package uses a `src/` layout, run examples with:

```bash
PYTHONPATH=src python examples/run_2d_examples.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python examples/run_2d_examples.py
```

## Run the examples

### 2D examples

```bash
PYTHONPATH=src python examples/run_2d_examples.py
```

This creates:

```text
outputs/principal_curves_2d.png
```

### 3D examples

```bash
PYTHONPATH=src python examples/run_3d_examples.py
```

This creates:

```text
outputs/principal_curves_3d.png
```

### 3D animation

```bash
PYTHONPATH=src python examples/animate_3d_curve.py
```

This creates:

```text
outputs/principal_curve_3d_animation.html
```

Open this HTML file in a browser to inspect the evolution of the curve over the iterations.

## Parameters

The main parameters are:

| Parameter | Meaning |
|---|---|
| `n_nodes` | Number of nodes used to discretize the curve |
| `n_iter` | Number of refinement iterations |
| `bandwidth` | Width of the local averaging window along the curve |
| `smoothness` | Strength of the smoothing step, between 0 and 1 |

Typical values:

```python
n_nodes = 55
n_iter = 60
bandwidth = 3.0
smoothness = 0.35
```

## Example use in Python

```python
from principal_curves_demo.datasets import generate_datasets_2d
from principal_curves_demo.curves import principal_curve

X, y = generate_datasets_2d()["two_moons"]
result = principal_curve(X, n_nodes=55, n_iter=60)
curve = result.curve
history = result.history
```

## Limitations

This implementation learns **one global curve**. On data made of separated clusters, the curve may connect clusters artificially. This is expected: the method is designed to summarize a global geometry, not to perform clustering directly.

For clustering, one can combine principal curves with density-based criteria, local unimodality tests, or density-valley separators.

## Scientific context

Principal curves are related to the idea of representing data by a smooth central skeleton. Density-ridge methods provide a more rigorous differential framework: a density ridge is defined using the density, its gradient, and its Hessian, and can represent curves or higher-dimensional principal manifolds.
