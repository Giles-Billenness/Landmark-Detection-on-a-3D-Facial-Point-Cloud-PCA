# Landmark Detection on a 3D Facial Point Cloud using PCA

This project explores different methods for detecting the nose tip from 3D facial landmark data. It uses the AFLW2000-3D dataset and implements and evaluates three distinct approaches:

1. **Max Z-Coordinate Heuristic:** A simple baseline method that assumes the nose tip is the point with the maximum z-value.
2. **Principal Component Analysis (PCA):** A more robust method that identifies the nose tip by projecting landmarks onto the principal component representing depth.
3. **PCA with Depth Correction:** An enhanced PCA approach that includes a heuristic to ensure the depth component consistently points outward from the face, improving accuracy in varied head poses.

The Jupyter Notebook `task2.ipynb` contains the implementation, visualization, and evaluation of these methods.

## Dataset

The project uses the [AFLW2000-3D dataset](https://www.kaggle.com/datasets/mohamedadlyi/aflw2000-3d). This dataset provides 3D facial landmarks and corresponding 2D images.

## Key Findings

The evaluation demonstrates the progression of accuracy:

| Method                     | Mean Distance Error | Max Distance Error | Accuracy |
| -------------------------- | ------------------- | ------------------ | -------- |
| Max Z                      | 48.2591             | 216.5532           | 0.447    |
| PCA                        | 22.4034             | 318.898            | 0.8605   |
| PCA with Depth Correction  | 0.5972              | 203.396            | 0.988    |

The PCA method with depth correction significantly outperforms the simpler heuristics, achieving high accuracy in identifying the nose tip.

## Setup and Usage

### Environment Setup

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

This project uses `uv` for Python environment and package management.

2. **Install uv:**
    Follow the official installation instructions for `uv` from [docs.astral.sh/uv/getting-started/](https://docs.astral.sh/uv/getting-started/installation/).

3. **Create a Python virtual environment or easily use uv:**

    ```bash
    uv sync
    ```

4. **Install dependencies separately if required:**
    The core dependencies are:

    ```
    ipykernel
    ipywidgets
    matplotlib
    numpy
    scikit-learn
    ```

    More details are available in the `pyproject.toml` file, which can be used to create a different virtual environment.

    An extracted packages list `requirements.txt` is included, generated using `uv export --no-emit-workspace --no-dev --no-header --no-hashes --output-file requirements.txt`

## Project Structure

* `task2.ipynb`: The main Jupyter Notebook containing the code, analysis, and visualizations.
* `AFLW2000/`: (Assumed directory) Contains the AFLW2000-3D dataset files (not included in this repo, must be downloaded separately).
* `README.md`: This file.
* `requirements.txt` and `pyproject.toml` for `uv` environment setup.

## Challenges and Future Work

* **Dataset Quality:** Performance can be affected by the quality and consistency of landmark annotations in the dataset.
* **Robustness:** While PCA with depth correction is robust, extreme poses or occlusions can still pose challenges.
* **Future Directions:**
  * Exploring machine learning models (potentially multi-modal, using both 3D points and image data) for improved accuracy and robustness.
  * Using PCA components as features for ML models.
  * Investigating performance on higher-quality or more diverse datasets.
