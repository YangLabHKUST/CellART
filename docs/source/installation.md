
## Installation

CellART can be installed via two approaches: using pip or directly from the GitHub repository.

### Installing through pip
You can install the stable release of CellART directly from pip. First, ensure you have set up a compatible Python environment using `conda` or any other package manager. Then, execute the following commands:

```shell
$ conda create -n cellart python=3.10
$ conda activate cellart
$ pip install cellart
```

### Installing from GitHub
To install the latest version of CellART, you can clone the repository from GitHub and install it manually.

```shell
# Clone the repository
$ git clone https://github.com/YangLabHKUST/CellART.git

# Navigate to the cloned directory
$ cd CellART

# Install the package
$ pip install .
```

### Note
CellART relies on deep learning frameworks such as PyTorch and TensorFlow for its core operations. Before installing CellART, ensure that the required version of PyTorch or TensorFlow is compatible with your system's hardware (e.g., CUDA support for GPU acceleration).