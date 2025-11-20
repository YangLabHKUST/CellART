# CellART
A unified framework for extracting single-cell information from high-resolution spatial transcriptomics

[![pypi](https://img.shields.io/pypi/v/cellart.svg)](https://pypi.python.org/pypi/cellart/)
![GitHub repo
size](https://img.shields.io/github/repo-size/YangLabHKUST/CellART)
![GitHub last
commit](https://img.shields.io/github/last-commit/YangLabHKUST/CellART)
![GitHub
License](https://img.shields.io/github/license/YangLabHKUST/CellART)
![GitHub Repo
stars](https://img.shields.io/github/stars/YangLabHKUST/CellART) ![GitHub
forks](https://img.shields.io/github/forks/YangLabHKUST/CellART)

CellART is a unified framework for extracting single-cell information from high-resolution ST data. The primary objectives are to accurately delineate boundaries for individual cells and further annotate their cell types. By integrating deep neural networks with probabilistic models, CellART leverages multimodal data, including spatial transcriptomics, staining images, and scRNA-seq references, to perform simultaneous cell segmentation and cell type annotation, thereby optimizing the analytical process.

<figure>

<img src="docs/source/method.jpg" style="width:95.0%"
alt="Pipeline" />
</figure>


Visit our [documentation](https://cellart.readthedocs.io/en/latest/) for installation, examples and reproducing the results in our paper.

## Installation

CellART can be installed via two approaches: using pip or directly from the GitHub repository.

### Installing through pip (Recommended)
You can install the stable release of CellART directly from pip. First, ensure you have set up a compatible Python environment using `conda` or any other package manager. Then, execute the following commands:

```shell
$ conda create -n cellart python=3.10
$ conda activate cellart
$ pip install cellart
```

> **_NOTE:_**  Due to differences in GPU models and CUDA versions, you may need to manually reinstall PyTorch and Tensorflow to ensure compatibility. CellART relies heavily on GPU acceleration for efficient processing of large-scale spatial transcriptomics datasets.


### Installing from GitHub

```shell
# Clone the repository
$ git clone https://github.com/YangLabHKUST/CellART.git

# Navigate to the cloned directory
$ cd CellART

# Install the package
$ pip install .
```

## Real data examples

- [Segment and annotate Xenium CRC dataset from raw data](https://cellart.readthedocs.io/en/latest/tutorials/xenium_crc.html)
- [Segment and annotate VisiumHD CRC dataset from raw data](https://cellart.readthedocs.io/en/latest/tutorials/visiumhd_crc.html)


## Reference

<!-- If you find the `CellART` package or any of the source code in this
repository useful for your work, please cite: -->
CellART is currently under review.

## Development

The python package `CellART` is developed and maintained by Yuheng Chen.

Please feel free to contact [Yuheng Chen](mailto:ychenlp@connect.ust.hk), [Prof. Jiashun Xiao](mailto:xiaojsh8@mail.sysu.edu.cn) or [Prof. Can Yang](mailto:macyang@ust.hk) if any inquiries.

