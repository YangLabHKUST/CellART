from setuptools import Command, find_packages, setup

__lib_name__ = "cellart"
__lib_version__ = "0.0.65"
__description__ = "CellART: a unified framework for extracting single-cell information from high-resolution spatial transcriptomics"
__url__ = "https://github.com/YangLabHKUST/cellart"
__author__ = "Yuheng Chen"
__author_email__ = "ychenlp@connect.ust.hk"

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    zip_safe = False,
    include_package_data = True,
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown", 
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(),
    install_requires=[
        "wandb==0.20.0",
        "tensorflow==2.19.0",
        "anndata==0.11.4",
        "bin2cell==0.3.3",
        "distinctipy==1.3.4",
        "einops==0.8.1",
        "matplotlib==3.7.5",
        "numpy==1.26.0",
        "opencv_python==4.10.0.84",
        "opencv_python_headless==4.8.0.76",
        "pandas==2.2.3",
        "scanpy==1.11.2",
        "scipy==1.13",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "tqdm==4.65.0",
    ],
)