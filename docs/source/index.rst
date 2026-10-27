CellART
===================================

CellART is a unified framework for extracting single-cell information from high-resolution ST data. The
primary objectives are to accurately delineate boundaries for individual cells and further annotate their
cell types.  By integrating deep neural networks with probabilistic models, CellART leverages multimodal data, including
spatial transcriptomics, staining images, and scRNA-seq references, to perform simultaneous cell segmentation and cell type annotation, thereby optimizing the analytical process.

.. image:: method.jpg
   :width: 800
   :align: center

.. note::

   This project is under active development.

CellART Manuscript
------------------

Chen, Y., Liu, Y., Wang, Z. et al. **CellART: a unified framework for extracting single-cell information from high-resolution spatial transcriptomics.** *Nature Computational Science* (2026). https://doi.org/10.1038/s43588-026-01054-1

CellART Installation & Usage & Demo
------------------

.. toctree::
   :maxdepth: 2

   installation 
   usage
   tutorials/visiumhd_crc
   tutorials/xenium_crc

Reproduce CellART Results
------------------

.. toctree::
   :maxdepth: 1

   tutorials/figure_2
   tutorials/figure_3
   tutorials/figure_4
   tutorials/figure_5
   tutorials/figure_6
