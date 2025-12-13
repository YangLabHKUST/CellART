# Figure 2
cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumHumanLungPreview/cell-polygons.geojson ./xenium/proseg-cell-polygons.geojson
cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumHumanLungPreview_NMF_test_boundary/new_segmentation_mask.npy ./xenium/cellart_segmentation_mask.npy
cp /import/home2/yhchenmath/Dataset/CellSeg/Xenium_human_lung/preprocessed/nuclei_cellpose.tif ./xenium/nuclei_cellpose.tif
cp /import/home3/yhchenmath/baysor_out/xenium_human_lung/baysor_segmentation_mask.tif ./xenium/baysor_segmentation_mask.tif
cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumHumanLungPreview/10X_cell_segmentation_mask.npy ./xenium/10X_cell_segmentation_mask.npy

cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumhdMouseBrain_NMF/new_segmentation_mask.npy ./hd/cellart_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumMouseBrain/spot_id_map.npy ./hd/spot_id_map.npy
cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumhdMouseBrain_NMF/epoch_2000/cell_deconv.h5ad ./hd/cell_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumMouseBrainNMF/bin2cell_segmentation_mask.npy ./hd/bin2cell_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumMouseBrainNMF/segmentation_mask.npy ./hd/stardist_segmentation_mask.npy
cp /import/home3/yhchenmath/Code/SVTBenchmarking/figure_refined/all_figure_data/nucleus_corr.csv ./hd/nucleus_corr.csv
cp /import/home3/yhchenmath/Code/SVTBenchmarking/figure_refined/all_figure_data/overlapping_corr.csv ./hd/overlapping_corr.csv

cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/cell_deconv_xenium.h5ad ./annotation/cell_deconv_xenium.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/xenium_scvi.h5ad ./annotation/xenium_scvi.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/xenium_tangram.h5ad ./annotation/xenium_tangram.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/rctd_xenium.h5ad ./annotation/rctd_xenium.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/cell_deconv_visiumHD.h5ad ./annotation/cell_deconv_visiumHD.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/visiumhd_scvi.h5ad ./annotation/visiumhd_scvi.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/visiumhd_tangram.h5ad ./annotation/visiumhd_tangram.h5ad
cp /import/home2/yhchenmath/Code/SVTBenchmarking/annotation_benchmarking_pair/nucl_result/rctd_visiumhd.h5ad ./annotation/rctd_visiumhd.h5ad

# Figure 3
cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumhdMouseBrain_Annotation_Long/epoch_550/cell_deconv.h5ad ./hd_cell_deconv.h5ad
cp /import/home2/yhchenmath/Code/SVT/LOG/XeniumMouseBrain_Annotation_with_feature/epoch_250/cell_deconv.h5ad ./xenium_cell_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumhdMouseBrain_Annotation_Long/new_segmentation_mask.npy ./hd_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/SVT/LOG/XeniumMouseBrain_Annotation_with_feature/new_segmentation_mask.npy ./xenium_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/LOG/MerfishMouseBrain_Annotation/epoch_600/cell_deconv.h5ad ./merfish_cell_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/StereoseqAnnotationNoPatch/epoch_850/cell_deconv.h5ad ./stereoseq_cell_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/MerfishMouseBrain_Annotation/new_segmentation_mask.npy ./merfish_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/LOG/StereoseqAnnotationNoPatch/new_segmentation_mask.npy ./stereoseq_segmentation_mask.npy

cp /import/home3/yhchenmath/Code/SVT/Data/MerfishMouseBrain/vizgen_scvi.h5ad ./vizgen_scvi.h5ad
cp /import/home3/yhchenmath/Code/ucs/data/vizgen_mouse_brain/cell_vizgen_mask.npy ./cell_vizgen_mask.npy

cp /import/home3/yhchenmath/Dataset/DeconvSeg/stereoseq_mouse_brain/resized_dapi.npy ./stereoseq_dapi.npy

cp /home/share/yqzeng/data/stereoseq_mouse_brain/bin50.h5ad ./stereoseq_bin50.h5ad
cp /home/share/yqzeng/data/stereoseq_mouse_brain/stereoseq_mouse_brain_rctd_results.csv ./stereoseq_rctd_results.csv

cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumMouseBrain/10X_scvi.h5ad ./xenium_10X_scvi.h5ad

cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumMouseBrain/proseg_cell-polygons.geojson ./proseg-cell-polygons.geojson
cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumMouseBrain/proseg_scvi.h5ad ./proseg_scvi.h5ad

cp /import/home3/yhchenmath/baysor_out/xenium_mouse_brain_baysor_segmentation_mask.npy ./xenium_baysor_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumMouseBrain/baysor_scvi.h5ad ./baysor_scvi.h5ad

cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumhdMouseBrain_Annotation_Long/rctd.h5ad ./hd_rctd.h5ad
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumMouseBrainNMF/annotated_adata_bin2cell_scvi.h5ad ./annotated_adata_bin2cell_scvi.h5ad
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumMouseBrainNMF/annotated_adata_stardist_scvi.h5ad ./annotated_adata_stardist_scvi.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumhdMouseBrain_Annotation_Long/bin50_annotated.h5ad ./bin50_annotated.h5ad
cp /import/home3/yhchenmath/Dataset/DeconvSeg/stereoseq_mouse_brain/ensembl_ids.txt ./ensembl_ids.txt

# Figure 4
cp /home/share/yhchen/niche/adata_breast_cancer_rep1.h5ad ./adata_breast_cancer_rep1.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancer_Rep2/epoch_200/cell_deconv.h5ad ./adata_breast_cancer_rep2.h5ad

cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancer_Annotation_with_feature/new_segmentation_mask.npy ./cellart_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/Data/XeniumBreastCancer/segmentation_mask.npy ./nuclei_segmentation_mask.npy
cp /import/home2/yhchenmath/Log/CellSeg/BIDCell_breast_cancer/standard_result/epoch_1_step_4000_connected.tif ./bidcell_segmentation_mask.tif
cp /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_breast_cancer/10X_cell_mask.tif ./10X_cell_segmentation_mask.tif
cp /import/home3/yhchenmath/baysor_out/xenium_breast_cancer/baysor_segmentation_mask.tif ./baysor_segmentation_mask.tif
cp /import/home2/yhchenmath/Dataset/CellSeg/TestSeg/preprocessed/nuclei.tif ./cellpose_nuclei.tif

cp /import/home2/yhchenmath/Code/TripletBenchmarking/figure_2/breast_cancer_cells/annotating_cells/10X_scvi.h5ad ./10X_scvi.h5ad
cp /import/home2/yhchenmath/Code/TripletBenchmarking/figure_2/breast_cancer_cells/annotating_cells/10X_tangram.h5ad ./10X_tangram.h5ad
cp /import/home2/yhchenmath/Code/TripletBenchmarking/figure_2/breast_cancer_cells/annotating_cells/10X_rctd.h5ad ./10X_rctd.h5ad

cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancer_Annotation_with_feature/epoch_100/cell_deconv.h5ad 
cp /home/share/yhchen/data_st/breast_cancer/filtered_sc.h5ad ./filtered_sc.h5ad

cp /import/home2/yhchenmath/Code/ucs/paper_data/downstream_xenium_breast_cancer/scVI_output/Cell_10X/annotated_adata_st_with_ecc.h5ad ./10X_scvi_annotated_adata_st_with_ecc.h5ad
cp /import/home2/yhchenmath/Code/ucs/paper_data/downstream_xenium_breast_cancer/scVI_output/BIDCell/annotated_adata_st_with_ecc.h5ad ./bidcell_scvi_annotated_adata_st_with_ecc.h5ad
cp /import/home2/yhchenmath/Code/ucs/paper_data/downstream_xenium_breast_cancer/scVI_output/Baysor/annotated_adata_st_with_ecc.h5ad ./baysor_scvi_annotated_adata_st_with_ecc.h5ad

cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancerFullOldModelLong/epoch_120/cell_deconv.h5ad ./cellart_subtype_cell_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancerFullOldModelLong/new_segmentation_mask.npy  ./cellart_subtype_segmentation_mask.npy

cp /import/home2/yhchenmath/Code/SVT/data/XeniumBreastCancerFullExample/segmentation_mask.npy ./cellart_subtype_nuclei.npy

cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancer_Annotation_with_feature/cyto.h5ad ./cyto.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/XeniumBreastCancer_Annotation_with_feature/nuclei.h5ad ./nuclei.h5ad

# Figure 5
cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumHD_P2_CRC_Annotation/epoch_400/cell_deconv.h5ad ./hd_cellart_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/Xenium_P2_CRC_Annotation/epoch_200/cell_deconv.h5ad ./xenium_cellart_deconv.h5ad
cp /import/home2/yhchenmath/Code/Triplet/LOG/VisiumHD_P2_CRC_Annotation/new_segmentation_mask.npy ./hd_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumHD_P2_CRC/segmentation_mask.npy ./hd_nuclei_segmentation_mask.npy
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumHD_P2_CRC/spot_id_map.npy ./hd_spot_id_map.npy

cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumHD_P2_CRC/cdata_expended_scvi.h5ad ./bin2cell_scvi.h5ad
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumHD_P2_CRC/cdata_scvi.h5ad ./stardist_scvi.h5ad

cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumHD_P2_CRC/DeconvolutionResults_P2CRC.csv ./rctd_deconv_results.csv
cp /import/home3/yhchenmath/Dataset/DeconvSeg/CRC/scRNA_ref/adata_sc_p2.h5ad ./adata_sc_p2.h5ad
cp /import/home2/yhchenmath/Code/Triplet/Data/VisiumHD_P2_CRC/filtered_gene_names.txt ./filtered_gene_names.txt

cp /import/home2/yhchenmath/Code/SVTBenchmarking/figure_refined/summary_de.csv ./summary_de.csv

cp /import/home2/yhchenmath/Code/SVTBenchmarking/figure_refined/go_svt.csv ./go_cellart.csv
cp /import/home2/yhchenmath/Code/SVTBenchmarking/figure_refined/go_rctd.csv ./go_rctd.csv

# Figure 6
cp /import/home2/yhchenmath/Code/SVTBenchmarking/figure_refined/adata_svt.h5ad ./cellart_crc_adata.h5ad