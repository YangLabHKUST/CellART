import numpy as np
from spatialdata_io import xenium
from skimage.measure import regionprops
from shapely.geometry import Polygon
from skimage.measure import find_contours
import geopandas as gpd
import spatialdata as sd
import pandas as pd

def append_visiumhd_boundary(segmentation_mask, spot_id_map, sdata, shape_key, new_shape_key, celltype = None):
    spot_id_map_flat = spot_id_map.flatten()
    segmentation_flat = segmentation_mask.flatten()

    segmentation_flat = segmentation_flat[spot_id_map_flat > 0]
    spot_id_map_flat = spot_id_map_flat[spot_id_map_flat > 0]

    adata = sdata.tables['square_002um'].copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs["spot_id"] = np.arange(adata.shape[0]) + 1
    # Set adata.obs index column to spot_id and keep the original index
    adata.obs['original_index'] = adata.obs.index
    # spot id column to string
    adata.obs['spot_id'] = adata.obs['spot_id'].astype(str)
    adata.obs = adata.obs.set_index('spot_id')

    origin_index = adata.obs['original_index'].values
    adata_re = adata[spot_id_map_flat.astype(str).tolist()]

    adata_re.obs["seg"] = segmentation_flat
    adata_re.obs.set_index('original_index', inplace=True)
    adata_re = adata_re[origin_index]
    sdata.tables[f'square_002um_{new_shape_key}'] = adata_re
    sdata.tables[f'square_002um_{new_shape_key}'].obs["seg"] = sdata.tables[f'square_002um_{new_shape_key}'].obs["seg"].astype(
        int)

    sdata.shapes[shape_key][new_shape_key] = \
    sdata.tables[f'square_002um_{new_shape_key}'].obs["seg"].astype('category').values
    sdata.shapes[new_shape_key] = \
    sdata.shapes[shape_key][
        sdata.shapes[shape_key][new_shape_key] != 0]

    if celltype is not None:
        celltype.index = celltype.index.astype("int32")
        sdata.shapes[new_shape_key]['celltype'] = sdata.shapes[new_shape_key][new_shape_key].map(celltype)


def append_xenium_boundary(segmentation_mask, sdata, append_name, celltype = None):
    segmentation_mask = segmentation_mask.astype("int32")
    cell_polygons = {}
    props = regionprops(segmentation_mask)

    for prop in props:
        cell_id = prop.label
        minr, minc, maxr, maxc = prop.bbox
        
        # submask = segmentation_mask[minr - 5:maxr + 5, minc - 5:maxc + 5] == cell_id
        # Sometimes minr - 5 or minc - 5 is less than 0, so we need to check the range
        minr = max(0, minr - 5)
        minc = max(0, minc - 5)
        maxr = min(segmentation_mask.shape[0], maxr + 5)
        maxc = min(segmentation_mask.shape[1], maxc + 5)
        submask = segmentation_mask[minr:maxr, minc:maxc] == cell_id
        # print(submask.shape, minr, maxr, minc, maxc)
        contours = find_contours(submask, level=0.5)
        largest = max(contours, key=lambda x: x.shape[0])
        polygon = Polygon([(minc - 5 + x, minr - 5 + y) for y, x in largest])
        cell_polygons[cell_id] = polygon

    polygons = []
    cell_ids = []
    for cell_id, polygon in cell_polygons.items():
        polygons.append(polygon)
        cell_ids.append(cell_id)
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': polygons}, crs="EPSG:4326")
    gdf = gdf.set_index("cell_id")
    if celltype is not None:
        celltype.index = celltype.index.astype("int32")
        intersection = gdf.index.intersection(celltype.index)
        gdf = gdf.loc[intersection]
        gdf["celltype"] = celltype.loc[intersection].values

    if sdata is not None:
        boundaries = sd.models.ShapesModel.parse(gdf)
        sdata.shapes[append_name] = boundaries
        seg_transformation = sd.transformations.get_transformation(sdata.shapes["nucleus_boundaries"])
        sd.transformations.set_transformation(sdata.shapes[append_name], seg_transformation,
                                              to_coordinate_system="global")
        
        if celltype is not None:
            center_df = sdata.shapes[append_name].centroid
            point_df = pd.DataFrame()
            point_df['x'] = center_df.geometry.x
            point_df['y'] = center_df.geometry.y
            point_df["celltype"] = sdata.shapes[append_name].celltype
            point = sd.models.PointsModel.parse(point_df, coordinates = {'x': 'x', 'y': 'y'})
            sdata.points[append_name + "_centroid"] = point
            sd.transformations.set_transformation(sdata.points[append_name + "_centroid"], seg_transformation,
                                                  to_coordinate_system="global")
        
    else:
        print("Not sdata provided")
        return gdf
