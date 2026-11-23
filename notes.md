Input: A folder created with run\_pipeline.py from plausibilitree

Process

## Point cloud preparation (plot level)

Each feature/label pair represents one entire plot.

### Normalization

The user can specify if the point cloud should be normalized. Options include rescaling each axis to \[-1,1\] or \[0,1\]. After normalization has been applied, the original bounding box (including geospatial information if available) needs to be preserved. 

### Subsampling

1. User specifies how the point clouds should be downsampled.  
2. The options are a voxel downsample (equivalent to [PDAL voxeldownsize with mode center](https://pdal.io/en/2.9.2/stages/filters.voxeldownsize.html#filters-voxeldownsize)) or a farthest points subsample.  
   1. If the user selects voxel downsample, they provide the voxel resolution (VOXEL\_RES)  
   2. If the user selects farther points subsample, they provide the number of points to subsample too (N\_FPS)  
3. Optionally, the user can also optionally provide a value for the minimum number of points a point cloud must have (MIN\_POINTS\_DROP). If the point cloud has fewer points, further processing stops and it is excluded from the outputs.  
4. Furthermore, the user can also provide a value for the minimum number of points a point cloud must have (MIN\_POINTS\_PAD). If the point cloud has fewer points, then points are randomly duplicated with a small jitter in their position until the point cloud reaches MIN\_POINTS\_PAD (PAD\_JITTER). Dropping occurs  
5. Furthermore, The user can also provide an optional parameter that is the minimum number of points a cloud must have after subsampling (MIN\_POINTS\_PAD)

Some notes:

* The drop check must occur before padding.  
* If MIN\_POINTS\_DROP \> MIN\_POINTS\_PAD, return an error.

## Feature and label extraction

1. Features include the subsampled/normalized point cloud and any remaining necessary info, such as bounding box  
2. The labels are generated automatically from the plot info JSON (e.g. NUM\_TREES, BASAL\_AREA, VOL\_TREES, etc.).   
3. The user can optionally specify if these values should be normalized by the plot area.  
4. Features and labels should be exported as separate files with a shared name so that they are portable to any deep learning libraries.  
5. The user can optionally specify to include any generated CHMs or DTMs. The goal is that, during model development, users can specify which relevant labels they want to train on.

