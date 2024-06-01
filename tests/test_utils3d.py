from utils3d import utils3d
import numpy as np
from utils3d.utils3d import pctodepthimage
import PIL

def test_pctodepthimage():
    """Testing by converting a point cloud from the KITTI dataset to a depth image."""
    
    path = "pointclouds/um_000000.pcd"
    height = 512
    width = 1382
    
    extrinsics = np.array([[7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
                      [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02],
                      [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01]])

    intrinsics = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02],
                      [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02],
                      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    
    depth_image = pctodepthimage(path, extrinsics, intrinsics, height, width, 0.15)
    
    assert type(depth_image) == PIL.Image.Image, "Not transforming properly!"
