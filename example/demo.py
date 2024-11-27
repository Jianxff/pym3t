# standard library
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
# third party
import numpy as np
import cv2
# m3t
import pym3t

"""
This is a demo script for pym3t.
Demo data is borrowed from the 'SMu1' sequence of HO3D_v3 dataset.
"""
DEMO_DIR = Path(__file__).absolute().parent


def read_image_depth(index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a pair of image and depth data.
    Args:
        index (int): Index of the data.
    Returns:
        Tuple[np.ndarray, np.ndarray]: RGB Image and float32 depth data
    """
    image_path = image_list[index]
    depth_path = depth_list[index]
    # read rgb
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # read depth
    depth = cv2.imread(str(depth_path))
    depth = depth[:, :, 2] + depth[:, :, 1] * 256
    depth = depth.astype(np.float32)
    depth = depth * 0.00012498664727900177 # scale
    
    return (image, depth)


## load demo image data
image_list = sorted(list((DEMO_DIR / 'data' / 'rgb').glob('*.jpg')))
depth_list = sorted(list((DEMO_DIR / 'data' / 'depth').glob('*.png')))

## load model
obj_path = DEMO_DIR / 'data/obj/textured_simple.obj'

## static data
intrinsic = np.array([
    [617.343,   0,    312.42],
    [0,    617.343, 241.42],
    [0,      0,      1   ]
])

pose0 = np.array([
    [0.19550252,  0.98069179, -0.00473666,  0.02659293],
    [0.20227897, -0.0355977,   0.97868073, -0.05623148],
    [0.95961553, -0.19229268, -0.20533276, -0.40142447],
    [0,          0,          0,          1        ]
])

## init M3T
model = pym3t.Model(
    name='mug',
    geometry_path=str(obj_path),
    region_meta_path=str(DEMO_DIR / 'mug.region.meta'),
    depth_meta_path=str(DEMO_DIR / 'mug.depth.meta'),
)

tracker = pym3t.Tracker(
    image_width=640,
    image_height=480,
    K=intrinsic,
    use_region=True,
    use_depth=True,
    use_texture=False,
)
tracker.add_model(model)

viewer = pym3t.Viewer(tracker)

## tracking
model.reset_pose_gl(pose0)
for i in range(len(image_list)):
    image, depth = read_image_depth(i)
    tracker.step(image, depth)

    img = viewer.view_color(rgb_format=False)
    cv2.imshow('img', img)

    img = viewer.view_depth(rgb_format=False)
    cv2.imshow('img2', img)

    q = cv2.waitKey(0)
    if q == ord('q'):
        break