# pym3t

Python binding for [M3T: A Multi-body Multi-modality Multi-camera 3D Tracker](https://github.com/DLR-RM/3DObjectTracking/tree/master/M3T).



## Modification
#### 1. Limitation
The original `M3T` framework considers multi-body multi-modality multi-camera tracking. 
Currently, `pym3t` only focuses on multi-rigid-object multi-modality tracking with one camera.

#### 2. Removed Modules
- Special Camera
    The original modules for `Azure Kinect` camera and `RealSense` camera are removed. You will need to provide RGB image (and depth image) manually in `OpenCV::Mat` format.
- Publisher and Subscriber
    The two modules are not important to the core tracking pipeline.
- Detector
    The original `StaticDetector` and `ManualDetector` are removed, as there are ways to do it better. The `Refiner` is not used as well (reserved).
- Image Viewer
    The original `ColorImageViewer` and `DepthImageViewer` are removed. As the input images are provided manually, the visualization of them can be easily implemented through `opencv-python`. 

#### 3. Detached Normal Viewer
The normal viewers are detached from the original tracker. You can manually create one if needed.


## Installation
Make sure that you have installed the following packages:
- GLEW
- glfw3
- Eigen3
- OpenCV 4 with Contrib Modules
- OpenMP (may have already installed on your system)

Your compilation environment should support `C++ 17`.

#### Dependency Instruction
- [method 1] Through system path
  ```
  # build tools
  $ sudo apt install build-essential cmake
  
  # libraries
  $ sudo apt install libglew-dev libglfw3-dev libeigen3-dev

  # BUILD opencv with extra contrib modules from source code:
    - https://github.com/opencv/opencv
    - https://github.com/opencv/opencv_contrib
  ```

- [method 2] Under conda env
  ```
  $ conda create -n ${your_env_name} python=3.9
  $ conda activate ${your_env_name}
  $ conda install cmake eigen glfw glew libopencv
  ```

- Other dependencies for `demo.py`
  ```
  pip install numpy==1.26.0
  pip install opencv-python
  ```

#### Package Installation
```
cd ${repo_root}
pip install .
```


## Demo
*The demo data is from `SMu1` sequence of [HO3D_v3](https://github.com/shreyashampali/ho3d)*

```
cd ${repo_root}/example
python demo.py
```



## Interface
Follow `source/pym3t/pym3t.cpp`.


## Note
This algorithm is for object tracking only, without global pose estimation. In consequence, an initial pose (4 by 4 matrix under opencv coordinate) must be provided before you start tracking.
