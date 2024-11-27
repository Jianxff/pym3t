#include "m3t.hpp"

// pybind
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
namespace py = pybind11;


class ModelWrapper : public Model {
public:
    ModelWrapper(
        const std::string& name,
        const std::string& geometry_path,
        const std::string& region_meta_path = "",
        const std::string& depth_meta_path = "",
        const float unit_in_meter = 1.0f,
        const float shperer_radius = 0.8f
    );
}; // class ModelWrapper


class MultiModalTrackerWrapper : public Tracker {
public:
    // constructor
    MultiModalTrackerWrapper(
        const int image_width,
        const int image_height,
        const Eigen::Matrix3f& K,
        const bool use_region = true,
        const bool use_depth = false,
        const bool use_texture = false,
        const float depth_scale = 1.0f
    );
    // update method
    void step_wrapper(const py::array_t<uint8_t>& color_image, const py::array_t<float>& depth_image);
    // add model
    void add_model_wrapper(const std::shared_ptr<ModelWrapper>& model);
}; // class MultiModalTrackerWrapper

// class RGBTrackerWrapper : public Tracker {
// public:
//     // constructor
//     RGBTrackerWrapper(
//         const int image_width,
//         const int image_height,
//         const Eigen::Matrix3f& K,
//         const bool use_texture = false
//     );
//     // update method
//     void step_forward(const py::array_t<uint8_t>& color_image);
// }; // class RGBTrackerWrapper



// class RGBDTrackerWrapper : public Tracker {
// public:
//     RGBDTrackerWrapper(
//         const int image_width,
//         const int image_height,
//         const Eigen::Matrix3f& K,
//         const float depth_scale,
//         const bool use_texture = false
//     );
//     // update method
//     void step_forward(const py::array_t<uint8_t>& color_image, const py::array_t<float>& depth_image);
// }; // class RGBDTrackerWrapper



class ViewerWrapper : public Viewer {
public:
    ViewerWrapper(const std::shared_ptr<MultiModalTrackerWrapper>& tracker);
    // render color image
    py::array_t<uint8_t> view_color(const bool rgb_format = true);
    // render depth image
    py::array_t<uint8_t> view_depth(const bool rgb_format = true);
    // convertor
    py::array_t<uint8_t> convert_to_numpy(const cv::Mat& image);
}; // class ViewerWrapper