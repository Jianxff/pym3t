#include "pym3t.hpp"


ModelWrapper::ModelWrapper(
    const std::string &name,
    const std::string &geometry_path,
    const std::string &region_meta_path,
    const std::string &depth_meta_path,
    const float unit_in_meter,
    const float shperer_radius
) : Model(name, geometry_path, region_meta_path, depth_meta_path, unit_in_meter, shperer_radius) {}


RGBTrackerWrapper::RGBTrackerWrapper(
    const int image_width,
    const int image_height,
    const Eigen::Matrix3f &K,
    const bool use_texture
) : Tracker(image_width, image_height, K, false, 0.0f, use_texture) {}

void RGBTrackerWrapper::step_forward(const py::array_t<uint8_t> &color_image) {
    // convert to cv::Mat
    if (color_image.ndim() != 3) {
        throw std::runtime_error("RGB color image must have 3 dimensions");
    }
    // get buffer
    py::buffer_info buf = color_image.request();
    cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t *)buf.ptr);
    // step
    step(image);
}



RGBDTrackerWrapper::RGBDTrackerWrapper(
    const int image_width,
    const int image_height,
    const Eigen::Matrix3f &K,
    const float depth_scale,
    const bool use_texture
) : Tracker(image_width, image_height, K, true, depth_scale, use_texture) {}


void RGBDTrackerWrapper::step_forward(const py::array_t<uint8_t> &color_image, const py::array_t<float> &depth_image) {
    // convert to cv::Mat
    if (color_image.ndim() != 3) {
        throw std::runtime_error("RGB color image must have 3 dimensions");
    }
    if (depth_image.ndim() != 2) {
        throw std::runtime_error("Depth image must have 2 dimensions");
    }
    // get buffer
    py::buffer_info buf_color = color_image.request();
    py::buffer_info buf_depth = depth_image.request();
    cv::Mat image(buf_color.shape[0], buf_color.shape[1], CV_8UC3, (uint8_t *)buf_color.ptr);
    cv::Mat depth(buf_depth.shape[0], buf_depth.shape[1], CV_32FC1, (float *)buf_depth.ptr);
    // step
    step(image, depth);
}



ViewerWrapper::ViewerWrapper(const std::shared_ptr<Tracker> &tracker) : Viewer(tracker) {}

py::array_t<uint8_t> ViewerWrapper::convert_to_numpy(const cv::Mat &image) {
    // convert to numpy array
    py::array_t<uint8_t> image_np = py::array_t<uint8_t>(
        {image.rows, image.cols, image.channels()}, 
        {image.cols * image.channels(), image.channels(), 1}, 
        image.data
    );
    return image_np;
}

py::array_t<uint8_t> ViewerWrapper::view_color() {
    // render color image
    cv::Mat image = render_image();
    return convert_to_numpy(image);
}

py::array_t<uint8_t> ViewerWrapper::view_depth() {
    // render depth image
    cv::Mat image = render_depth();
    return convert_to_numpy(image);
}




PYBIND11_MODULE(pym3t, m) {
    m.doc() = "Python binding for M3T";
    m.attr("__version__") = "241120";

    py::class_<ModelWrapper, std::shared_ptr<ModelWrapper>>(m, "Model")
        .def(py::init<
            const std::string&, 
            const std::string&, 
            const std::string&, 
            const std::string&, 
            const float, 
            const float>(),
            py::arg("name"),
            py::arg("geometry_path"),
            py::arg("region_meta_path") = "",
            py::arg("depth_meta_path") = "",
            py::arg("unit_in_meter") = 1.0f,
            py::arg("shperer_radius") = 0.8f
        )
        .def("reset_pose", &ModelWrapper::reset_pose_opencv, 
            py::arg("pose"), "reset pose from 4x4 matrix under OpenCV coordinate")
        .def("reset_pose_cv", &ModelWrapper::reset_pose_opencv, 
            py::arg("pose"), "reset pose from 4x4 matrix under OpenCV coordinate")
        .def("reset_pose_gl", &ModelWrapper::reset_pose_opengl, 
            py::arg("pose"), "reset pose from 4x4 matrix under OpenGL coordinate")
        .def("setup", &ModelWrapper::setup, "setup model")
        .def_property_readonly("name", [](const ModelWrapper &m) { return m.name_; }, "model name")
        .def_property_readonly("pose", &ModelWrapper::get_pose_opencv, "get pose under OpenCV coordinate")
        .def_property_readonly("pose_cv", &ModelWrapper::get_pose_opencv, "get pose under OpenCV coordinate")
        .def_property_readonly("pose_gl", &ModelWrapper::get_pose_opengl, "get pose under OpenGL coordinate");

    py::class_<RGBTrackerWrapper, std::shared_ptr<RGBTrackerWrapper>>(m, "RGBTracker")
        .def(py::init<
            const int, 
            const int, 
            const Eigen::Matrix3f&, 
            const bool>(),
            py::arg("image_width"),
            py::arg("image_height"),
            py::arg("K") = Eigen::Matrix3f::Zero(),
            py::arg("use_texture") = false
        )
        .def("add_model", &RGBTrackerWrapper::add_model, py::arg("model"), "add model")
        .def("step", &RGBTrackerWrapper::step_forward, 
            py::arg("image"), "step with RGB color image")
        .def_property_readonly("K_", [](const RGBTrackerWrapper &t) { return t.K_; }, "camera intrinsic matrix")
        .def_property_readonly("models_", [](const RGBTrackerWrapper &t) {return t.models_;}, "get models dict");    

    py::class_<RGBDTrackerWrapper, std::shared_ptr<RGBDTrackerWrapper>>(m, "RGBDTracker")
        .def(py::init<
            const int, 
            const int, 
            const Eigen::Matrix3f&, 
            const float, 
            const bool>(),
            py::arg("image_width"),
            py::arg("image_height"),
            py::arg("K") = Eigen::Matrix3f::Zero(),
            py::arg("depth_scale") = 1.0f,
            py::arg("use_texture") = false
        )
        .def("add_model", &RGBDTrackerWrapper::add_model, py::arg("model"), "add model")
        .def("step", &RGBDTrackerWrapper::step_forward, py::arg("image"), 
            py::arg("depth"), "step with RGB color and depth images")
        .def_property_readonly("K_", [](const RGBTrackerWrapper &t) { return t.K_; }, "camera intrinsic matrix")
        .def_property_readonly("models_", [](const RGBTrackerWrapper &t) {return t.models_;}, "get models dict");

    py::class_<ViewerWrapper, std::shared_ptr<ViewerWrapper>>(m, "Viewer")
        .def(py::init<std::shared_ptr<Tracker>>(), py::arg("tracker"), "initialize viewer")
        .def("view_color", &ViewerWrapper::view_color, "render color image")
        .def("view_depth", &ViewerWrapper::view_depth, "render depth image");

};