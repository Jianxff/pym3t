#pragma once

/// basic
#include <Eigen/Geometry>
#include <filesystem>
#include <unordered_map>
#include <memory>
#include <string>


/// m3t
#include <m3t/common.h>
// model
#include <m3t/body.h>
#include <m3t/region_model.h>
#include <m3t/depth_model.h>
// modalities
#include <m3t/region_modality.h>
#include <m3t/depth_modality.h>
#include <m3t/texture_modality.h>
// tarcker main
#include <m3t/tracker.h>
#include <m3t/renderer_geometry.h>
#include <link.h>
// renderer
#include <m3t/basic_depth_renderer.h>
#include <m3t/normal_viewer.h>
// camera
#include "virtual_camera.hpp"


class Model {
public:
    // constructor
    Model(
        const std::string& name,
        const std::string& geometry_path,
        const std::string& region_meta_path = "",
        const std::string& depth_meta_path = "",
        const float unit_in_meter = 1.0f,
        const float shperer_radius = 0.8f,
        const Eigen::Matrix4f& geometry2body = Eigen::Matrix4f::Identity()
    );
    // setup method
    void setup();
    // get pose
    const Eigen::Matrix4f get_pose_opencv();
    const Eigen::Matrix4f get_pose_opengl();
    // set pose
    void reset_pose_opencv(const Eigen::Matrix4f& pose);
    void reset_pose_opengl(const Eigen::Matrix4f& pose);

    // variables
    const std::string name_;
    std::shared_ptr<m3t::Body> body_ptr_ = nullptr;
    std::shared_ptr<m3t::RegionModel> region_model_ptr_ = nullptr;
    std::shared_ptr<m3t::DepthModel> depth_model_ptr_ = nullptr;
    std::shared_ptr<m3t::RegionModality> region_modality_ptr_ = nullptr;
    std::shared_ptr<m3t::DepthModality> depth_modality_ptr_ = nullptr;
    std::shared_ptr<m3t::TextureModality> texture_modality_ptr_ = nullptr;
}; // class M3TModel


class Tracker {
public:
    // constructor
    Tracker(
        const int image_width,
        const int image_height,
        const Eigen::Matrix3f& K = Eigen::Matrix3f::Zero(),
        const bool use_region = true,
        const bool use_depth = true,
        const float depth_scale = 1.0f,
        const bool use_texture = false
    );
    // add mode
    void add_model(const std::shared_ptr<Model>& model);
    // update method
    void step(const cv::Mat& color_image, const cv::Mat& depth_image = cv::Mat());
    // delete method
    void terminate();

    int image_width_;
    int image_height_;
    Eigen::Matrix3f K_;
    const bool use_region_ = true;
    const bool use_depth_ = true;
    const bool use_texture_ = false;

    // model map
    std::unordered_map<std::string, std::shared_ptr<Model>> models_;
    // tracker pointer
    std::shared_ptr<m3t::Tracker> tracker_ptr_ = nullptr;
    std::shared_ptr<m3t::RendererGeometry> renderer_geometry_ptr_ = nullptr;
    // depth renderer pointer
    std::shared_ptr<m3t::FocusedBasicDepthRenderer> color_depth_renderer_ptr_ = nullptr;
    std::shared_ptr<m3t::FocusedSilhouetteRenderer> color_silhouette_renderer_ptr_ = nullptr;
    std::shared_ptr<m3t::FocusedBasicDepthRenderer> depth_depth_renderer_ptr_ = nullptr;
    // camera pointer
    std::shared_ptr<m3t::VirtualColorCamera> color_camera_ptr_ = nullptr;
    std::shared_ptr<m3t::VirtualDepthCamera> depth_camera_ptr_ = nullptr;

}; // class Tracker



class Viewer {
public:
    // constructor
    Viewer(const std::shared_ptr<Tracker>& tracker_);
    cv::Mat render_image();
    cv::Mat render_depth();

    const bool use_color_ = false;
    const bool use_depth_ = false;
    std::shared_ptr<Tracker> tracker_;
    std::shared_ptr<m3t::RendererGeometry> renderer_geometry_ptr_ = nullptr;
    std::shared_ptr<m3t::NormalColorViewer> color_viewer_ptr_ = nullptr;
    std::shared_ptr<m3t::NormalDepthViewer> depth_viewer_ptr_ = nullptr;
    std::shared_ptr<m3t::VirtualColorCamera> color_camera_ptr_ = nullptr;
    std::shared_ptr<m3t::VirtualDepthCamera> depth_camera_ptr_ = nullptr;

}; // class Renderer