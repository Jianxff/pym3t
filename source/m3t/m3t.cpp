#include "m3t.hpp"


const Eigen::Matrix4f TRANSFORM_OPENCV_OPENGL = 
    (Eigen::Matrix4f() << 
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1 ).finished();


Model::Model(
    const std::string& name,
    const std::string& geometry_path,
    const std::string& region_meta_path,
    const std::string& depth_meta_path,
    const float unit_in_meter,
    const float shperer_radius,
    const Eigen::Matrix4f& geometry2body)
: name_(name) 
{ 
    m3t::Transform3fA geometry2body_pose(geometry2body);
    // make body geometry
    body_ptr_ = std::make_shared<m3t::Body>(name, geometry_path, 
        unit_in_meter, true, true, geometry2body_pose);
    // reset meta path if not given
    auto _region_meta_path = region_meta_path;
    auto _depth_meta_path = depth_meta_path;
    if (_region_meta_path.empty()) {
        _region_meta_path = geometry_path + ".region.meta";
    }
    if (_depth_meta_path.empty()) {
        _depth_meta_path = geometry_path + ".depth.meta";
    }
    // make model
    region_model_ptr_ = std::make_shared<m3t::RegionModel>(
        name, body_ptr_, _region_meta_path);
    depth_model_ptr_ = std::make_shared<m3t::DepthModel>(
        name, body_ptr_, _depth_meta_path);
}


void Model::setup() {
    body_ptr_->SetUp();
    region_model_ptr_->SetUp();
    depth_model_ptr_->SetUp();
}


const Eigen::Matrix4f Model::get_pose_opencv() {
    return body_ptr_->body2world_pose().matrix();
}

const Eigen::Matrix4f Model::get_pose_opengl() {
    Eigen::Matrix4f pose_cv = body_ptr_->body2world_pose().matrix();
    return TRANSFORM_OPENCV_OPENGL * pose_cv;
}

void Model::reset_pose_opencv(const Eigen::Matrix4f& pose) {
    m3t::Transform3fA body2world_pose(pose);
    body_ptr_->set_body2world_pose(body2world_pose);
}

void Model::reset_pose_opengl(const Eigen::Matrix4f& pose) {
    Eigen::Matrix4f pose_cv = pose;
    m3t::Transform3fA body2world_pose(TRANSFORM_OPENCV_OPENGL * pose_cv);
    body_ptr_->set_body2world_pose(body2world_pose);
}



Tracker::Tracker(
    const int image_width,
    const int image_height,
    const Eigen::Matrix3f& K,
    const bool use_depth,
    const float depth_scale,
    const bool use_texture
) : image_width_(image_width), 
    image_height_(image_height), 
    K_(K), 
    use_depth_(use_depth), 
    use_texture_(use_texture) 
{
    // make tracker
    tracker_ptr_ = std::make_shared<m3t::Tracker>("tracker");
    renderer_geometry_ptr_ = std::make_shared<m3t::RendererGeometry>("renderer_geometry");

    // set intrinsic
    float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    m3t::Intrinsics intrinsics{fx, fy, cx, cy, image_width, image_height};
    
    // set color camera
    color_camera_ptr_ = std::make_shared<m3t::VirtualColorCamera>("color_camera", intrinsics);
    color_depth_renderer_ptr_ = std::make_shared<m3t::FocusedBasicDepthRenderer>(
        "color_depth_renderer", renderer_geometry_ptr_, color_camera_ptr_);
    if (use_texture_) {
        color_silhouette_renderer_ptr_ = std::make_shared<m3t::FocusedSilhouetteRenderer>(
            "color_silhouette_renderer", renderer_geometry_ptr_, color_camera_ptr_);
    }

    // set depth camera
    if (use_depth_) {
        depth_camera_ptr_ = std::make_shared<m3t::VirtualDepthCamera>("depth_camera", intrinsics, depth_scale);
        depth_depth_renderer_ptr_ = std::make_shared<m3t::FocusedBasicDepthRenderer>(
            "depth_depth_renderer", renderer_geometry_ptr_, depth_camera_ptr_);
    }

}


void Tracker::add_model(const std::shared_ptr<Model>& model) {
    // add model
    auto ret = models_.try_emplace(model->name_, model);
    if (!ret.second) {
        throw std::runtime_error("model '" + model->name_ + "' already exists");
    }
    // assign ptrs
    renderer_geometry_ptr_->AddBody(model->body_ptr_);
    auto link_ptr = std::make_shared<m3t::Link>(model->name_, model->body_ptr_);

    // region modality
    color_depth_renderer_ptr_->AddReferencedBody(model->body_ptr_);
    auto region_modality_ptr = std::make_shared<m3t::RegionModality>(
        model->name_, model->body_ptr_, color_camera_ptr_, model->region_model_ptr_);
    link_ptr->AddModality(region_modality_ptr);
    
    // depth modality
    if (use_depth_) {
        depth_depth_renderer_ptr_->AddReferencedBody(model->body_ptr_);
        auto depth_modality_ptr = std::make_shared<m3t::DepthModality>(
            model->name_, model->body_ptr_, depth_camera_ptr_, model->depth_model_ptr_);
        region_modality_ptr->MeasureOcclusions(depth_camera_ptr_);
        depth_modality_ptr->MeasureOcclusions();
        // add to link
        link_ptr->AddModality(depth_modality_ptr);
    }
    
    // texture modality
    if (use_texture_) {
        color_silhouette_renderer_ptr_->AddReferencedBody(model->body_ptr_);
        auto texture_modality_ptr = std::make_shared<m3t::TextureModality>(
            model->name_, model->body_ptr_, color_camera_ptr_, color_silhouette_renderer_ptr_);
        if (use_depth_) {
            texture_modality_ptr->MeasureOcclusions(depth_camera_ptr_);
        }
        // add to link
        link_ptr->AddModality(texture_modality_ptr);
    }

    // optimizer
    auto optimizer_ptr = std::make_shared<m3t::Optimizer>(model->name_, link_ptr);

    // add to tracker
    tracker_ptr_->AddOptimizer(optimizer_ptr);
    // setup tracker
    if (!tracker_ptr_->SetUp()) {
        throw std::runtime_error("failed to setup tracker");
    }
}


void Tracker::step(const cv::Mat& color_image, const cv::Mat& depth_image) {
    // set color image
    color_camera_ptr_->PushImage(color_image);
    if (use_depth_) depth_camera_ptr_->PushImage(depth_image);

    // forward tracking
    tracker_ptr_->RunTrackerOneStep();
}



Viewer::Viewer(const std::shared_ptr<Tracker>& tracker)
 : tracker_(tracker), use_depth_(tracker->use_depth_)
{
    // set renderer
    renderer_geometry_ptr_ = tracker_->renderer_geometry_ptr_;

    // setup color camera
    color_camera_ptr_ = tracker_->color_camera_ptr_;
    color_viewer_ptr_ = std::make_shared<m3t::NormalColorViewer>(
        "color_viewer", color_camera_ptr_, renderer_geometry_ptr_);
    color_camera_ptr_->SetUp();
    
    // setup depth camera
    if (use_depth_) {
        depth_camera_ptr_ = tracker_->depth_camera_ptr_;
        depth_viewer_ptr_ = std::make_shared<m3t::NormalDepthViewer>(
            "depth_viewer", depth_camera_ptr_, renderer_geometry_ptr_);
        depth_camera_ptr_->SetUp();
    }
}


cv::Mat Viewer::render_image() {
    // render color image
    if(!color_camera_ptr_->UpdateImage(true)) return cv::Mat();
    color_viewer_ptr_->UpdateViewer(0);
    return color_viewer_ptr_->viewer_image();
}

cv::Mat Viewer::render_depth() {
    // render depth image
    if (use_depth_) {
        if(!depth_camera_ptr_->UpdateImage(true)) return cv::Mat();
        depth_viewer_ptr_->UpdateViewer(0);
        return depth_viewer_ptr_->viewer_image();
    }
    throw std::runtime_error("depth modality is not available");
}