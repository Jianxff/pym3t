#pragma once

#include <m3t/camera.h>
#include <opencv2/opencv.hpp>
#include <queue>


namespace m3t {

class VirtualColorCamera : public ColorCamera {
public:
    VirtualColorCamera(const std::string &name, const Intrinsics &intrinsics, const bool force_realtime = false)
        : ColorCamera(name) 
    {
        intrinsics_ = intrinsics;
        force_realtime_ = force_realtime;
    }

    // setup method
    bool SetUp() override {
        set_up_ = true;
        return true;
    }

    // image add
    bool PushImage(const cv::Mat& image) {
        if(image.empty()) {
            printf("[warn] got empty image\n");
            return false;
        }
        image_list_.push(image.clone());
    }

    // image update
    bool UpdateImage(bool synchronized = true) override {
        if (image_list_.empty()) return false;
        
        image_ = image_list_.front();
        image_list_.pop();
        // drop frames
        if (force_realtime_) {
            while (image_list_.size() > 0) {
                image_ = image_list_.front();
                image_list_.pop();
            }
        }
        return true;
    }

private:
    std::queue<cv::Mat> image_list_;
    bool force_realtime_ = false;
}; // class VirtualColorCamera



class VirtualDepthCamera : public DepthCamera {
public:
    VirtualDepthCamera(const std::string& name, const Intrinsics& intrinsics, 
        const float depth_scale = 1.0f, const bool force_realtime = false)
     : DepthCamera(name) 
    {
        intrinsics_ = intrinsics;
        depth_scale_ = depth_scale;
        force_realtime_ = force_realtime;
    }

    // setup method
    bool SetUp() override {
        set_up_ = true;
        return true;
    }

    // image add
    bool PushImage(const cv::Mat& image) {
        if(image.empty()) {
            printf("[warn] got empty image\n");
            return false;
        }
        image_list_.push(image.clone());
    }

    // image update
    bool UpdateImage(bool synchronized = true) override {
        if (image_list_.empty()) return false;
        
        image_ = image_list_.front();
        image_list_.pop();
        // drop frames
        if (force_realtime_) {
            while (image_list_.size() > 0) {
                image_ = image_list_.front();
                image_list_.pop();
            }
        }
        return true;
    }

private:
    std::queue<cv::Mat> image_list_;
    bool force_realtime_ = false;
}; // class VirtualDepthCamera


} // namespace m3t