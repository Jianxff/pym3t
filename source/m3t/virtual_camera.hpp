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
        image_list_.emplace(image.clone());
        return true;
    }

    // image update
    bool UpdateImage(bool synchronized = true) override {
        if (image_list_.empty()) {
            if (image_.empty()) return false;
            return true;
        }
        
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
        depth_scale_ = 0.001f; // will be calculate in 'mm' while tracking
        force_realtime_ = force_realtime;
        convert_to_mm_ushort_ = depth_scale * 1000.0f;
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
        cv::Mat depth_meter = image.clone() * convert_to_mm_ushort_;
        depth_meter.convertTo(depth_meter, CV_16U);
        image_list_.emplace(depth_meter);
        return true;
    }

    // image update
    bool UpdateImage(bool synchronized = true) override {
        if (image_list_.empty()) {
            if (image_.empty()) return false;
            return true;
        }
        
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
    float convert_to_mm_ushort_ = 1.0f;
}; // class VirtualDepthCamera


} // namespace m3t