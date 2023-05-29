#pragma once

// std
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

namespace cppsam {

class ONNXExecutorInterface {
public:
    virtual ~ONNXExecutorInterface() = default;

    virtual bool EncodeImage(cv::Mat input_image) = 0;
    virtual cv::Mat predict(const std::vector<cv::Point2f>& positive, const std::vector<cv::Point2f>& negative) = 0;
};

}
