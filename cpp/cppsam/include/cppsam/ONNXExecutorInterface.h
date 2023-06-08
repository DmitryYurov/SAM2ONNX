#pragma once

// std
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

namespace cppsam {

class ONNXExecutorInterface {
public:
    virtual ~ONNXExecutorInterface() = default;

    // returns the expected size of the input image
    virtual cv::Size input_size() const = 0;

    // returns the expected size of the output mask
    virtual cv::Size output_size() const = 0;

    virtual bool encode_image(cv::Mat input_image) = 0;

    /*
    * points: Coordinates of sparse input prompts, corresponding to both point inputs and box inputs.
    * Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner.
    * Coordinates must correspond to the image shape expected by the image encoder.
    * 
    * labels: Labels for the sparse input prompts.
    *   0 is a negative input point,
    *   1 is a positive input point,
    *   2 is a top-left box corner,
    *   3 is a bottom-right box corner,
    *   -1 is a padding point.
    * If there is no box input, a single padding point with label -1 and coordinates (0, 0) should be concatenated.
    */
    virtual cv::Mat predict(const std::vector<cv::Point2f>& points, const std::vector<float>& labels) = 0;
};

}
