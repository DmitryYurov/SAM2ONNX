#pragma once

// std
#include <memory>

// local
#include <cppsam/ONNXExecutorInterface.h>

namespace cppsam {

struct SAMModelImpl;

class SAMModel {
public:
    SAMModel(std::shared_ptr<ONNXExecutorInterface> executor);
    ~SAMModel();

    bool setInput(cv::Mat image);

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
    cv::Mat predict(const std::vector<cv::Point2f>& points, const std::vector<float>& labels) const;

private:
    std::unique_ptr<SAMModelImpl> m_impl;
};

} // namespace cppsam
