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
    cv::Mat predict(const std::vector<cv::Point2f>& positive_pos, const std::vector<cv::Point2f>& negative_pos) const;

private:
    std::unique_ptr<SAMModelImpl> m_impl;
};

} // namespace cppsam
