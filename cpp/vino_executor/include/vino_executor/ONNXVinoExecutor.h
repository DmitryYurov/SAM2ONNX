#pragma once

#include <cppsam/ONNXExecutorInterface.h>

// std includes
#include <filesystem>
#include <memory>

// OpenVINO includes
#include <openvino/openvino.hpp>

namespace vino_executor {

class ONNXVinoExecutor : public cppsam::ONNXExecutorInterface {
public:
    ONNXVinoExecutor(std::shared_ptr<ov::Core> core,
                     std::filesystem::path im_encoder_path,
                     std::filesystem::path the_rest_path);
    ~ONNXVinoExecutor() override;

    bool EncodeImage(cv::Mat input_image) override;
    cv::Mat predict(const std::vector<cv::Point2f>& positive, const std::vector<cv::Point2f>& negative) override;

private:
    std::shared_ptr<ov::Core> m_core;
    std::shared_ptr<ov::Tensor> m_encoded_image;

    // models
    std::shared_ptr<ov::Model> m_im_encoder;
    std::shared_ptr<ov::Model> m_the_rest;

    std::shared_ptr<ov::CompiledModel> m_encoder_comp;
    std::shared_ptr<ov::CompiledModel> m_the_rest_comp;
};

}
