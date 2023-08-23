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
    ONNXVinoExecutor(ov::Core core,
                     std::filesystem::path im_encoder_path,
                     std::filesystem::path the_rest_path,
                     std::string hardware = "CPU");
    ~ONNXVinoExecutor() override;

    cv::Size input_size() const override;
    cv::Size output_size() const override;

    bool encode_image(cv::Mat input_image) override;
    cv::Mat predict(const std::vector<cv::Point2f>& points, const std::vector<float>& labels) override;

private:
    ov::Core m_core;
    std::string m_hardware; // hardware label

    // models
    std::shared_ptr<ov::Model> m_im_encoder;
    ov::CompiledModel m_im_enc_compiled;
    ov::InferRequest m_im_enc_infer;

    std::shared_ptr<ov::Model> m_the_rest;
    ov::CompiledModel m_the_rest_compiled;
    ov::InferRequest m_the_rest_infer;
};

}
