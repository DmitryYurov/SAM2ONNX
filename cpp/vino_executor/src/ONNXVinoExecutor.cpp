#include <vino_executor/ONNXVinoExecutor.h>

namespace vino_executor {

ONNXVinoExecutor::ONNXVinoExecutor(std::shared_ptr<ov::Core> core,
                                   std::filesystem::path im_encoder_path,
                                   std::filesystem::path the_rest_path)
    : m_core(core)
    , m_im_encoder(core->read_model(im_encoder_path.c_str()))
    , m_the_rest(core->read_model(the_rest_path.c_str()))
{
}

ONNXVinoExecutor::~ONNXVinoExecutor() = default;

bool ONNXVinoExecutor::EncodeImage(cv::Mat input_image) {
    return false;
}

cv::Mat ONNXVinoExecutor::predict(const std::vector<cv::Point2f>& positive, const std::vector<cv::Point2f>& negative) {
    return cv::Mat();
}

}
