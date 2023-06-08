#include <cppsam/SAMModel.h>

#include <opencv2/core.hpp>

#include "Processing.h"

namespace cppsam {

struct SAMModelImpl {
    SAMModelImpl(std::shared_ptr<ONNXExecutorInterface> executor)
        : m_executor(executor)
    {}

    bool setInput(cv::Mat image) {
        try {
            if (!m_executor) return false;
            m_processor = std::make_unique<Processing>(image, m_executor->input_size(), m_executor->output_size());
            m_executor->encode_image(m_processor->getNormalizedImage());
        }
        catch (...) { return false; }

        return true;
    }
    cv::Mat predict(const std::vector<cv::Point2f>& points, const std::vector<float>& labels) const {
        if (!m_processor || !m_executor)
            return cv::Mat();

        auto positions = m_processor->map<Processing::Direction::Forwards>(points);

        cv::Mat mask = m_executor->predict(positions, labels);
        return mask.empty() ? cv::Mat() : m_processor->map<Processing::Direction::Backwards>(mask);
    }

    std::shared_ptr<ONNXExecutorInterface> m_executor;
    std::unique_ptr<Processing> m_processor;
};

SAMModel::SAMModel(std::shared_ptr<ONNXExecutorInterface> executor)
    : m_impl(std::make_unique<SAMModelImpl>(executor))
{}

SAMModel::~SAMModel() = default;

bool SAMModel::setInput(cv::Mat image) {
    return m_impl->setInput(image);
}

cv::Mat SAMModel::predict(const std::vector<cv::Point2f>& points, const std::vector<float>& labels) const {
    return m_impl->predict(points, labels);
}

}
