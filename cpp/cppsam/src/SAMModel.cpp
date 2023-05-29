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
            m_processor = std::make_unique<Processing>(image);
            m_executor->EncodeImage(m_processor->getNormalizedImage());
        }
        catch (...) { return false; }

        return true;
    }
    cv::Mat predict(const std::vector<cv::Point2f>& positive_pos,
                    const std::vector<cv::Point2f>& negative_pos) const {
        if (!m_processor || !m_executor)
            return cv::Mat();

        auto ppos = m_processor->map<Processing::Direction::Forwards>(positive_pos);
        auto npos = m_processor->map<Processing::Direction::Forwards>(negative_pos);

        cv::Mat mask = m_executor->predict(ppos, npos);
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

cv::Mat SAMModel::predict(const std::vector<cv::Point2f>& positive_pos,
                          const std::vector<cv::Point2f>& negative_pos) const {
    return m_impl->predict(positive_pos, negative_pos);
}

}
