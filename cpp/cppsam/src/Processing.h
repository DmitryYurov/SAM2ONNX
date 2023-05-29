#pragma once

// std includes
#include <tuple>
#include <vector>

// opencv includes
#include <opencv2/core.hpp>

namespace cppsam
{

class Processing {
public:
    enum class Direction {
        Forwards,
        Backwards
    };

    // Accepts the target size (px) of a side of the output image (output images is to be square)
    Processing(const cv::Mat& input_image);

    cv::Mat getNormalizedImage() { return m_normalized; };

    template<Direction direction>
    cv::Mat map(const cv::Mat& input_image) const;

    template <Direction direction>
    std::vector<cv::Point2f> map(const std::vector<cv::Point2f>& input_data) const;

private:
    template<Direction direction>
    double getScale() const;

    cv::Size m_original_size;
    cv::Mat m_normalized;
};

template<> cv::Mat Processing::map<Processing::Direction::Forwards>(const cv::Mat& input_image) const;
template<> cv::Mat Processing::map<Processing::Direction::Backwards>(const cv::Mat& input_image) const;
template<> std::vector<cv::Point2f> Processing::map<Processing::Direction::Forwards>(const std::vector<cv::Point2f>& input_data) const;
template<> std::vector<cv::Point2f> Processing::map<Processing::Direction::Backwards>(const std::vector<cv::Point2f>& input_data) const;
template<> double Processing::getScale<Processing::Direction::Forwards>() const;
template<> double Processing::getScale<Processing::Direction::Backwards>() const;

} // namespace cppsam
