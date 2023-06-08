#include "Processing.h"

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local
#include "Constants.h"

namespace cppsam {

template<Processing::Direction direction>
auto Processing::getScale() const -> Scale {
    const double max_in_size = std::max(m_original_size.width, m_original_size.height);
    if constexpr (direction == Direction::Forwards) {
        return { m_net_input_size.width / max_in_size, m_net_input_size.height / max_in_size };
    }
    else if constexpr (direction == Direction::Backwards) {
        return { max_in_size / m_net_output_size.width, max_in_size / m_net_output_size.height };
    }
    else {
        static_assert(direction == Direction::Backwards || direction == Direction::Forwards,
                      "Error in Processing::getScale: unknown direction");
    }
}

template<>
cv::Mat Processing::map<Processing::Direction::Forwards, cv::Mat>(const cv::Mat& input_image) const {
    assert(input_image.rows == m_original_size.height);
    assert(input_image.cols == m_original_size.width);
    assert(input_image.type() == CV_32FC3); // assuming three-channel image as input

    auto scale = getScale<Processing::Direction::Forwards>();
    cv::Mat scaled;
    cv::resize(input_image, scaled, cv::Size(), scale.x, scale.y, cv::INTER_LINEAR);

    if (scaled.rows == m_net_input_size.height &&
        scaled.cols == m_net_input_size.width) return scaled;

    const int pad_y = m_net_input_size.height - scaled.rows;
    const int pad_x = m_net_input_size.width - scaled.cols;

    cv::Mat padded;
    static const cv::Scalar pad_val = cv::Scalar(pixel_mean[0], pixel_mean[1], pixel_mean[2]);
    cv::copyMakeBorder(scaled, padded, 0, pad_y, 0, pad_x, cv::BORDER_CONSTANT, pad_val);

    return padded;
}

template<>
cv::Mat Processing::map<Processing::Direction::Backwards, cv::Mat>(const cv::Mat& input_image) const {
    assert(input_image.rows == m_net_output_size.height);
    assert(input_image.cols == m_net_output_size.width);
    assert(input_image.type() == CV_32FC1); // assuming one-channel mask as input

    auto scale = getScale<Processing::Direction::Backwards>();

    cv::Mat scaled;
    cv::resize(input_image, scaled, cv::Size(), scale.x, scale.y, cv::INTER_LINEAR);
    scaled.setTo(255, scaled > 0);
    scaled.setTo(0, scaled <= 0);
    scaled.convertTo(scaled, CV_8UC1);

    return scaled(cv::Rect({ 0, 0 }, m_original_size)).clone();
}

template<>
std::vector<cv::Point2f> Processing::map<Processing::Direction::Forwards, std::vector<cv::Point2f>>(const std::vector<cv::Point2f>& input_data) const {
    std::vector<cv::Point2f> result;
    result.reserve(input_data.size());

    auto scale = getScale<Processing::Direction::Forwards>();
    std::ranges::transform(input_data, std::back_inserter(result),
                           [scale](const cv::Point2f& p) { return cv::Point2f(p.x * scale.x, p.y * scale.y); });
    return result;
}


Processing::Processing(const cv::Mat& input_image, cv::Size net_input_size, cv::Size net_output_size)
    : m_original_size(input_image.size())
    , m_net_input_size(net_input_size)
    , m_net_output_size(net_output_size)
{
    if (input_image.empty())
        throw std::runtime_error("Error in Processing::Processing: input image is empty");

    const int n_channels = input_image.channels();
    if (n_channels != 1 && n_channels != 3)
        throw std::runtime_error("Error in Processing::Processing: number of input image channels must be either one or three");

    cv::Mat converted;
    {
        cv::Mat interm;
        input_image.convertTo(interm, CV_32F);
        if (n_channels < 3) cv::merge(std::vector{interm, interm, interm}, converted);
        else converted = interm;
    }

    m_normalized = map<Direction::Forwards>(converted);

    // normalization
    static const cv::Scalar mean_val = cv::Scalar(pixel_mean[0], pixel_mean[1], pixel_mean[2]);
    static const cv::Scalar std_val = cv::Scalar(pixel_std[0], pixel_std[1], pixel_std[2]);
    m_normalized -= mean_val;
    m_normalized /= std_val;
}

}
