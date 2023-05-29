#include "Processing.h"

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local
#include "Constants.h"

namespace cppsam {

Processing::Processing(const cv::Mat& input_image)
    : m_original_size(input_image.size())
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

template<>
cv::Mat Processing::map<Processing::Direction::Forwards>(const cv::Mat& input_image) const {
    assert(input_image.rows == m_original_size.height);
    assert(input_image.cols == m_original_size.width);
    assert(input_image.type() == CV_32FC3); // assuming three-channel image as input

    const double scale = getScale<Processing::Direction::Forwards>();
    cv::Mat scaled;
    cv::resize(input_image, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);

    if (scaled.rows == target_size && scaled.cols == target_size)
        return scaled;

    const int pad_y = static_cast<int>(target_size) - scaled.rows;
    const int pad_x = static_cast<int>(target_size) - scaled.cols;

    cv::Mat padded;
    static const cv::Scalar pad_val = cv::Scalar(pixel_mean[0], pixel_mean[1], pixel_mean[2]);
    cv::copyMakeBorder(scaled, padded, 0, pad_y, 0, pad_x, cv::BORDER_CONSTANT, pad_val);

    return padded;
}

template<>
cv::Mat Processing::map<Processing::Direction::Backwards>(const cv::Mat& input_image) const {
    assert(input_image.rows == target_size);
    assert(input_image.cols == target_size);
    assert(input_image.type() == CV_8UC1); // assuming one-channel mask as input

    const double scale = getScale<Processing::Direction::Backwards>();
    cv::Mat scaled;
    cv::resize(input_image, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
    scaled.setTo(255, scaled > 0);

    if (scaled.size() == m_original_size)
        return scaled;

    const int pad_y = static_cast<int>(target_size) - scaled.rows;
    const int pad_x = static_cast<int>(target_size) - scaled.cols;

    return scaled(cv::Rect({ 0, 0 }, m_original_size)).clone();
}

template<Processing::Direction direction>
std::vector<cv::Point2f> Processing::map(const std::vector<cv::Point2f>& input_data) const {
    std::vector<cv::Point2f> result;
    result.reserve(input_data.size());

    const double scale = getScale<direction>();
    std::ranges::transform(input_data, std::back_inserter(result),
                           [scale](const cv::Point2f& p) { return p * scale; });
    return result;
}

template<>
double Processing::getScale<Processing::Direction::Forwards>() const {
    const int max_in_size = std::max(m_original_size.width, m_original_size.height);
    return static_cast<double>(target_size) / max_in_size;
}

template<>
double Processing::getScale<Processing::Direction::Backwards>() const {
    const int max_in_size = std::max(m_original_size.width, m_original_size.height);
    return static_cast<double>(max_in_size) / target_size;
}

}
