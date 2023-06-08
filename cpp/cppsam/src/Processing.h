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
    Processing(const cv::Mat& input_image, cv::Size net_input_size, cv::Size net_output_size);

    cv::Mat getNormalizedImage() { return m_normalized; };

    template<Direction direction, class Type>
    Type map(const Type& input_image) const;

private:
    struct Scale {
        double x, y;
    };

    template<Direction direction>
    Scale getScale() const;

    cv::Size m_original_size;
    cv::Size m_net_input_size;
    cv::Size m_net_output_size;
    cv::Mat m_normalized;
};

} // namespace cppsam
