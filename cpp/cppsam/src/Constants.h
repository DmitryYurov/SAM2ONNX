#pragma once

// std includes
#include <array>

namespace cppsam {
static constexpr size_t target_size = 1024u; //px
static constexpr std::array<double, 3> pixel_mean = { 123.675, 116.28, 103.53 };
static constexpr std::array<double, 3> pixel_std = { 58.395, 57.12, 57.375 };
}
