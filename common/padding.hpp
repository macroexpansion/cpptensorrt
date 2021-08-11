#ifndef PADDING_HPP
#define PADDING_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

namespace pad {
    class FixedSizePadding {
        public:
            int m_max_height, m_max_width;

            FixedSizePadding(int max_height, int max_width)
                : m_max_height(max_height)
                , m_max_width(max_width) {}

            template <class T>
            T pad(const T& mat);

            void calculate_pad(int height, int width, int& left_pad, int& right_pad, int& top_pad, int& bot_pad);
            
    };

    void FixedSizePadding::calculate_pad(int height, int width, int& left_pad, int& right_pad, int& top_pad, int& bot_pad) {
        if (height < m_max_height) {
            int diff {m_max_height - height};
            top_pad = diff / 2;
            bot_pad = diff % 2 == 0 ? diff / 2 : diff / 2 + 1;
        }
        if (width < m_max_width) {
            int diff {m_max_width - width};
            left_pad = diff / 2;
            right_pad = diff % 2 == 0 ? diff / 2 : diff / 2 + 1;
        }
    }

    template <>
    cv::Mat FixedSizePadding::pad<cv::Mat>(const cv::Mat& mat) {
        int height {mat.size().height}, width {mat.size().width};
        int left_pad {0}, right_pad {0}, top_pad {0}, bot_pad {0};
        calculate_pad(height, width, left_pad, right_pad, top_pad, bot_pad);

        cv::Mat bordered_mat;
        cv::copyMakeBorder(mat, bordered_mat, top_pad, bot_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar {0.0});
        return bordered_mat;
    }

    template <>
    cv::cuda::GpuMat FixedSizePadding::pad<cv::cuda::GpuMat>(const cv::cuda::GpuMat& mat) {
        int height {mat.size().height}, width {mat.size().width};
        int left_pad {0}, right_pad {0}, top_pad {0}, bot_pad {0};
        calculate_pad(height, width, left_pad, right_pad, top_pad, bot_pad);

        cv::cuda::GpuMat bordered_mat;
        cv::cuda::copyMakeBorder(mat, bordered_mat, top_pad, bot_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar {0.0});
        return bordered_mat;
    }
}

#endif