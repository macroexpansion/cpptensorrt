#ifndef UTIL_HPP
#define UTIL_HPP

#include <algorithm>
#include <memory>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <exception>
#include <opencv2/imgproc.hpp>
#include <NvInfer.h>
#include "logger.hpp"

namespace utils {
    struct cudaStreamDeleter {
        void operator()(cudaStream_t* stream) {
            if (cudaStreamDestroy(*stream) != cudaSuccess) {
                logger::gLogError << "ERROR: cuda stream destruction failed." << std::endl;
            }
        }
    };

    struct cudaMemDeleter {
        void operator()(float* mem) {
            if (cudaFree(mem) != cudaSuccess) {
                logger::gLogError << "ERROR: cuda free memory failed." << std::endl;
            }
        } 
    };

    struct trtDeleter {
        template <class T>
        void operator()(T* obj) const {
            if (obj) {
                obj->destroy();
            }
        }
    };

    template <class T>
    using trt_ptr = std::unique_ptr<T, trtDeleter>;

    size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size) {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int32_t>()) * elem_size;
    }

    struct PPM {
        std::string filename;
        std::string magic;
        int c;
        int h;
        int w;
        int max;
        std::vector<uint8_t> buffer;
    };

    class ImageBase {
        public:
            ImageBase(const std::string& filename, const nvinfer1::Dims& dims)
                : mDims(dims) {
                assert(4 == mDims.nbDims);
                assert(1 == mDims.d[0]);
                mPPM.filename = filename;
            }

            virtual ~ImageBase() {}

            virtual size_t volume() const {
                return mDims.d[3] /* w */ * mDims.d[2] /* h */ * 3;
            }

            void read() {
                std::ifstream infile(mPPM.filename, std::ifstream::binary);
                if (!infile.is_open())
                {
                    std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
                }
                infile >> mPPM.magic >> mPPM.w >> mPPM.h >> mPPM.max;

                infile.seekg(1, infile.cur);
                mPPM.buffer.resize(volume());
                infile.read(reinterpret_cast<char*>(mPPM.buffer.data()), volume());
                infile.close();
            }

            void write() {
                std::ofstream outfile(mPPM.filename, std::ofstream::binary);
                if (!outfile.is_open())
                {
                    std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
                }
                outfile << mPPM.magic << " " << mPPM.w << " " << mPPM.h << " " << mPPM.max << std::endl;
                outfile.write(reinterpret_cast<char*>(mPPM.buffer.data()), volume());
                outfile.close();
            }

        protected:
            nvinfer1::Dims mDims;
            PPM mPPM;
    };

    class RGBImageReader : public ImageBase {
        public:
            RGBImageReader(const std::string& filename, const nvinfer1::Dims& dims, const std::vector<float>& mean, const std::vector<float>& std)
                : ImageBase(filename, dims)
                , mMean(mean)
                , mStd(std) {}

            std::unique_ptr<float> process() const {
                const int channels = mDims.d[1];
                const int height = mDims.d[2];
                const int width = mDims.d[3];
                auto buffer = std::unique_ptr<float>{new float[volume()]};

                if (mPPM.h == height && mPPM.w == width) {
                    for (int c = 0; c < channels; c++) {
                        for (int j = 0, HW = height * width; j < HW; ++j) {
                            buffer.get()[c * HW + j] = (static_cast<float>(mPPM.buffer[j * channels + c])/mPPM.max - mMean[c]) / mStd[c];
                        }
                    }
                } else {
                    assert(!"Specified dimensions don't match PPM image");
                }

                return buffer;
            }

        private:
            std::vector<float> mMean;
            std::vector<float> mStd;
    };

    class ArgmaxImageWriter : public ImageBase {
        public:
            ArgmaxImageWriter(const std::string& filename, const nvinfer1::Dims& dims, const std::vector<int>& palette, const int num_classes)
                : ImageBase(filename, dims)
                , mNumClasses(num_classes)
                , mPalette(palette) { }

            void process(const int* buffer) {
                mPPM.magic = "P6";
                mPPM.w = mDims.d[3];
                mPPM.h = mDims.d[2];
                mPPM.max = 255;
                mPPM.buffer.resize(volume());
                std::vector<std::vector<int>> colors;
                for (auto i = 0, max = mPPM.max; i < mNumClasses; i++) {
                    std::vector<int> c{mPalette};
                    std::transform(c.begin(), c.end(), c.begin(), [i, max](int p){return (p*i) % max;});
                    colors.push_back(c);
                }
                for (int j = 0, HW = mPPM.h * mPPM.w; j < HW; ++j) {
                    auto clsid{static_cast<uint8_t>(buffer[j])};
                    mPPM.buffer.data()[j*3] = colors[clsid][0];
                    mPPM.buffer.data()[j*3+1] = colors[clsid][1];
                    mPPM.buffer.data()[j*3+2] = colors[clsid][2];
                }
            }

        private:
            int mNumClasses;
            std::vector<int> mPalette;
    };

    class NoContourException : public std::exception {
        public:
            const char* what () const throw () {
                return "No contour found";
            }
    };

    std::vector<std::vector<cv::Point>> find_contours(cv::Mat mat) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mat, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        if (contours.size() == 0) throw NoContourException();

        float max_area {0.0};
        int max_index;
        for (size_t i = 0; i < contours.size(); i++) {
            auto area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_index = i;
            }
        }
        return std::vector<std::vector<cv::Point>> {contours[max_index]};
    }
    
    void draw_and_save_contours(cv::Mat mat,
                       std::vector<std::vector<cv::Point>> contours,
                       std::vector<std::vector<cv::Point>> contours_poly,
                       std::vector<cv::Rect> boundRect) {
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(mat, contours_poly, (int)i, color, 2, cv::LINE_8);
            cv::rectangle(mat, boundRect[i].tl(), boundRect[i].br(), color, 2);
        }
        cv::imwrite("saved.jpg", mat);
    }

    std::tuple<std::vector<std::vector<cv::Point>>, std::vector<cv::Rect>> find_bounding_boxes(std::vector<std::vector<cv::Point>> contours) {
        std::vector<std::vector<cv::Point>> contours_poly (contours.size());
        std::vector<cv::Rect> boundRects (contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
            boundRects[i] = cv::boundingRect(contours_poly[i]);
        }
        return std::make_tuple(contours_poly, boundRects);
    }

    std::vector<cv::Mat> crop_rectangles(const cv::Mat source, std::vector<cv::Rect> boundRects, int pad=80) {
        std::vector<cv::Mat> cropped_mats;
        auto size = source.size();
        for (cv::Rect boundRect : boundRects) {
            cv::Rect paddingRect = boundRect;
            paddingRect.height = std::min(paddingRect.height + 2 * pad, size.height);
            paddingRect.width = std::min(paddingRect.width + 2 * pad, size.width);
            paddingRect.x = std::max(paddingRect.x - pad, 0);
            paddingRect.y = std::max(paddingRect.y - pad, 0);

            cv::Mat crop = source(paddingRect); 
            cv::Mat copyCrop;
            crop.copyTo(copyCrop); // copy data
            cropped_mats.push_back(copyCrop);
        }
        return cropped_mats;
    }
};

#endif