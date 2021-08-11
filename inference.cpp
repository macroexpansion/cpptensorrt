#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <exception>
#include <chrono>

#include "common/utils.hpp"
#include "common/logger.hpp"
#include "models/model.hpp"


template <class A, class B>
void pipeline(cv::Mat frame,
              std::unique_ptr<A>& segment_model,
              std::unique_ptr<B>& tirad_model) {
    try {
        cv::Mat segment {segment_model->forward(frame)};

        std::vector<std::vector<cv::Point>> contours = utils::find_contours(segment);

        std::vector<std::vector<cv::Point>> contours_poly;
        std::vector<cv::Rect> bounding_boxes;
        std::tie(contours_poly, bounding_boxes) = utils::find_bounding_boxes(contours);

        std::vector<cv::Mat> cropped_mats = utils::crop_rectangles(frame, bounding_boxes);

        utils::draw_and_save_contours(frame, contours, contours_poly, bounding_boxes);

        for (cv::Mat cropped : cropped_mats) {
            std::unique_ptr<float> tirad_out {tirad_model->forward(cropped)};
            auto ptr = tirad_out.get();
            auto max_index = std::distance(ptr, std::max_element(ptr, ptr + 2)); // get index of max element
            // std::cout << max_index << std::endl; 
        }
    } catch (const utils::NoContourException& e) {
        std::cerr << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    cudaSetDevice(1);

    cv::Mat frame = cv::imread("test.BMP");
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    std::unique_ptr<rt::SegmentModel> segment_model {std::make_unique<rt::SegmentModel>()};
    segment_model->deserialize("onnx/segment_model_sigmoid.engine", 352, 352);

    std::unique_ptr<rt::TiradModel> tirad_model {std::make_unique<rt::TiradModel>()};
    tirad_model->deserialize("onnx/tirad_model.engine", 258, 366);

    double sum {0.0};
    for (size_t i = 0; i < 1005; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        /**/
        pipeline<rt::SegmentModel, rt::TiradModel>(frame.clone(), segment_model, tirad_model);
        /**/
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);

        std::cout << duration.count() << std::endl;
        if (i < 5) continue;
        sum += duration.count();
    }
    std::cout << "final: " << sum / 1000 << std::endl;
}