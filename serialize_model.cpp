#include <iostream>

#include "models/model.hpp"

void serialize() {
    std::unique_ptr<rt::SegmentModel> segment_model {std::make_unique<rt::SegmentModel>()};
    std::cout << "Serializing SegmentModel" << std::endl;
    segment_model->build_from_onnx("onnx/nasHard2-0.89.onnx", 352, 352);
    segment_model->serialize("onnx/segment_model_sigmoid.engine");
    std::cout << "Finished serializing SegmentModel" << std::endl;

    std::cout << "Serializing TiradModel" << std::endl;
    std::unique_ptr<rt::TiradModel> tirad_model {std::make_unique<rt::TiradModel>()};
    tirad_model->build_from_onnx("onnx/tirad_model.onnx", 258, 366);
    tirad_model->serialize("onnx/tirad_model.engine");
    std::cout << "Finished serializing TiradModel" << std::endl;
}

int main(int argc, char* argv[]) {
    serialize();
}