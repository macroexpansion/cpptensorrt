#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include <exception>
#include <opencv2/core/cuda.hpp>

#include "trt_model.hpp"
#include "../common/utils.hpp"
#include "../common/padding.hpp"

namespace rt {
    template <class T>
    using trt_ptr = std::unique_ptr<T, utils::trtDeleter>;

    class TiradModel : public rt::TRTModel {
        public:
            std::unique_ptr<pad::FixedSizePadding> m_padding;

            TiradModel()
                : TRTModel() 
                , m_padding(std::make_unique<pad::FixedSizePadding>(258, 366)) {}
            
            void preprocess(const cv::Mat& input, float* buffer) {
                cv::Mat rgb_image {input};
                cv::cuda::GpuMat gpu_rgb_image;
                gpu_rgb_image.upload(rgb_image);
                gpu_rgb_image = m_padding->pad<cv::cuda::GpuMat>(gpu_rgb_image);

                /* convert to float */
                cv::cuda::GpuMat float_image;
                gpu_rgb_image.convertTo(float_image, CV_32FC3, 1.f / 255.f);

                auto input_channel = m_input_dims.d[1];
                auto input_height = m_input_dims.d[2];
                auto input_width = m_input_dims.d[3];
                auto input_shape = cv::Size(input_width, input_height);
                /* to tensor */
                std::vector<cv::cuda::GpuMat> cuda_tensor;
                for (size_t i = 0; i < input_channel; ++i) {
                    cuda_tensor.emplace_back(cv::cuda::GpuMat(input_shape, CV_32FC1, buffer + i * input_width * input_height));
                }
                cv::cuda::split(float_image, cuda_tensor);
            }

            std::unique_ptr<float> forward(const cv::Mat& input) {
                float* input_ptr;
                float* output_ptr;
                if (!m_cuda_malloc(input_ptr, output_ptr)) throw std::exception();
                std::unique_ptr<float, utils::cudaMemDeleter> input_mem (input_ptr);
                std::unique_ptr<float, utils::cudaMemDeleter> output_mem (output_ptr);

                preprocess(input, (float*)input_mem.get());
                std::unique_ptr<float> out {inference((void*)input_mem.get(), (void*)output_mem.get())};
                return out;
            }
    };

    class SegmentModel : public rt::TRTModel {
        public:
            SegmentModel()
                : TRTModel() {}

            void build_from_onnx(const std::string& onnx_path, int input_height, int input_width) {
                trt_ptr<nvinfer1::IBuilder> builder {nvinfer1::createInferBuilder(logger::gLogger)};

                const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
                trt_ptr<nvinfer1::INetworkDefinition> network {builder->createNetworkV2(explicitBatch)};
                trt_ptr<nvonnxparser::IParser> parser {nvonnxparser::createParser(*network, logger::gLogger)};
                parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
                nvinfer1::IActivationLayer* sigmoid {network->addActivation(*network->getOutput(0), nvinfer1::ActivationType::kSIGMOID)};
                assert(sigmoid);
                network->getOutput(0)->setName("old_output");
                network->unmarkOutput(*network->getOutput(0));
                sigmoid->getOutput(0)->setName("output");
                network->markOutput(*sigmoid->getOutput(0));

                m_optimize_engine(network, builder);
                m_get_mem_size(input_height, input_width);
            }

            void preprocess(const cv::Mat& input, float* buffer) {
                cv::Mat rgb_image {input};
                
                if (rgb_image.empty()) {
                    std::cerr << "input load failed\n";
                    return;
                }

                cv::cuda::GpuMat gpu_rgb_image;
                gpu_rgb_image.upload(rgb_image);

                auto input_channel = m_input_dims.d[1];
                auto input_height = m_input_dims.d[2];
                auto input_width = m_input_dims.d[3];
                auto input_shape = cv::Size(input_width, input_height);

                cv::cuda::GpuMat resized;
                cv::cuda::resize(gpu_rgb_image, resized, input_shape, 0, 0, cv::INTER_NEAREST);

                cv::cuda::GpuMat float_image;
                resized.convertTo(float_image, CV_32FC3, 1.f / 255.f);

                std::vector<cv::cuda::GpuMat> cuda_tensor;
                for (size_t i = 0; i < input_channel; ++i) {
                    cuda_tensor.emplace_back(cv::cuda::GpuMat(input_shape, CV_32FC1, buffer + i * input_width * input_height));
                }
                cv::cuda::split(float_image, cuda_tensor);
            }

            cv::cuda::GpuMat inference(void* input_mem, void* output_mem) {
                std::unique_ptr<cudaStream_t, utils::cudaStreamDeleter> stream (new cudaStream_t);
                if (cudaStreamCreate(stream.get()) != cudaSuccess) {
                    logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
                }
                void* bindings[] = {input_mem, output_mem};
                bool status = m_context->enqueueV2(bindings, *stream, nullptr);
                if (!status) {
                    logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
                }
                cudaStreamSynchronize(*stream);
                
                cv::cuda::GpuMat gpu_mat (cv::Size(m_output_dims.d[3], m_output_dims.d[2]), CV_32FC1, (float*)output_mem);
                cv::cuda::GpuMat gpu_thresholded;
                cv::cuda::threshold(gpu_mat, gpu_thresholded, 0.1, 1., cv::THRESH_BINARY);
                gpu_thresholded.convertTo(gpu_thresholded, CV_8UC1);

                return gpu_thresholded;
            }

            cv::Mat forward(const cv::Mat& source) {
                float* input_ptr;
                float* output_ptr;
                if (!m_cuda_malloc(input_ptr, output_ptr)) throw std::exception();
                std::unique_ptr<float, utils::cudaMemDeleter> input_mem (input_ptr);
                std::unique_ptr<float, utils::cudaMemDeleter> output_mem (output_ptr);

                preprocess(source, (float*)input_mem.get());
                cv::cuda::GpuMat gpu_thresholded {inference((void*)input_mem.get(), (void*)output_mem.get())};

                cv::cuda::GpuMat gpu_resized;
                cv::cuda::resize(gpu_thresholded, gpu_resized, source.size(), 0, 0, cv::INTER_NEAREST);

                cv::Mat cpu_resized;
                gpu_resized.download(cpu_resized);

                return cpu_resized;
            }
    };

    class FnaModel : public rt::TRTModel {

    };
};

#endif