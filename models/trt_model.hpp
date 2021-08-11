#ifndef TRT_MODEL_HPP
#define TRT_MODEL_HPP

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#include "../common/utils.hpp"
#include "../common/logger.hpp"

namespace rt {
    template <class T>
    using trt_ptr = std::unique_ptr<T, utils::trtDeleter>;

    class TRTModel {
        protected:
            const std::string m_engine_filename;
            trt_ptr<nvinfer1::ICudaEngine> m_engine;
            trt_ptr<nvinfer1::IExecutionContext> m_context;

            void m_optimize_engine(trt_ptr<nvinfer1::INetworkDefinition>& network, trt_ptr<nvinfer1::IBuilder>& builder);
            bool m_get_mem_size(int height, int width);
            bool m_cuda_malloc(float*& input_ptr, float*& output_ptr);

        public:
            nvinfer1::Dims m_input_dims, m_output_dims;
            int m_input_size, m_output_size;

            TRTModel();
            void build_from_engine();
            void build_from_onnx(const std::string& onnx_path, int input_height, int input_width);
            void preprocess(const cv::Mat& input, float* buffer);
            void serialize(const std::string& path);
            void deserialize(const std::string& path, int input_height, int input_width);
            std::unique_ptr<float> inference(void* input_mem, void* output_mem);
            std::unique_ptr<float> forward(const cv::Mat& input);

            friend std::ostream& operator<<(std::ostream& os, const TRTModel& obj);
            friend std::string read_buffer(const std::string& path);
    };

    std::ostream& operator<<(std::ostream& os, const TRTModel& obj) {
        os << "TRTModel";
        return os;
    }

    TRTModel::TRTModel()
        : m_engine(nullptr)
        , m_context(nullptr) {}

    void TRTModel::serialize(const std::string& path) {
        if (path.empty()) return;
        trt_ptr<nvinfer1::IHostMemory> serialized_model (m_engine->serialize());
        std::ofstream ofs (path, std::ios::out | std::ios::binary);
        ofs.write((const char*)serialized_model->data(), serialized_model->size());
        ofs.close();
    }

    std::string read_buffer(const std::string& path) {
        std::string buffer;
        std::ifstream stream (path.c_str(), std::ios::binary);

        if (stream) {
            stream >> std::noskipws;
            std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), std::back_inserter(buffer));
        }

        return buffer;
    }

    void TRTModel::deserialize(const std::string& path, int input_height, int input_width) {
        std::string buffer = read_buffer(path);
        if (buffer.size()) {
            trt_ptr<nvinfer1::IRuntime> runtime {nvinfer1::createInferRuntime(logger::gLogger)};
            m_engine.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr));
            assert(m_engine.get() != nullptr);
            m_context.reset(m_engine->createExecutionContext());
            assert(m_context.get() != nullptr);

            m_get_mem_size(input_height, input_width);
        }
    }

    void TRTModel::m_optimize_engine(trt_ptr<nvinfer1::INetworkDefinition>& network, trt_ptr<nvinfer1::IBuilder>& builder) {
        trt_ptr<nvinfer1::IBuilderConfig> config {builder->createBuilderConfig()};

        /* set up 14Gb GPU mem for TensorRT tactic selection */
        config->setMaxWorkspaceSize(13ULL << 30);
        /* use FP16 when possible */
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        builder->setMaxBatchSize(1);
        /* create optimized TensorRT engine */
        m_engine.reset(builder->buildEngineWithConfig( *network, *config ));
        assert(m_engine.get() != nullptr);
        /* create context excecution */
        m_context.reset(m_engine->createExecutionContext());
        assert(m_context.get() != nullptr);
    }

    bool TRTModel::m_get_mem_size(int height, int width) {
        auto input_index = m_engine->getBindingIndex("input");
        if (input_index == -1) return false; 
        assert(m_engine->getBindingDataType(input_index) == nvinfer1::DataType::kFLOAT);
        m_input_dims = nvinfer1::Dims4{1, 3, height, width};
        m_context->setBindingDimensions(input_index, m_input_dims);
        m_input_size = utils::getMemorySize(m_input_dims, sizeof(float));

        auto output_index = m_engine->getBindingIndex("output");
        if (output_index == -1) return false; 
        assert(m_engine->getBindingDataType(output_index) == nvinfer1::DataType::kFLOAT);
        m_output_dims = m_context->getBindingDimensions(output_index);
        m_output_size = utils::getMemorySize(m_output_dims, sizeof(float));

        return true;
    }

    bool TRTModel::m_cuda_malloc(float*& input_ptr, float*& output_ptr) {
        /* allocate CUDA memory for input and output bindings */
        if (cudaMalloc((void**)&input_ptr, m_input_size) != cudaSuccess) {
            logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << m_input_size << " bytes" << std::endl;
            return false;
        }
        if (cudaMalloc((void**)&output_ptr, m_output_size) != cudaSuccess) {
            logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << m_output_size << " bytes" << std::endl;
            return false;
        }
        return true;
    }

    void TRTModel::build_from_onnx(const std::string& onnx_path, int input_height, int input_width) {
        trt_ptr<nvinfer1::IBuilder> builder {nvinfer1::createInferBuilder(logger::gLogger)};

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        trt_ptr<nvinfer1::INetworkDefinition> network {builder->createNetworkV2(explicitBatch)};
        trt_ptr<nvonnxparser::IParser> parser {nvonnxparser::createParser(*network, logger::gLogger)};
        parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

        m_optimize_engine(network, builder);
        m_get_mem_size(input_height, input_width);
    }

    std::unique_ptr<float> TRTModel::inference(void* input_mem, void* output_mem) {
        /* log input */
        // std::unique_ptr<float> test {new float[m_input_size]};
        // if (cudaMemcpy(test.get(), input_mem, m_input_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        //     logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        //     return nullptr;
        // }
        // for (size_t i = 0; i < m_input_size / sizeof(float); i++) {
        //     std::cout << *(test.get() + i) << " ";
        // }
        // std::cout << std::endl;
        /* end log input */

        std::unique_ptr<cudaStream_t, utils::cudaStreamDeleter> stream (new cudaStream_t);
        if (cudaStreamCreate(stream.get()) != cudaSuccess) {
            logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
            return nullptr;
        }
        void* bindings[] = {input_mem, output_mem};
        bool status = m_context->enqueueV2(bindings, *stream, nullptr);
        if (!status) {
            logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
            return nullptr;
        }
        auto output_buffer = std::unique_ptr<float>{new float[m_output_size]};
        if (cudaMemcpyAsync(output_buffer.get(), output_mem, m_output_size, cudaMemcpyDeviceToHost, *stream) != cudaSuccess) {
            logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << m_output_size <<" bytes" << std::endl;
            return nullptr;
        }
        cudaStreamSynchronize(*stream);

        return output_buffer;
    }
};

#endif