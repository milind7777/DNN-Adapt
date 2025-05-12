#ifndef PROFILER_H
#define PROFILER_H

#include <string>
#include <iostream>
#include "NodeRunner.h"

class Profiler {
public:
    std::string _model_name;
    std::string _onnx_file_path;
    Profiler(std::string model_name, std::string onnx_file_path): _model_name(model_name), _onnx_file_path(onnx_file_path) {};
    void profile(int max_batch_size, int gpu_id) {
        // loop over all batch sizes
        // create a onnx runtime session with profiling enabled
        // check how the memory profiling looks like

        // Run with single batch value to test output

        // Create ORTRunner with profiling enabled
        CUDACHECK(cudaSetDevice(gpu_id));
        auto ort_runner = getORTRunner(gpu_id, "profile_test");
        auto start = std::chrono::high_resolution_clock::now();

        ort_runner->run_inference(512);
        cudaStreamSynchronize(ort_runner->_runner_stream);

        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Latency is: " << latency_ms << std::endl;
    }

private:
    std::shared_ptr<ORTRunner> getORTRunner(int gpu_id, std::string profile_log_name) {
        Ort::SessionOptions gpu_session_options;
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = gpu_id;

        cudaStream_t ort_stream;
        cudaError_t err = cudaStreamCreate(&ort_stream);
        if(err != cudaSuccess) {
            throw std::runtime_error("Faile to create CUDA stream in ORTRunner");
        }
        cuda_options.has_user_compute_stream = 1;
        cuda_options.user_compute_stream = static_cast<void*> (ort_stream);
        gpu_session_options.AppendExecutionProvider_CUDA(cuda_options);
        gpu_session_options.EnableProfiling(profile_log_name.c_str());

        std::shared_ptr<ORTRunner> ort_ptr;
        try { 
            ort_ptr = std::make_shared<ORTRunner>(
                _onnx_file_path, 
                std::move(gpu_session_options), 
                "ORTRunner_" + std::to_string(gpu_id),
                std::move(ort_stream),
                gpu_id
            );
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            std::terminate();
        } catch (const std::exception& e) {
            std::cerr << "Standard exception: " << e.what() << std::endl;
            std::terminate();
        } catch (...) {
            std::cerr << "Unknown exception occurred while constructing ORTRunner." << std::endl;
            std::terminate();
        }

        return ort_ptr;
    }
};

#endif