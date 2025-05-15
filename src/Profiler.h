#ifndef PROFILER_H
#define PROFILER_H

#include <string>
#include <iostream>
#include "NodeRunner.h"
#include <fstream>
#include <filesystem>

class CsvLogger {
public:
    CsvLogger(const std::string& filename) : filename_(filename) {
        bool file_exists = std::filesystem::exists(filename_);

        // Open the file in append mode if it exists, otherwise create new and write header
        file_.open(filename_, std::ios::out | std::ios::app);
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename_);
        }

        if (!file_exists) {
            file_ << "batch_size,avg_latency_ms,peak_memory_mb\n";
            file_.flush();
        }
    }

    void log(int batch_size, double avg_latency_ms, double peak_memory_mb) {
        file_ << batch_size << ',' << avg_latency_ms << ',' << peak_memory_mb << '\n';
        file_.flush(); // Ensure the line is written immediately
    }

    ~CsvLogger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

private:
    std::string filename_;
    std::ofstream file_;
};


class Profiler {
public:
    std::string _model_name;
    std::string _onnx_file_path;
    Profiler(std::string model_name, std::string onnx_file_path): _model_name(model_name), _onnx_file_path(onnx_file_path) {};
    void profile(int max_batch_size, int gpu_id) {        
        // initialize csv file
        CsvLogger logger("models/profiles/system/" + _model_name + ".csv");
        
        // set gpu id
        CUDACHECK(cudaSetDevice(gpu_id));
        
        // loop over all batch sizes and log the average latency and peak memory used
        for(int i=1;i<=max_batch_size;i++) {
            std::vector<double> latencies;
            std::vector<size_t> peak_memory;
            for(int j=0;j<6;j++) {
                cudaDeviceSynchronize();
                cudaDeviceReset();
                auto ort_runner = getORTRunner(gpu_id);
                
                inference_running = true;
                peak_mem_used = 0;
                std::thread monitor(&Profiler::monitor_total_gpu_usage, this);
                auto start = std::chrono::high_resolution_clock::now();
    
                ort_runner->run_inference(i);
                cudaStreamSynchronize(ort_runner->_runner_stream);
    
                auto end = std::chrono::high_resolution_clock::now();
                inference_running = false;
                monitor.join();
            
                double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
                if(j!=0) latencies.push_back(latency_ms);
                if(j!=0) peak_memory.push_back(peak_mem_used);
            }

            double avg_latency_ms = 0;
            for(auto l:latencies) avg_latency_ms += l; 
            avg_latency_ms /= latencies.size();

            double peak_memory_mb = 0;
            for(auto m:peak_memory) peak_memory_mb = std::max(m/(1024.0 * 1024.0), peak_memory_mb);

            logger.log(i, avg_latency_ms, peak_memory_mb);
        }

    }

private:
    std::atomic<bool> inference_running{false};
    size_t peak_mem_used = 0;

    void monitor_total_gpu_usage() {
        size_t total_mem = 0;
        size_t current_used = 0;
        size_t max_used = 0;

        while (inference_running.load()) {
            size_t free_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);

            current_used = total_mem - free_mem;
            if (current_used > max_used) {
                max_used = current_used;
            }

            std::this_thread::sleep_for(std::chrono::microseconds(50));  // 0.05 ms
        }

        peak_mem_used = max_used;
    }

    std::shared_ptr<ORTRunner> getORTRunner(int gpu_id) {
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
        gpu_session_options.SetLogSeverityLevel(3);

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