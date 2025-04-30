#ifndef NODERUNNER_H
#define NODERUNNER_H

#include "SchedulerInterface.h"
#include "imageInput.h"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

// has mmaped image bin file
extern const int CHANNELS;
extern const int HEIGHT;
extern const int WIDTH;
extern MappedBin gMappedImageBin;

class ORTRunner {
    // This class runs the onnx runtime, each instance will have a ort session, path to onnx file, cuda stream associated with it
public:
    explicit ORTRunner(const std::string onnx_file_path, Ort::SessionOptions&& session_options, const std::string runner_name, cudaStream_t&& runner_stream, int gpu_id): 
        _env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, runner_name.c_str())), 
        _session_options(std::move(session_options)), 
        _session(_env, onnx_file_path.c_str(), _session_options), 
        _runner_name(runner_name),
        _runner_stream(std::move(runner_stream)),
        _gpu_id(gpu_id)
    {}
    
    ~ORTRunner() {
        cudaStreamDestroy(_runner_stream);
    }

    void run_inference(int batch_size) {
        size_t total_elements = batch_size * CHANNELS * HEIGHT * WIDTH;
        size_t total_bytes = total_elements * sizeof(float);

        float * gpu_ptr = nullptr;
        cudaMalloc((void**)&gpu_ptr, total_bytes);

        // copy input image over to GPU memory
        cudaMemcpyAsync(gpu_ptr, gMappedImageBin.data_ptr, total_bytes, cudaMemcpyHostToDevice, _runner_stream);

        Ort::MemoryInfo gpu_memory_info = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, _gpu_id, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), CHANNELS, HEIGHT, WIDTH};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            gpu_memory_info,
            gpu_ptr,
            total_elements,
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        auto output_tensor = _session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );
    }

    Ort::Env _env;
    Ort::Session _session;
    Ort::SessionOptions _session_options;
    cudaStream_t _runner_stream;
    std::string _runner_name;
    int _gpu_id;
};

class NodeRunner {
    // This class manages complete execution on 1 whole GPU

    // what should the node runner contain?
    // 1. node object with session list
    // 2. list of ort runtime sessions and cuda streams for each model in the session list
    //    * This should be created before schedule launch and recreated on schedule update
    // 3. gpu id
    // 4. object to hold the updated node schedule
    // 5. bool flag to represent there is pending update, needs to be thread safe
    // 6. mutex for above thread safety

    // what should the node runner expose?
    // 1. updateNode(node): To update the schedule for the node runner
    // 2. start(): To start running the current schedule list

    // TO DO: Move all function bodies to .cpp file
public:
    NodeRunner(std::shared_ptr<Node> node, int gpu_id): running_node(*node), gpu_id(gpu_id) {
        // initialize ORTRunners for each session in node schedule
        std::cout << "NodeRunner CONSTRUCTOR START\n";
        for(auto& [session_ptr, _]: running_node.session_list) {
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

            const std::string onnx_file_path = "models/" + session_ptr->model_name + ".onnx";
            
            std::cout << "Initializing ORTRunner for model: " << session_ptr->model_name << std::endl;
            const auto ort_ptr = std::make_shared<ORTRunner>(
                onnx_file_path, 
                std::move(gpu_session_options), 
                "ORTRunner_" + std::to_string(gpu_id),
                std::move(ort_stream),
                gpu_id
            );
            std::cout << "Finished ORTRunner for model: " << session_ptr->model_name << std::endl;
            ort_list.push_back({ort_ptr, ort_ptr->_runner_stream});
        }
        std::cout << "NodeRunner CONSTRUCTOR END\n";
    }

    void updateNode(Node updated_node) {
        _update.lock();
        new_node = updated_node;
        pedningUpdate = true;
        _update.unlock();
    }

    void start() {
        cudaSetDevice(gpu_id);
        if(!_running) _runner_thread = std::thread(&NodeRunner::run, this);
    }

    void stop() {
        _running = false;
        if(_runner_thread.joinable()) {
            _runner_thread.join();
        }
    }

private:
    std::mutex _update;
    bool pedningUpdate;
    Node new_node;

    void run() {
        _running = true;
        while(_running) {
            if(!_running) break;

            // loop through all session in the schedule
            if(running_node.session_list.size() > 0) {
                std::vector<cudaStream_t> running_streams;
                auto schedule_start = std::chrono::steady_clock::now();
                for(int i=0;i<running_node.session_list.size();i++) {
                    auto session_ptr = running_node.session_list[i].first;
                    auto occ_ratio = running_node.session_list[i].second.first;
                    auto is_parallel = running_node.session_list[i].second.second;

                    if(is_parallel == 0) {
                        // wait for previous streams to complete
                        for(auto stream:running_streams) {
                            cudaStreamSynchronize(stream);
                        }

                        running_streams.clear();
                    }

                    // launch inference using ORTRunner
                    auto ort_ptr = ort_list[i].first;
                    ort_ptr->run_inference(session_ptr->batch_size);
                    running_streams.push_back(ort_ptr->_runner_stream);
                } if (running_streams.size()) {
                    // wait for all running streams to complete
                    for(auto stream:running_streams) {
                        cudaStreamSynchronize(stream);
                    }

                    running_streams.clear();
                }

                // wait for remaining time in duty cycle
                auto schedule_end = std::chrono::steady_clock::now();
                auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(schedule_end - schedule_start);
                auto duration_ms = running_node.duty_cycle;
                if(elapsed_time.count() < duration_ms) {
                    auto sleep_time_ms = std::chrono::milliseconds((long long)duration_ms - elapsed_time.count());
                    std::this_thread::sleep_for(sleep_time_ms);
                }
            }

            // check if update is needded
            _update.lock();
            if(pedningUpdate) {
                // update node with new_node
            } _update.unlock();
        }
    }

    std::atomic<bool> _running = false;
    std::thread _runner_thread;

    int gpu_id;
    Node running_node;
    std::vector<std::pair<std::shared_ptr<ORTRunner>, cudaStream_t>> ort_list;
};

#endif