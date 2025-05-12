#ifndef NODERUNNER_H
#define NODERUNNER_H

#include "SchedulerInterface.h"
#include "imageInput.h"
#include "Logger.h"
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

#define CUDACHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

class ORTRunner {
    // This class runs the onnx runtime, each instance will have a ort session, path to onnx file, cuda stream associated with it
public:
    explicit ORTRunner(const std::string onnx_file_path, Ort::SessionOptions&& session_options, const std::string runner_name, cudaStream_t&& runner_stream, int gpu_id): 
        _env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, runner_name.c_str())), 
        // _session_options(std::move(session_options)), 
        // _session(_env, onnx_file_path.c_str(), _session_options), 
        _runner_name(runner_name),
        _runner_stream(std::move(runner_stream)),
        _gpu_id(gpu_id)
    {
        // get logger for node runner and cuda callback
        _logger = Logger::getInstance().getLogger("ORTRunner");
        if (!_logger) {
            std::cerr << "Failed to get logger for NodeRunner(). Exiting.\n";
            exit(EXIT_FAILURE);
        }

        //print the log level
        
        _session_options = std::move(session_options);
        _session = Ort::Session(_env, onnx_file_path.c_str(), _session_options);
    }
    
    ~ORTRunner() {
        cudaStreamDestroy(_runner_stream);
    }

    void run_inference(int batch_size) {   
        if(batch_size == 0) return; 
        size_t total_elements = batch_size * CHANNELS * HEIGHT * WIDTH;
        size_t total_bytes = total_elements * sizeof(float);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "CUDA MEM INFO: " << free_mem << " " << total_mem << "\n";

        float * gpu_ptr = nullptr;
        cudaError_t cuda_err;
        CUDACHECK(cudaMalloc((void**)&gpu_ptr, total_bytes));
        std::cout << "CUDA MALLOC DONE" << std::endl;

        // copy input image over to GPU memory
        assert(gpu_ptr != nullptr);
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, gpu_ptr);

        assert(attr.type == cudaMemoryTypeDevice);
        assert(gMappedImageBin.data_ptr != nullptr);
        assert(total_bytes > 0 && total_bytes < 1ULL << 32);

        CUDACHECK(cudaMemcpyAsync(gpu_ptr, 
                                gMappedImageBin.data_ptr, 
                                total_bytes, 
                                cudaMemcpyHostToDevice,
                                _runner_stream
        ));
        std::cout << "MEM CPY DONE\n";

        Ort::MemoryInfo gpu_memory_info = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, _gpu_id, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), CHANNELS, HEIGHT, WIDTH};

        _input_tensor = Ort::Value::CreateTensor<float>(
            gpu_memory_info,
            gpu_ptr,
            total_elements,
            input_shape.data(),
            input_shape.size()
        );

        LOG_INFO(_logger, "Start inference for {} on gpu {} with batch {}", _runner_name, _gpu_id, batch_size);
        try {
            auto output_tensor = _session.Run(
                Ort::RunOptions{nullptr},
                _input_names.data(),
                &_input_tensor,
                1,
                _output_names.data(),
                1
            );
        } catch (const Ort::Exception& e) {
            std::cerr << "ORT Exception: " << e.what() << std::endl;
            std::terminate();
        }

    }

    Ort::Env _env;
    Ort::Session _session{nullptr};
    Ort::SessionOptions _session_options;
    cudaStream_t _runner_stream;
    std::string _runner_name;
    int _gpu_id;
    std::vector<const char*> _input_names = {"input"};
    std::vector<const char*> _output_names = {"output"};
    Ort::Value _input_tensor;
    std::shared_ptr<spdlog::logger> _logger;
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
    NodeRunner(std::shared_ptr<Node> node, int gpu_id, std::map<std::string, std::shared_ptr<RequestProcessor>> &request_processors): 
            running_node(*node), gpu_id(gpu_id), request_processors(request_processors) 
    {
        // get logger for node runner and cuda callback
        _logger = Logger::getInstance().getLogger("NodeRunner");
        if (!_logger) {
            std::cerr << "Failed to get logger for NodeRunner(). Exiting.\n";
            exit(EXIT_FAILURE);
        }
        
        _callback_logger = Logger::getInstance().getLogger("CudaCallback");
        if (!_callback_logger) {
            std::cerr << "Failed to get cuda callback logger for NodeRunner(). Exiting.\n";
            exit(EXIT_FAILURE);
        }     

        // initialize ORTRunners for each session in node schedule
        cudaSetDevice(gpu_id);
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
            std::shared_ptr<ORTRunner> ort_ptr;
            try { 
                ort_ptr = std::make_shared<ORTRunner>(
                    onnx_file_path, 
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
            std::cout << "Finished ORTRunner for model: " << session_ptr->model_name << std::endl;
            ort_list.push_back({ort_ptr, ort_ptr->_runner_stream});
        }
    }

    void updateNode(Node updated_node) {
        _update.lock();

        cudaSetDevice(gpu_id);
        for(auto& [session_ptr, _]: updated_node.session_list) {
            auto model_name = session_ptr->model_name;

            // check if model name already exists in existing sesison list
            // if it does.. use the corresponding ort runner
            // if not create new runner and add to list
            int existingInd = -1;
            for(int i=0;i<running_node.session_list.size();i++) {
                auto cur_session_ptr = running_node.session_list[i].first;
                auto cur_model_name = cur_session_ptr->model_name;
                if(model_name == cur_model_name) {
                    existingInd = i;
                    break;
                }
            }

            if(existingInd != -1) {
                update_ort_list.push_back(ort_list[existingInd]);
            } else {
                LOG_DEBUG(_logger, "UPDATE DETECTED: New ORTRunner being created on gpu:{} for model {}", gpu_id, session_ptr->model_name);
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
                std::shared_ptr<ORTRunner> ort_ptr;
                try { 
                    ort_ptr = std::make_shared<ORTRunner>(
                        onnx_file_path, 
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
                update_ort_list.push_back({ort_ptr, ort_ptr->_runner_stream});
            }
        }
        
        update_node = updated_node;
        pendingUpdate = true;
        _update.unlock();
    }

    void start() {
        if(!_running) {
            LOG_DEBUG(_logger, "Starting Node for gpu: {}", gpu_id);
            _runner_thread = std::thread(&NodeRunner::run, this);   
        }
    }

    void stop() {
        _running = false;
        if(_runner_thread.joinable()) {
            _runner_thread.join();
        }
    }

    ~NodeRunner() {
        if (_runner_thread.joinable()) {
            _runner_thread.join();  // or _runner_thread.detach() if appropriate
        }
    }

    std::thread _runner_thread;

private:
    std::mutex _update;
    bool pendingUpdate;
    Node update_node;
    std::vector<std::pair<std::shared_ptr<ORTRunner>, cudaStream_t>> update_ort_list;

    struct CallbackData {
        std::string _tag;
        std::shared_ptr<spdlog::logger> _logger;
        CallbackData(std::string tag, std::shared_ptr<spdlog::logger> logger): _tag(tag), _logger(logger) {}
    };

    static void CUDART_CB log_callback(void* user_data) {
        auto* data = static_cast<CallbackData*>(user_data);
    
        auto now = std::chrono::high_resolution_clock::now();
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    
        //data->_logger->info("BATCH PROCESSED: {} @ {}", data->_tag, now_us);
        data->_logger->info("BATCH PROCESSED: {} @ {}", data->_tag, now_us);
        delete data;
    }

    void run() {
        CUDACHECK(cudaSetDevice(gpu_id));
        _running = true;
        while(_running) {
            if(!_running) break;

            // loop through all session in the schedule
            if(running_node.session_list.size() > 0) {
                LOG_DEBUG(_logger, "Duty cycle start with session size {} on gpu {}", running_node.session_list.size(), gpu_id);
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

                    // get batch from request processor
                    auto batch_current = request_processors[session_ptr->model_name]->form_batch(session_ptr->batch_size, gpu_id);

                    // launch inference using ORTRunner
                    auto ort_ptr = ort_list[i].first;
                    if(batch_current>0) {
                        ort_ptr->run_inference(batch_current);

                        // add logging callback
                        std::string id = session_ptr->model_name + "_" + std::to_string(gpu_id);
                        auto* callBackData = new CallbackData(id, _callback_logger);
                        cudaLaunchHostFunc(
                            ort_ptr->_runner_stream,
                            log_callback,
                            static_cast<void*>(callBackData) 
                        );

                        running_streams.push_back(ort_ptr->_runner_stream);
                    }
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
            if(pendingUpdate) {
                _update.lock();
                    LOG_DEBUG(_logger, "Acquired lock on node to make schedule update");
                    // update node and ort list
                    running_node = update_node;
                    ort_list = update_ort_list;

                    pendingUpdate = false;
                    update_ort_list.clear();
                    LOG_DEBUG(_logger, "Releasing lock after update finished, new list size is {}", ort_list.size());
                _update.unlock();
            }
        }
    }

    std::atomic<bool> _running = false;
    
    int gpu_id;
    Node running_node;
    std::vector<std::pair<std::shared_ptr<ORTRunner>, cudaStream_t>> ort_list;
    std::map<std::string, std::shared_ptr<RequestProcessor>> request_processors;
    std::shared_ptr<spdlog::logger> _logger;
    std::shared_ptr<spdlog::logger> _callback_logger;
};

#endif