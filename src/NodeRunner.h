#ifndef NODERUNNER_H
#define NODERUNNER_H

#include "SchedulerInterface.h"
#include "imageInput.h"
#include "Logger.h"
#include "RequestProcessor.h"
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
extern const int SLOTS_PER_GPU;

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
        // std::cout << "CUDA MEM INFO: " << free_mem << " " << total_mem << "\n";

        float * gpu_ptr = nullptr;
        cudaError_t cuda_err;
        CUDACHECK(cudaMalloc((void**)&gpu_ptr, total_bytes));
        // std::cout << "CUDA MALLOC DONE" << std::endl;

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
        // std::cout << "MEM CPY DONE\n";

        Ort::MemoryInfo gpu_memory_info = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, _gpu_id, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), CHANNELS, HEIGHT, WIDTH};

        _input_tensor = Ort::Value::CreateTensor<float>(
            gpu_memory_info,
            gpu_ptr,
            total_elements,
            input_shape.data(),
            input_shape.size()
        );

        LOG_DEBUG(_logger, "Start inference for {} on gpu {} with batch {}", _runner_name, _gpu_id, batch_size);
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

class NodeRunner : public std::enable_shared_from_this<NodeRunner> {
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
    NodeRunner(std::shared_ptr<Node> node, int gpu_id, std::map<std::string, std::string> &modelsList, std::map<std::string, std::shared_ptr<RequestProcessor>> &request_processors, std::map<std::string, double> &latencies): 
            running_node(*node), gpu_id(gpu_id), request_processors(request_processors), modelsList(modelsList), latencies(latencies)
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

        // initialize SLO tracking for each model
        for(const auto& [model_name, _]:modelsList) {
            slo_total_request_count[model_name] = std::vector<float> (slo_track_size, 0.0f);
            slo_failure_rate_percent[model_name] = std::vector<float> (slo_track_size, 0.0f);
            slo_failure_rate_raw[model_name] = std::vector<float> (slo_track_size, 0.0f);
            slo_track_ind[model_name] = 0;

            inference_latency[model_name] = std::vector<float> (inference_latency_size, -1.0);
            inference_latency_ind[model_name] = 0;
        }

        // initialize batch size tracking for each model
        for(auto [model_name, _]:modelsList) {
            batch_for_model[model_name] = 0;
            batch_for_model_update[model_name] = 0;
        }

        // initialize batch fill rate tracking for each model
        batch_fill_rate = std::vector<std::vector<float>> (SLOTS_PER_GPU, std::vector<float> (batch_fill_rate_track_size, 0.0f));
        batch_fill_track_ind = std::vector<int> (batch_fill_rate_track_size, 0);

        // initialize ORTRunners for each session in node schedule
        cudaSetDevice(gpu_id);
        for(auto& [session_ptr, _]: running_node.session_list) {
            // update batch tracking
            batch_for_model[session_ptr->model_name] = session_ptr->batch_size;

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

        for(auto [model_name, _]:modelsList) {
            batch_for_model_update[model_name] = 0;
        }

        cudaSetDevice(gpu_id);
        for(auto& [session_ptr, _]: updated_node.session_list) {
            auto model_name = session_ptr->model_name;

            // update batch tracking
            batch_for_model_update[model_name] = session_ptr->batch_size;

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

    std::vector<float> get_slo_failure_persent(int num_of_schedules) {
        std::vector<float> stats_percent;
        _lock_stats.lock();
        for(auto [model_name, rates]:slo_failure_rate_percent) {
            auto ind = (slo_track_ind[model_name] - 1 + slo_track_size) % slo_track_size;
            float val = 0;
            for(int j=0;j<num_of_schedules;j++) {
                auto ref_ind = (ind - j + slo_track_size) % slo_track_size;
                val += rates[ref_ind];
            }

            stats_percent.push_back(val/num_of_schedules);
        }

        _lock_stats.unlock();
        return stats_percent;
    }

    float get_batch_fill_rate(int num_of_schedules) {
        // std::vector<float> stats_rate;
        float fill_rate = 0;
        _lock_stats.lock();
        for(int i=0;i<batch_fill_rate.size();i++) {
            auto ind = (batch_fill_track_ind[i] - 1 + batch_fill_rate_track_size) % batch_fill_rate_track_size;
            float val = 0;
            for(int j=0;j<num_of_schedules;j++) {
                auto ref_ind = (ind - j + slo_track_size) % slo_track_size;
                val += batch_fill_rate[i][ref_ind];
            }

            fill_rate += val/num_of_schedules;
        } fill_rate /= SLOTS_PER_GPU;

        _lock_stats.unlock();
        return fill_rate;
    }

    std::vector<std::vector<float>> get_slo_rate(int num_of_schedules) {
        std::vector<float> stats_percent;
        std::vector<float> stats_raw;
        std::vector<float> stats_total;

        _lock_stats.lock();
        for(auto [model_name, rates]:slo_total_request_count) {
            auto ind = (slo_track_ind[model_name] - 1 + slo_track_size) % slo_track_size;
            float val = 0;
            for(int j=0;j<num_of_schedules;j++) {
                auto ref_ind = (ind - j + slo_track_size) % slo_track_size;
                val += rates[ref_ind];
            }

            stats_total.push_back(val/num_of_schedules);
        }

        for(auto [model_name, rates]:slo_failure_rate_percent) {
            auto ind = (slo_track_ind[model_name] - 1 + slo_track_size) % slo_track_size;
            float val = 0;
            for(int j=0;j<num_of_schedules;j++) {
                auto ref_ind = (ind - j + slo_track_size) % slo_track_size;
                val += rates[ref_ind];
            }

            stats_percent.push_back(val/num_of_schedules);
        }

        for(auto [model_name, rates]:slo_failure_rate_raw) {
            auto ind = (slo_track_ind[model_name] - 1 + slo_track_size) % slo_track_size;
            float val = 0;
            for(int j=0;j<num_of_schedules;j++) {
                auto ref_ind = (ind - j + slo_track_size) % slo_track_size;
                val += rates[ref_ind];
            }

            stats_raw.push_back(val/num_of_schedules);
        }

        _lock_stats.unlock();
        return {stats_percent, stats_raw, stats_total};
    }

    std::vector<float> get_inference_latency(int num_of_schedules) {
        std::vector<float> stats;
        
        _lock_stats.lock();
        for(auto [model_name, lat]:inference_latency) {
            auto ind = (inference_latency_ind[model_name] - 1 + inference_latency_size) % inference_latency_size;
            float val = 0;
            for(int j=0;j<num_of_schedules;j++) {
                auto ref_ind = (ind - j + inference_latency_size) % inference_latency_size;
                val += lat[ref_ind];
            }

            stats.push_back(val/num_of_schedules);
        }

        _lock_stats.unlock();
        return stats;
    }

    std::vector<int> get_batch_per_model() {
        std::vector<int> batch_sizes;
        for(const auto &[_, batch_size]:batch_for_model) {
            batch_sizes.push_back(batch_size);
        }

        return batch_sizes;
    }

    bool gpu_in_use() {
        for(const auto &[_, batch_size]:batch_for_model) {
            if(batch_size > 0) return true;
        }

        return false;
    }

    float get_peak_memory() {
        return peak_mem_per_schedule;
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
    std::map<std::string, std::string> modelsList;
    std::map<std::string, std::vector<float>> slo_failure_rate_raw;
    std::map<std::string, std::vector<float>> slo_failure_rate_percent;
    std::map<std::string, std::vector<float>> slo_total_request_count;
    std::map<std::string, std::vector<float>> inference_latency;
    std::map<std::string, int> slo_track_ind;
    std::vector<std::vector<float>> batch_fill_rate;
    std::vector<int> batch_fill_track_ind;
    std::map<std::string, int> inference_latency_ind;
    int inference_latency_size = 10;
    int slo_track_size = 10;
    int batch_fill_rate_track_size = 10;
    std::mutex _lock_stats;
    std::map<std::string, double> latencies;
    std::map<std::string, int> batch_for_model_update;
    std::map<std::string, int> batch_for_model;
    int peak_mem_per_schedule = 0; // in MB
    int peak_mem_per_schedule_cur = 0; // in MB

    struct CallbackData {
        std::string _tag;
        std::shared_ptr<spdlog::logger> _logger;
        std::shared_ptr<NodeRunner> _runner;
        BatchInfo _batch_info;
        int64_t _inference_start_time;
        CallbackData(std::string tag, std::shared_ptr<spdlog::logger> logger, std::shared_ptr<NodeRunner> runner, BatchInfo batch_info, int64_t inference_start_time): 
        _tag(tag), _logger(logger), _runner(runner), _batch_info(batch_info), _inference_start_time(inference_start_time) {}
    };

    static void CUDART_CB log_callback(void* user_data) {
        auto* data = static_cast<CallbackData*>(user_data);
    
        auto now = std::chrono::high_resolution_clock::now();
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    
        //data->_logger->info("BATCH PROCESSED: {} @ {}", data->_tag, now_us);
        data->_logger->info("BATCH PROCESSED: {} @ {}", data->_tag, now_us);

        // call post process to log slo rate and latencies
        data->_runner->post_process_async(data->_tag, data->_batch_info, data->_inference_start_time, now_us);
        delete data;
    }

    void post_process_async(std::string tag, BatchInfo batch_info, int64_t inference_start_time, int64_t inference_end_time) {
        std::thread([this, tag, batch_info, inference_start_time, inference_end_time] {
            try {
                _lock_stats.lock();

                // get model name
                size_t pos = tag.find('_');
                std::string model_name = (pos == std::string::npos) ? tag : tag.substr(0, pos);

                // process slo rate
                float fail_count = batch_info._stale_req_count;
                float success_count = 0;
                double slo_latency = latencies[model_name];

                for(auto entry:batch_info._batch_timing_info) {
                    auto count = entry.first;
                    auto arrival_time = entry.second;

                    double service_latency = (double(inference_end_time - arrival_time))/1000.0;
                    if(service_latency > slo_latency) {
                        fail_count += count;
                    } else {
                        success_count += count;
                    }
                }
                
                slo_failure_rate_raw[model_name][slo_track_ind[model_name]] = fail_count;
                slo_total_request_count[model_name][slo_track_ind[model_name]] = success_count + fail_count;
                slo_failure_rate_percent[model_name][slo_track_ind[model_name]] = 0;
                if((success_count + fail_count) > 0) slo_failure_rate_percent[model_name][slo_track_ind[model_name]] = (fail_count / (success_count + fail_count)) * 100;
                slo_track_ind[model_name] = (slo_track_ind[model_name] + 1) % slo_track_size;

                // process inference latency
                double infernce_latency_cal = (double(inference_end_time - inference_start_time)) / 1000.0;
                inference_latency[model_name][inference_latency_ind[model_name]] = infernce_latency_cal;
                inference_latency_ind[model_name] = (inference_latency_ind[model_name] + 1) / inference_latency_size;

                _lock_stats.unlock();
            } catch (const std::exception& e) {
                LOG_ERROR(_logger, "Exception is post process async: {}", e.what());
            } catch (...) {
                LOG_ERROR(_logger, "Unknown exception in post process async");
            }
        }).detach();
    }

    void monitor_total_gpu_usage() {
        size_t total_mem = 0;
        size_t current_used = 0;
        size_t max_used = 0;

        while (_schedule_running.load()) {
            size_t free_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);

            current_used = total_mem - free_mem;
            if (current_used > max_used) {
                max_used = current_used;
            }

            std::this_thread::sleep_for(std::chrono::microseconds(100));  // 0.1 ms
        }

        peak_mem_per_schedule_cur = max_used;
        peak_mem_per_schedule = std::max(peak_mem_per_schedule, peak_mem_per_schedule_cur);
    }

    void run() {
        CUDACHECK(cudaSetDevice(gpu_id));
        _running = true;
        while(_running) {
            if(!_running) break;

            // reset peak memory tracking for GPU
            peak_mem_per_schedule = 0;
            _schedule_running = true;
            std::thread monitor(&NodeRunner::monitor_total_gpu_usage, this);

            // loop through all session in the schedule
            if(running_node.session_list.size() > 0) {
                // LOG_DEBUG(_logger, "Duty cycle start with session size {} on gpu {}", running_node.session_list.size(), gpu_id);
                
                // keep track of active streams to enable parallel execution
                std::vector<cudaStream_t> running_streams;
                
                // track batch fill rate per slot
                int slot_num = 0;
                
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
                    
                    auto now = std::chrono::high_resolution_clock::now();
                    int64_t inference_start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch()
                    ).count();

                    // get batch from request processor
                    auto batch_info = request_processors[session_ptr->model_name]->form_batch(session_ptr->batch_size, gpu_id);
                    auto batch_current = batch_info._batch_size;
                    
                    // update batch fill rate
                    batch_fill_rate[slot_num][batch_fill_track_ind[slot_num]] = 100.0;
                    if(session_ptr->batch_size > 0) batch_fill_rate[slot_num][batch_fill_track_ind[slot_num]] = (batch_current / session_ptr->batch_size) * 100.0;
                    batch_fill_track_ind[slot_num] = (batch_fill_track_ind[slot_num] + 1) % batch_fill_rate_track_size;

                    // launch inference using ORTRunner
                    auto ort_ptr = ort_list[i].first;
                    if(batch_current>0) {
                        ort_ptr->run_inference(batch_current);

                        // add logging callback
                        std::string id = session_ptr->model_name + "_" + std::to_string(gpu_id);
                        auto* callBackData = new CallbackData(id, _callback_logger, shared_from_this(), batch_info, inference_start_time);
                        cudaLaunchHostFunc(
                            ort_ptr->_runner_stream,
                            log_callback,
                            static_cast<void*>(callBackData) 
                        );

                        running_streams.push_back(ort_ptr->_runner_stream);
                    } slot_num++ ;
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
            
            // stop monitoring thread
            _schedule_running = false;
            
            // check if update is needded
            if(pendingUpdate) {
                _update.lock();
                    LOG_DEBUG(_logger, "Acquired lock on node to make schedule update");
                    // update node and ort list
                    running_node = update_node;
                    ort_list = update_ort_list;

                    // update batch size info
                    batch_for_model = batch_for_model_update;

                    // reset memory tracking
                    peak_mem_per_schedule = 0;
                    peak_mem_per_schedule_cur = 0;

                    pendingUpdate = false;
                    update_ort_list.clear();
                    LOG_DEBUG(_logger, "Releasing lock after update finished, new list size is {}", ort_list.size());
                _update.unlock();
            }

            // join the monitor thread
            if(monitor.joinable()) monitor.join();
        }
    }

    std::atomic<bool> _running = false;
    std::atomic<bool> _schedule_running = false;

    int gpu_id;
    Node running_node;
    std::vector<std::pair<std::shared_ptr<ORTRunner>, cudaStream_t>> ort_list;
    std::map<std::string, std::shared_ptr<RequestProcessor>> request_processors;
    std::shared_ptr<spdlog::logger> _logger;
    std::shared_ptr<spdlog::logger> _callback_logger;
};

#endif