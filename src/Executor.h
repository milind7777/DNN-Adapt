#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "SchedulerInterface.h"
#include "nexus.h"
#include "NodeRunner.h"
#include "Logger.h"
#include "Simulator.h"
#include <fstream>
#include "com/agent_scheduler.grpc.pb.h"

using gpu_scheduler::StepRequest;
using gpu_scheduler::ScheduleEntry;

class Executor {
public:
    Executor(std::map<std::string, std::string> &modelsList, 
             std::vector<std::shared_ptr<Gpu>> &gpuList,
             std::map<std::string, std::shared_ptr<RequestProcessor>> &requestProcessorList,
             std::shared_ptr<Simulator> simulator,
             std::map<std::string, double> &latencies,
             std::string profilingFolder): 
                _gpuList(gpuList), _modelsList(modelsList), _requestProcessorList(requestProcessorList), _simulator(simulator), _latencies(latencies), _profilingFolder(profilingFolder)
    {
        // get logger
        _logger = Logger::getInstance().getLogger("Executor");
        if (!_logger) {
            std::cerr << "Failed to get logger for Executor(). Exiting.\n";
            exit(EXIT_FAILURE);
        }
        
        // set id to model vector
        for(const auto &[model_name, _]:_modelsList) {
            id_to_model.push_back(model_name);
        }

        // Load SLO configuration
        // load_slo_config("/home/cching1/DNNAdapt/DNN-Adapt/util/slo_config.json");
        
        // initialize NodeRunners according to the gpuList
        LOG_DEBUG(_logger, "Initialize NodeRunners");
        for(int i=0;i<_gpuList.size();i++) {
            auto emptyNode = std::make_shared<Node>();
            _nodeRunnersList.push_back(std::make_shared<NodeRunner>(emptyNode, i, _modelsList, _requestProcessorList, _latencies));
        }

        // initialize scheduler: currently using nexus
        LOG_DEBUG(_logger, "Initialize Scheduler");
        std::vector<std::string> modelNames;
        for(auto [name, _]: modelsList) modelNames.push_back(name);
        _scheduler = std::make_shared<NexusScheduler>(_gpuList, modelNames, _profilingFolder);
    };

    bool isDone() {
        return _simulator->isDone();
    }

    void reset(int seed) {
        LOG_DEBUG(_logger, "Received RESET in Executor");
        // stop simulation
        _simulator->reset();
        if(_simulation_thread.joinable()) {
            LOG_DEBUG(_logger, "Calling join on simulator thread");
            _simulation_thread.join();
        }

        // clear request queues
        for(const auto &[model_name, processor]:_requestProcessorList) {
            processor->clear_queue();
        }

        // stop node runners
        for(auto runner:_nodeRunnersList) {
            runner->stop();
        }
        
        // clear node runners
        for(auto runner:_nodeRunnersList) {
            Node emptyNode;
            runner->updateNode(emptyNode);
        }

        // restart simulation
        LOG_DEBUG(_logger, "Restarting simulator thread");
        _simulation_thread = std::thread(&Simulator::run, _simulator.get(), seed);
    
        // restart node runners
        for(auto runner:_nodeRunnersList) {
            runner->start();
        }
    }

    std::vector<float> get_observation() {
        LOG_DEBUG(_logger, "Received observation request on Executor");
        // For each model
        // 1. Request rate - float (normalized)
        LOG_DEBUG(_logger, "Fetching request rates");
        std::vector<float> req_rates;
        for(const auto &[_, processor]:_requestProcessorList) {
            req_rates.push_back(processor->get_request_rate());
        }

        // 2. SLO latency in ms - float (normalized)
        std::vector<float> slo;
        for(const auto &[_, latency]:_latencies) {
            slo.push_back(latency);
        }

        // 3. SLO latency satisfaction % - array of [float] of size num_gpus (normalized)
        LOG_DEBUG(_logger, "Fetching slo rates");
        std::vector<std::vector<float>> slo_rate;
        for(auto runner:_nodeRunnersList) {
            slo_rate.push_back(runner->get_slo_failure_persent(3));
        }
        
        // 4. GPU locations - array of [int] of size num_gpus - if batch size is zero then model is not present
        // 5. Batch size - array of [int] of size
        LOG_DEBUG(_logger, "Fetching batch sizes");
        std::vector<std::vector<int>> batch_sizes;
        for(auto runner:_nodeRunnersList) {
            batch_sizes.push_back(runner->get_batch_per_model());
        }

        // For each GPU
        // 1. Peak memory in MB per schedule - float
        std::vector<float> peak_mem;
        LOG_DEBUG(_logger, "Fetching GPU memory usage");
        for(auto runner:_nodeRunnersList) {
            peak_mem.push_back(runner->get_peak_memory());
        }

        // 2. % GPU utilized per schedule - float (Hard to do skipping for now)

        // reconstruct observation: per model there should be 1+1+gpus+gpus+gpus entries
        std::vector<float> observation;
        LOG_DEBUG(_logger, "Reconstructing observation from fetched data");
        for(int i=0;i<_modelsList.size();i++) {
            observation.push_back(req_rates[i]);
            observation.push_back(slo[i]);
            
            for(int j=0;j<_gpuList.size();j++) {
                observation.push_back(slo_rate[j][i]);
            }

            for(int j=0;j<_gpuList.size();j++) {
                bool is_present = (batch_sizes[j][i] > 0);
                observation.push_back(is_present);
            }

            for(int j=0;j<_gpuList.size();j++) {
                observation.push_back(batch_sizes[j][i]);
            }
        }

        observation.insert(observation.end(), peak_mem.begin(), peak_mem.end());
        return observation;
    }

    float get_reward(int num_schedules) {
        // Reward = - alpha * slo failure rate
        //          - beta  * num of GPUs
        //          - gamma * batch fill rate
        //
        // alpha = 1.0f, beta = 0.5f, // gamma = 0.5f
        float alpha = 1.0f;
        float beta  = 0.5f;
        float gamma = 0.5f;
        LOG_DEBUG(_logger, "Fetching slo rates to calculate reward");
        std::vector<std::vector<std::vector<float>>> slo_rate;
        for(auto runner:_nodeRunnersList) {
            slo_rate.push_back(runner->get_slo_rate(num_schedules));
        }

        float total_request_count = 0;
        float total_fail_count_weighted = 0;
        for(auto& rates:slo_rate) {
            auto& per = rates[0];
            auto& raw = rates[1];
            auto& total = rates[2];

            for(int i=0;i<per.size();i++) {
                float count = total[i];
                total_request_count += count;
                total_fail_count_weighted += per[i] * count;
            }
        }

        float slo_penalty = 0;
        if(total_request_count > 0) slo_penalty = (total_fail_count_weighted / total_request_count) / 100.0;
        
        float gpu_count = 0;
        for(auto runner:_nodeRunnersList) gpu_count += runner->gpu_in_use();
        gpu_count /= _gpuList.size();
        
        float batch_fill_penalty = 0;
        for(auto runner:_nodeRunnersList) {
            batch_fill_penalty += runner->get_batch_fill_rate(3);
        } batch_fill_penalty /= (_nodeRunnersList.size() * 100.0);

        return - alpha * slo_penalty - beta * gpu_count - gamma * batch_fill_penalty;
    }

    void update_schedule(const StepRequest *request) {
        std::vector<int> batch_deltas;
        for(int delta:request->batch_deltas()) {
            batch_deltas.push_back(delta);
        }

        std::vector<std::vector<int>> batch_sizes;
        for(auto runner:_nodeRunnersList) {
            batch_sizes.push_back(runner->get_batch_per_model());
        }
        
        int count = 0;
        int gpu_id;
        int num_gpus = _gpuList.size();
        int num_models = _modelsList.size();
        std::vector<std::pair<std::shared_ptr<Session>, std::pair<double, bool>>> session_list;
        for(const auto& entry:request->entries()) {
            int model_id = entry.model_id();
            gpu_id = count / num_models;

            if(model_id != num_models) {
                std::string model_name = id_to_model[model_id];

                bool in_parallel = entry.in_parallel();

                int new_batch_size = std::max(0, batch_sizes[gpu_id][model_id] + batch_deltas[gpu_id * num_models + model_id]);
                
                if(new_batch_size>0) {
                    std::shared_ptr<Session> session = std::make_shared<Session>(model_name, _latencies[model_name], 0, new_batch_size);
                    session_list.push_back({session, {0, in_parallel}});
                }
            }

            count++;
            if(count % num_models == 0) {
                // make the node and call update
                Node new_node(session_list);
                _nodeRunnersList[gpu_id]->updateNode(new_node);

                session_list.clear();
            }
        }

        std::cout << "Done with update\n";
    }

    void start() {
        // start the loop for continuously checking the request rates and updating the schedule
        if(!_running) {
            LOG_DEBUG(_logger, "Launching Nodes");
            for(auto runner:_nodeRunnersList) {
                runner->start();
            }

            LOG_DEBUG(_logger, "Launching executor run");
            _runner_thread = std::thread(&Executor::run, this);
        }
    };

    void stop() {
        _running = false;
        if(_runner_thread.joinable()) {
            _runner_thread.join();
        }
    }

    ~Executor() {
        if(_running) stop();
    }

private:
    std::vector<std::shared_ptr<Gpu>> _gpuList;
    std::map<std::string, std::string> _modelsList;
    std::map<std::string, std::shared_ptr<RequestProcessor>> _requestProcessorList;
    std::shared_ptr<Simulator> _simulator;
    std::map<std::string, double> _latencies;
    std::string _profilingFolder;
    std::vector<std::shared_ptr<NodeRunner>> _nodeRunnersList;
    std::shared_ptr<Scheduler> _scheduler;
    std::thread _runner_thread;
    std::thread _simulation_thread;
    std::vector<std::string> id_to_model;
    bool _running = false;
    const int _interval = 5; // in seconds
    std::shared_ptr<spdlog::logger> _logger;

    std::map<std::string, double> _model_slos_us; // SLO thresholds in microseconds
    double _default_slo_us = 10000.0; // Default SLO threshold
    
    // void load_slo_config(const std::string& config_path) {
    //     try {
    //         std::ifstream file(config_path);
    //         if (!file.is_open()) {
    //             LOG_WARN(_logger, "Could not open SLO config file: {}. Using default SLOs.", config_path);
    //             return;
    //         }
            
    //         nlohmann::json config;
    //         file >> config;
            
    //         if (config.contains("model_slos_us")) {
    //             for (auto& [model, slo] : config["model_slos_us"].items()) {
    //                 _model_slos_us[model] = slo.get<double>();
    //                 LOG_INFO(_logger, "Loaded SLO for {}: {} μs", model, slo.get<double>());
    //             }
    //         }
            
    //         if (config.contains("default_slo_us")) {
    //             _default_slo_us = config["default_slo_us"].get<double>();
    //             LOG_INFO(_logger, "Loaded default SLO: {} μs", _default_slo_us);
    //         }
    //     } catch (std::exception& e) {
    //         LOG_ERROR(_logger, "Error loading SLO config: {}", e.what());
    //     }
    // }
    
    std::vector<std::shared_ptr<Session>> generate_sessions() {
        std::vector<std::shared_ptr<Session>> sessionList;
        for(auto [model_name, processor]:_requestProcessorList) {
            auto request_rate = processor->get_request_rate();
            sessionList.push_back(std::make_shared<Session>(model_name, _latencies[model_name], request_rate));
            LOG_DEBUG(_logger, "got {} requests per second for {}", request_rate, model_name);
        }

        return sessionList;
    };

    void run() {
        _running = true;
        // query request processors at given frequency to update schedule
        while(true) {
            if(!_running) break;
            
            // get session list by querying request processors
            LOG_DEBUG(_logger, "Querying request processors");
            auto sessionList = generate_sessions();

            // generate new shcedule
            auto nodeList = _scheduler->generate_schedule(sessionList);

            // check to make sure we are not using more than allowed GPUs
            if (nodeList.size() > _gpuList.size()) {
                std::cerr << "Assertion failed: nodeList.size() (" << nodeList.size()
                          << ") > _gpuList.size() (" << _gpuList.size() << ")" << std::endl;
                assert(false);
            }          

            /* // hack to test parallel execution on gpu streams
            if(nodeList.size() > 0) {
                for(int i=0;i<nodeList[0]->session_list.size();i++) {
                    nodeList[0]->session_list[i].second.second = 1;
                }
            } */

            // update NodeRunners with new schedule
            for(int i=0;i<nodeList.size();i++) {
                _nodeRunnersList[i]->updateNode(*nodeList[i]);
            }

            // pause before repeating
            std::this_thread::sleep_for(std::chrono::seconds(_interval));
        }
    }
};

#endif