#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "SchedulerInterface.h"
#include "nexus.h"
#include "NodeRunner.h"
#include "Logger.h"
#include <fstream>

class Executor {
public:
    Executor(std::map<std::string, std::string> &modelsList, 
             std::vector<std::shared_ptr<Gpu>> &gpuList,
             std::map<std::string, std::shared_ptr<RequestProcessor>> &requestProcessorList,
             std::map<std::string, double> &latencies,
             std::string profilingFolder): 
                _gpuList(gpuList), _modelsList(modelsList), _requestProcessorList(requestProcessorList), _latencies(latencies), _profilingFolder(profilingFolder)
    {
        // get logger
        _logger = Logger::getInstance().getLogger("Executor");
        if (!_logger) {
            std::cerr << "Failed to get logger for Executor(). Exiting.\n";
            exit(EXIT_FAILURE);
        }
        
        // Load SLO configuration
        // load_slo_config("/home/cching1/DNNAdapt/DNN-Adapt/util/slo_config.json");
        
        // initialize NodeRunners according to the gpuList
        LOG_DEBUG(_logger, "Initialize NodeRunners");
        for(int i=0;i<_gpuList.size();i++) {
            auto emptyNode = std::make_shared<Node>();
            _nodeRunnersList.push_back(std::make_shared<NodeRunner>(emptyNode, i, _requestProcessorList));
        }

        // initialize scheduler: currently using nexus
        LOG_DEBUG(_logger, "Initialize Scheduler");
        std::vector<std::string> modelNames;
        for(auto [name, _]: modelsList) modelNames.push_back(name);
        _scheduler = std::make_shared<NexusScheduler>(_gpuList, modelNames, _profilingFolder);
    };

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
    std::map<std::string, double> _latencies;
    std::string _profilingFolder;
    std::vector<std::shared_ptr<NodeRunner>> _nodeRunnersList;
    std::shared_ptr<Scheduler> _scheduler;
    std::thread _runner_thread;
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