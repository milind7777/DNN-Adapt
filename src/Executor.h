#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "SchedulerInterface.h"
#include "nexus.h"
#include "NodeRunner.h"

class Executor {
public:
    Executor(std::map<std::string, std::string> &modelsList, 
             std::vector<std::shared_ptr<Gpu>> &gpuList,
             std::map<std::string, std::shared_ptr<RequestProcessor>> &requestProcessorList,
             std::map<std::string, double> &latencies,
             std::string profilingFolder): 
                _gpuList(gpuList), _modelsList(modelsList), _requestProcessorList(requestProcessorList), _latencies(latencies), _profilingFolder(profilingFolder)
    {
        // initialize NodeRunners according to the gpuList
        for(int i=0;i<_gpuList.size();i++) {
            auto emptyNode = std::make_shared<Node>();
            _nodeRunnersList.push_back(std::make_shared<NodeRunner>(emptyNode, i, _requestProcessorList));
        }

        // initialize scheduler: currently using nexus
        std::vector<std::string> modelNames;
        for(auto [name, _]: modelsList) modelNames.push_back(name);
        _scheduler = std::make_shared<NexusScheduler>(_gpuList, modelNames, _profilingFolder);
    };

    void start() {
        // start the loop for continuously checking the request rates and updating the schedule
        if(!_running) {
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
    bool _running;
    const int _interval = 5; // in seconds

    std::vector<std::shared_ptr<Session>> generate_sessions() {
        std::vector<std::shared_ptr<Session>> sessionList;
        for(auto [model_name, processor]:_requestProcessorList) {
            sessionList.push_back(std::make_shared<Session>(model_name, _latencies[model_name], processor->get_request_rate()));
        }

        return sessionList;
    };

    void run() {
        _running = true;
        // query request processors at given frequency to update schedule
        while(true) {
            if(!_running) break;
            
            // get session list by querying request processors
            auto sessionList = generate_sessions();

            // generate new shcedule
            auto nodeList = _scheduler->generate_schedule(sessionList);

            // check to make sure we are not using more than allowed GPUs
            assert(nodeList.size() <= _gpuList.size());

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