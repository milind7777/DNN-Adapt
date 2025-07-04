#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <map>
#include <string>
#include <iostream>
#include "RequestProcessor.h"
#include "Logger.h"

enum class rate_type {
    flat,
    ramp,
    burst,
    sinusoidal,
    exponential_decay
};

class Simulator {
private:
    std::map<std::string, std::shared_ptr<RequestProcessor>> request_processors;
    std::shared_ptr<spdlog::logger> _logger;
    std::atomic<bool> stop_flag = false;
    std::atomic<bool> done_flag = false;

    void dynamic_request_rate_generator(std::shared_ptr<RequestProcessor> req_processor, std::vector<std::pair<rate_type, std::pair<double, std::vector<double>>>> &schedule, std::shared_ptr<InferenceRequest> request);

public:
    Simulator(std::map<std::string, std::shared_ptr<RequestProcessor>> _request_processors): request_processors(_request_processors) {
        _logger = Logger::getInstance().getLogger("Simulator");
        if (!_logger) {
            std::cerr << "Failed to get logger. Exiting.\n";
            exit(EXIT_FAILURE);
        }

        std::srand(7);
    };
    
    void reset() {
        stop_flag = true;
    }

    bool isDone() {
        return done_flag;
    }

    void run(int seed);
};

#endif