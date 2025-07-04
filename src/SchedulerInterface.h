#ifndef SCHEDULERINTERFACE_H
#define SCHEDULERINTERFACE_H

#include <string>
#include <vector>
#include <iostream>
#include <memory>

const std::string LATENCY_COLUMN = "avg_latency_ms";
const std::string BATCH_COLUMN = "batch_size";
const int MAX_BATCH_SIZE = 2048;

struct Session {
    std::string model_name;
    double latency; // in ms
    double request_rate; // in req/sec
    int batch_size;

    Session(const std::string &model_name, const double latency, const double request_rate, const int batch_size=0):
        model_name(model_name), latency(latency), request_rate(request_rate), batch_size(batch_size) {};
};

struct Gpu {
    std::string gpu_type;
    int gpu_mem; // in GB

    Gpu(const std::string &gpu_type, int gpu_mem): gpu_type(gpu_type), gpu_mem(gpu_mem) {};
};

struct Node {
    // each session has 2 attributes associated to it:
    // 1. occupancy ratio in the duty cycle
    // 2. whether the session can be run in parallel or not
    std::vector<std::pair<std::shared_ptr<Session>, std::pair<double, bool>>> session_list;
    double duty_cycle; // in ms
    Gpu gpu = Gpu("A6000", 48);

    Node(const std::vector<std::pair<std::shared_ptr<Session>, std::pair<double, bool>>> session_list = {},
        double duty_cycle = 0.0): session_list(session_list), duty_cycle(duty_cycle) {}

    double getOccupancy() {
        double total_occupancy = 0.0;
        for(int i=0;i<session_list.size();i++) {
            total_occupancy += session_list[i].second.first;
        }
        return total_occupancy;
    };

    void pretty_print() {
        std::cout << "****************************************************************************" << std::endl;
        std::cout << "    GPU TYPE: " << gpu.gpu_type << "  |  GPU MEMORY: " << gpu.gpu_mem << "GB | DUTY CYCLE: " << duty_cycle << "ms" << std::endl;
        std::cout << "    Session List: {model_name, SLO, request_rate, batch_size, occupancy}" << std::endl;
        for(auto [session, attr]:session_list) {
            std::cout << "          " << session->model_name << ", " << session->latency << ", " << session->request_rate << ", " << session->batch_size << ", " << attr.first << std::endl;
        }
        std::cout << "****************************************************************************" << std::endl;
    }
};

class Scheduler {
    public:
        Scheduler(const std::vector<std::shared_ptr<Gpu>>& gpus,
                  const std::vector<std::string> models,
                  const std::string profiling_folder): gpus_(gpus), models_(models), profiling_folder_(profiling_folder) {};
    
        virtual ~Scheduler() = default;
    
        // Generate schedule - must be implemented by derived classes
        virtual std::vector<std::shared_ptr<Node>> generate_schedule(const std::vector<std::shared_ptr<Session>>& sessions) = 0;
    
    protected:
        std::vector<std::shared_ptr<Gpu>> gpus_;
        std::vector<std::string> models_;
        std::string profiling_folder_;
    };

#endif