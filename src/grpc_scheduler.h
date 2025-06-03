#include <memory>
#include <grpcpp/grpcpp.h>
#include "com/agent_scheduler.grpc.pb.h"
#include "Executor.h"
#include "Logger.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using gpu_scheduler::SchedulerSim;
using gpu_scheduler::ResetRequest;
using gpu_scheduler::ResetResponse;
using gpu_scheduler::StepRequest;
using gpu_scheduler::StepResponse;
using gpu_scheduler::ScheduleEntry;
using gpu_scheduler::StepRequestReduced;

extern int SIMULATION_LEN;

class SchedulerSimServiceImpl final : public SchedulerSim::Service {
public:
    int step_count = 0;
    explicit SchedulerSimServiceImpl(std::shared_ptr<Executor> executor): _executor(executor) {
        // get logger for grpc service
        _logger = Logger::getInstance().getLogger("grpcpp_service");
        if (!_logger) {
            std::cerr << "Failed to get logger for grpcpp_service. Exiting.\n";
            exit(EXIT_FAILURE);
        }
    }

    Status Reset(ServerContext* context, const ResetRequest* request, ResetResponse* response) {
        LOG_DEBUG(_logger, "Received RESET request from python client");
        // reset the system
        _executor->reset(request->seed());
        
        // get observation after reset
        std::vector<float> observation = _executor->get_observation();
        for(auto val:observation) {
            response->add_observation(val);
        }

        LOG_DEBUG(_logger, "Servicd RESET request from python client");
        _logger->flush();
        return Status::OK;
    }

    Status Step(ServerContext* context, const StepRequest* request, StepResponse* response) {
        LOG_DEBUG(_logger, "Received STEP request from python client");
        _logger->flush();
        // update schedule in system
        _executor->update_schedule(request);

        // wait for 1 second for state to update
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        // get observation after step
        std::vector<float> observation = _executor->get_observation();
        for(auto val:observation) {
            response->add_observation(val);
        }

        // set reward
        std::vector<float> reward = _executor->get_reward(3);
        float total_reward = 0;
        for(auto entry:reward) {
            total_reward += entry;
            response->add_info(entry);
        }
        response->set_reward(total_reward);
        
        // update step count
        step_count++;
        bool is_done = false;
        if(step_count == SIMULATION_LEN) {
            // reset simulator
            _executor->stopSimulation();
            is_done = true;
            step_count = 0;
        }

        // check if simulation done
        response->set_done(is_done);

        LOG_DEBUG(_logger, "Serviced STEP request from python client");
        _logger->flush();
        return Status::OK;
    }

    Status StepReduced(ServerContext* context, const StepRequestReduced* request, StepResponse* response) {
        LOG_DEBUG(_logger, "Received STEP REDUCED request from python client");
        _logger->flush();
        // update schedule in system
        _executor->update_schedule_per_slot(request);
        LOG_DEBUG(_logger, "Finished executor update call");

        // reset slo count for next step
        _executor->reset_slo();

        // wait for 1-3 second for state to update (based on slo times set)
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));

        // get observation after step
        std::vector<float> observation = _executor->get_observation();
        for(auto val:observation) {
            response->add_observation(val);
        }

        // set reward
        std::vector<float> reward = _executor->get_reward(3);
        float total_reward = 0;
        for(auto entry:reward) {
            total_reward += entry;
            response->add_info(entry);
        }
        response->set_reward(total_reward);
        
        // update step count
        step_count++;
        bool is_done = false;
        if(step_count == SIMULATION_LEN) {
            // reset simulator
            _executor->stopSimulation();
            is_done = true;
            step_count = 0;
        }

        // check if simulation done
        response->set_done(is_done);

        LOG_DEBUG(_logger, "Serviced STEP REDUCED request from python client");
        _logger->flush();
        
        return Status::OK;
    }

private:
    std::shared_ptr<Executor> _executor;
    std::shared_ptr<spdlog::logger> _logger;
};

void RunServer(std::shared_ptr<Executor> executor) {
    std::string server_address("0.0.0.0:50051");
    SchedulerSimServiceImpl service(executor);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "gRPC server started\n";
    server->Wait();
}