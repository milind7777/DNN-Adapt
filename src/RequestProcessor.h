#ifndef REQUESTPROCESSOR_H
#define REQUESTPROCESSOR_H

#include "concurrentqueue.h"
#include <ctime>
#include <chrono>
#include <atomic>
#include <memory>

const int RATE_CALCULATION_DURATION = 3; // the time window in seconds over which request rate is calculated

struct InferenceRequest {
    std::string model_name;
    std::string input_file_path;
    time_t arrival_time;

    InferenceRequest(const std::string& model, const std::string& path)
        : model_name(model), input_file_path(path),
          arrival_time(0) {}
};

struct Slot {
    std::atomic<long> last_active_second{-1};
    std::atomic<int> counter{0};
};

class RequestProcessor {
private:
    moodycamel::ConcurrentQueue<std::shared_ptr<InferenceRequest>> queue;
    std::array<Slot, (size_t)(RATE_CALCULATION_DURATION+1)> slots;
    
public:
    void register_request(std::shared_ptr<InferenceRequest> req, int batch_size=1);
    std::vector<std::shared_ptr<InferenceRequest>> form_batch(int batch_size);
    size_t get_size() const;
    double get_request_rate();
};


#endif