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
    int request_count;
    int64_t arrival_time; // set at executor before system puts it in a queue

    InferenceRequest(const std::string& model, int request_count, int64_t arrival_time=0)
        : model_name(model), request_count(request_count),
          arrival_time(0) {}
};

struct Slot {
    std::atomic<long> last_active_second{-1};
    std::atomic<int> counter{0};
};

class RequestProcessor {
private:
    moodycamel::ConcurrentQueue<std::shared_ptr<InferenceRequest>> queue;
    std::shared_ptr<InferenceRequest> buffer = nullptr;
    std::array<Slot, (size_t)(RATE_CALCULATION_DURATION+1)> slots;
    std::string model_name;
    int _id = 0;
    std::mutex _lock_batch;
    
public:
    void register_request(std::shared_ptr<InferenceRequest> req);
    std::vector<std::shared_ptr<InferenceRequest>> form_batch(int batch_size);
    size_t get_size() const;
    double get_request_rate();
};


#endif