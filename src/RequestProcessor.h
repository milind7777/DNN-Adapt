#ifndef REQUESTPROCESSOR_H
#define REQUESTPROCESSOR_H

#include "concurrentqueue.h"
#include <ctime>
#include <chrono>

struct InferenceRequest {
    std::string model_name;
    std::string input_file_path;
    time_t arrival_time;

    InferenceRequest(const std::string& model, const std::string& path)
        : model_name(model), input_file_path(path),
          arrival_time(0) {}
};

class RequestProcessor {
private:
    moodycamel::ConcurrentQueue<std::shared_ptr<InferenceRequest>> queue;
    
public:
    void register_request(std::shared_ptr<InferenceRequest> req, int batch_size=1);
    std::vector<std::shared_ptr<InferenceRequest>> form_batch(int batch_size);
    size_t get_size() const;
};


#endif