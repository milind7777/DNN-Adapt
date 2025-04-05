#include "RequestProcessor.h"
#include <iostream>

/*
   NOTE: For the purpose of simulation when batch size is specified the same request is logged multiple times
*/
void RequestProcessor::register_request(std::shared_ptr<InferenceRequest> req, int batch_size) {
    for(int i=0;i<batch_size;i++) {
        auto request_copy = std::make_shared<InferenceRequest>(*req);
        request_copy->arrival_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        queue.enqueue(request_copy);
    }
}

std::vector<std::shared_ptr<InferenceRequest>> RequestProcessor::form_batch(int batch_size) {
    std::vector<std::shared_ptr<InferenceRequest>> batch;
    batch.reserve(batch_size);
    size_t count = queue.try_dequeue_bulk(std::back_inserter(batch), batch_size);
    batch.resize(count); // Actual number of dequeued items
    return batch;
}

// Approximate queue size (thread-safe)
size_t RequestProcessor::get_size() const {
    return queue.size_approx();
}