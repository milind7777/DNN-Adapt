#include "RequestProcessor.h"
#include <iostream>

/*
   NOTE: We do not care about the different inputs for each request for the purpose of these experiments, we only care about the request counts
*/
void RequestProcessor::register_request(std::shared_ptr<InferenceRequest> req) {
    // get request count
    int request_count = req->request_count;

    // recording request rate
    auto now = std::chrono::high_resolution_clock::now();
    long current_second = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto slot_index = current_second % (RATE_CALCULATION_DURATION + 1);
    Slot& slot = slots[slot_index]; 

    long expected_second = slot.last_active_second.load(std::memory_order_acquire);
    if(expected_second != current_second) {
        // reset this slot counter
        slot.counter.store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_relaxed);
        slot.last_active_second.store(current_second, std::memory_order_release);
        slot.counter.fetch_add(request_count, std::memory_order_relaxed);
    } else {
        slot.counter.fetch_add(request_count, std::memory_order_relaxed);
    }
    
    // add request to queue
    auto request_copy = std::make_shared<InferenceRequest>(*req);
    request_copy->arrival_time = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()
    ).count();;
    queue.enqueue(request_copy);
}

int RequestProcessor::form_batch(int batch_size) {
    int batch_cur = 0;
    int64_t last_request_arrival_time;
    std::vector<std::pair<int, int64_t>> batch_timing_info;
    
    // acquire lock to prevent parallel batch forming
    _lock_batch.lock();

    // try to form batch from buffer first
    if(buffer != nullptr) {
        if(buffer->request_count > batch_size) {
            buffer->request_count -= batch_size;
            batch_cur = batch_size;
            batch_timing_info.push_back({batch_size, buffer->arrival_time});
        } else {
            batch_cur = buffer->request_count;
            batch_timing_info.push_back({buffer->request_count, buffer->arrival_time});
            buffer = nullptr;
        }
    }

    // try to form from the request queue
    while(batch_cur < batch_size) {
        std::shared_ptr<InferenceRequest> request;
        if(queue.try_dequeue(request)) {
            last_request_arrival_time = request->arrival_time;
            batch_timing_info.push_back({std::min(request->request_count, batch_size - batch_cur), request->arrival_time});
            batch_cur += request->request_count;
        } else {
            break;
        }
    }

    if(batch_cur > batch_size) {
        buffer = std::make_shared<InferenceRequest>(model_name, batch_cur-batch_size, last_request_arrival_time);
        batch_cur = batch_size;
    }
    
    // release lock
    _lock_batch.unlock();

    // log the batch timing info
    std::cout << "BATCH FORMED: " << model_name << std::endl;
    for(auto entry:batch_timing_info) {
        std::cout << "request_count: " << entry.first << ", arrival_time: " << entry.second << std::endl;
    }
    
    return batch_cur;
}

// Approximate queue size (thread-safe) - useless now
size_t RequestProcessor::get_size() const {
    return queue.size_approx();
}

double RequestProcessor::get_request_rate() {
    const auto now = std::chrono::high_resolution_clock::now();
    auto time_since_epoch = now.time_since_epoch();
    auto current_second = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch).count();

    double total_count = 0;
    for(int i=0;i<=RATE_CALCULATION_DURATION;i++) {
        const long slot_second = slots[i].last_active_second.load(std::memory_order_acquire);

        if((current_second - slot_second) < (RATE_CALCULATION_DURATION+1)) {
            // ignore for current second    
            if(slot_second != current_second) {
                total_count += slots[i].counter.load(std::memory_order_relaxed);
            }
        }
    }

    // return total request count per second
    return total_count / RATE_CALCULATION_DURATION;
}