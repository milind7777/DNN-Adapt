#include "RequestProcessor.h"
#include <iostream>

/*
   NOTE: For the purpose of simulation when batch size is specified the same request is logged multiple times
*/
void RequestProcessor::register_request(std::shared_ptr<InferenceRequest> req, int batch_size) {
    // recording request rate
    auto now = std::chrono::steady_clock::now();
    long current_second = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto slot_index = current_second % (RATE_CALCULATION_DURATION + 1);
    Slot& slot = slots[slot_index]; 

    long expected_second = slot.last_active_second.load(std::memory_order_acquire);
    if(expected_second != current_second) {
        // reset this slot counter
        slot.counter.store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_relaxed);
        slot.last_active_second.store(current_second, std::memory_order_release);
        slot.counter.fetch_add(batch_size, std::memory_order_relaxed);
    } else {
        slot.counter.fetch_add(batch_size, std::memory_order_relaxed);
    }
    
    // add requests to queue
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

double RequestProcessor::get_request_rate() {
    const auto now = std::chrono::steady_clock::now();
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