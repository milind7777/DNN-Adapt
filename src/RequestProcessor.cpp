#include "RequestProcessor.h"
#include <iostream>
#include <filesystem>
#include "csv.h"

/*
   NOTE: We do not care about the different inputs for each request for the purpose of these experiments, we only care about the request counts
*/
void RequestProcessor::register_request(std::shared_ptr<InferenceRequest> req) {
    // get request count
    int request_count = req->request_count;

    _lock_size.lock();
    queue_size += request_count;
    _lock_size.unlock();

    // recording request rate
    auto now = std::chrono::high_resolution_clock::now();
    long current_second = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto slot_index = current_second % (RATE_CALCULATION_DURATION + 1);
    Slot& slot = slots[slot_index]; 

    // Log each incoming request with count and timestamp for per-second tracking
    auto time_now = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    LOG_INFO(_logger, "REQUEST RECEIVED: model_name:{} request_count:{} time_now:{}", model_name, request_count, time_now);

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
    ).count();
    queue.enqueue(request_copy);
}

std::vector<double> loadCSVFile(const std::string &filePath, const std::string column) {
    std::vector<double> latencies(MAX_BATCH_SIZE+1, 0.0);
    io::CSVReader<2> csvReader(filePath);
    csvReader.read_header(io::ignore_extra_column, BATCH_COLUMN, LATENCY_COLUMN);

    int batchSize;
    int rowCount = 0;
    double latency;

    while(csvReader.read_row(batchSize, latency)) {
        if(batchSize >= 1 && batchSize <= MAX_BATCH_SIZE) {
            latencies[batchSize] = latency;
            rowCount++;
        } else {
            std::cerr << "Warning: batch_size " << batchSize << " is out of bounds" << std::endl;
        }
    } latencies.resize(rowCount+1);

    return latencies;
}

std::map<std::string, std::map<std::string, std::vector<double>>> loadProfile(const std::string &folderPath) {
    std::map<std::string, std::map<std::string, std::vector<double>>> model_profiles;
    for(const auto &entry: std::filesystem::directory_iterator(folderPath)) {
        if(entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string filePath = entry.path().string();
            std::string modelName = entry.path().stem().string();

            model_profiles[modelName][LATENCY_COLUMN] = loadCSVFile(filePath, LATENCY_COLUMN); 
        }
    }

    return model_profiles;
}

std::map<std::string, std::map<std::string, std::vector<double>>> RequestProcessor::model_profiles = loadProfile("models/profiles/system");

BatchInfo RequestProcessor::form_batch(int batch_size, int gpu_id) {
    int batch_cur = 0;
    int stale_req = 0;
    int64_t last_request_arrival_time;
    std::vector<std::pair<int, int64_t>> batch_timing_info;
    
    // acquire lock to prevent parallel batch forming
    LOG_DEBUG(_logger, "WAITING TO ACQUIRE LOCK");
    _lock_batch.lock();
    if(batch_size == 0) LOG_DEBUG(_logger, "BATCH 0: form batch with batch 0");

    // get current time
    auto now = std::chrono::high_resolution_clock::now();
    auto time_now = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()
    ).count();

    // get time within which batch will be processed
    // auto est_process_time = time_now + (int64_t)((latency_slo / 2) * 1000);
    auto est_process_time = time_now + model_profiles[model_name][LATENCY_COLUMN][batch_size] * 1000;

    // try to form batch from buffer first
    if(buffer != nullptr) {
        // discard buffer is stale
        auto request_slo_time = buffer->arrival_time + (int64_t)(latency_slo * 1000);
        if(batch_size == 0) LOG_DEBUG(_logger, "BATCH 0: buffer not null: {}, {}", est_process_time, request_slo_time);
        if(est_process_time > request_slo_time) {
            LOG_WARN(_logger, "SLO VIOLATED: model_name:{} request_count:{} time_now:{}", model_name, buffer->request_count, time_now);
            stale_req += buffer->request_count;
            buffer = nullptr;
        } else {
            if(batch_size == 0) {
                _lock_batch.unlock();
                return BatchInfo(0, 0, {});
            }

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
    }

    // try to form from the request queue
    while(batch_cur <= batch_size) {
        std::shared_ptr<InferenceRequest> request;
        if(queue.try_dequeue(request)) {
            // discard stale requests
            auto request_slo_time = request->arrival_time + (int64_t)(latency_slo * 1000);
            if(batch_size == 0) LOG_DEBUG(_logger, "BATCH 0: trying from request queue: {}, {}", est_process_time, request_slo_time);
            if(est_process_time >= request_slo_time) {
                LOG_WARN(_logger, "SLO VIOLATED: model_name:{} request_count:{} time_now:{}", model_name, request->request_count, time_now);
                stale_req += request->request_count;
            } else {
                last_request_arrival_time = request->arrival_time;
                batch_timing_info.push_back({std::min(request->request_count, batch_size - batch_cur), request->arrival_time});
                batch_cur += request->request_count;
            }
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
    if(batch_cur>0) {
        LOG_INFO(_logger, "BATCH FORMED: id:{}_{} size: {}", model_name, gpu_id, batch_cur);
        for(auto entry:batch_timing_info) {
            LOG_INFO(_logger, "{}_{}: request_count: {}, arrival_time: {}", model_name, gpu_id, entry.first, entry.second);
        }
    }
    
    if(batch_size == 0) LOG_DEBUG(_logger, "BATCH 0: stale: {}, queue size: {}", stale_req, get_size());
    _lock_size.lock();
    queue_size -= batch_cur;
    _lock_size.unlock();
    return BatchInfo(batch_cur, stale_req, batch_timing_info);
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

void RequestProcessor::clear_queue() {
    // reset queue by redeclaring it
    LOG_DEBUG(_logger, "Queue clear called");
    _lock_batch.lock();
    buffer = nullptr;
    queue = moodycamel::ConcurrentQueue<std::shared_ptr<InferenceRequest>>();
    queue_size = 0;
    _lock_batch.unlock();
}

int RequestProcessor::get_queue_size() {
    return queue_size;
}