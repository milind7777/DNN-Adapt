#include "Simulator.h"
#include <iostream>
#include <cmath>
#include <random>

// The second level granularity for the request rate simulator
double QUANTIZATION_INTERVAL = 0.5;

/*
    Dynamic request rate simulator
    Each element of schedule is as follows:
    {rate_type, {duration in seconds, {list of variables in order specified below}}}

    Options in dynamic rate contorller:
        * Flat - Maintains constant request rate for specified duration: FLAT_RATE
        * Ramp - Modifies request rate by a constant factor for specified duration: RAMP_RATE
        * Sinusoidal - Simulates sinusoidal wave around the rate at the beginning of this mode: AMPLITUDE, FREQUENCY
        * Burst(Gaussian Pulse) - Simulates a burst of requests in the specified duration: AMPLITUDE, STDDEV
        * Exponential decay - Changes request rate by a given factor per second in the specified duration: DECAY_FACTOR
*/

void Simulator::dynamic_request_rate_generator(std::shared_ptr<RequestProcessor> req_processor, std::vector<std::pair<rate_type, std::pair<double, std::vector<double>>>> &schedule, std::shared_ptr<InferenceRequest> request) {
    using namespace std::chrono;
    
    double steady_rate = 0;
    for(auto block:schedule) {
        auto block_duration  = block.second.first;
        auto rate_type = block.first;
        auto iteration_count = int(block_duration / QUANTIZATION_INTERVAL);
        auto block_vars = block.second.second;
        auto residual_fraction = 0;
        for(int i=0;i<iteration_count;i++) {
            // stop if simulator is reset
            if(stop_flag) break;
            
            // register start time
            auto start_time = steady_clock::now();
            
            // get request rate based on rate type
            double request_rate;
            switch(rate_type) {
                case rate_type::flat: {
                    request_rate = block_vars[0];
                    steady_rate  = request_rate;
                    break;
                }
                case rate_type::ramp: {
                    request_rate = steady_rate + (block_vars[0] * QUANTIZATION_INTERVAL);
                    steady_rate  = request_rate;
                    break;
                }
                case rate_type::sinusoidal: {
                    auto amplitude = block_vars[0];
                    auto frequency = block_vars[1];
                    request_rate = steady_rate + int(amplitude * std::sin(2 * M_PI * frequency * (i * QUANTIZATION_INTERVAL)));
                    break;
                }
                case rate_type::burst: {
                    auto amplitude = block_vars[0];
                    auto stddev    = block_vars[1];
                    request_rate = steady_rate + int(amplitude * std::exp(-std::pow(((i - iteration_count/2) * QUANTIZATION_INTERVAL), 2) / (2 * std::pow(stddev, 2))));
                    break;
                }
                case rate_type::exponential_decay: {
                    auto iteratiosn_per_second = int(1 / QUANTIZATION_INTERVAL);
                    if(i%iteratiosn_per_second == 0) {
                        auto decay_factor = block_vars[0];
                        request_rate = request_rate * int(decay_factor);
                        steady_rate  = request_rate;
                    }
                    break;
                }
            }
            
            const auto now = std::chrono::high_resolution_clock::now();
            auto time_since_epoch = now.time_since_epoch();
            auto current_second = std::chrono::duration_cast<std::chrono::microseconds>(time_since_epoch).count();
            
            if(request_rate>0) {
                // logging request rate
                LOG_TRACE(_logger, "Simulating rate: {} for model {}", request_rate, request->model_name);
                LOG_TRACE(_logger, "Simulation Current Rate @ Current time: {} for model {} is @ rate {}", current_second, request->model_name, request_rate);
            }

            // get total request count by adding fractional count leftover from previous block
            auto request_count = request_rate * QUANTIZATION_INTERVAL + residual_fraction;
            double request_count_floor;
            residual_fraction = std::modf(request_count, &request_count_floor);

            // register requests
            request->request_count = request_count_floor;
            if(request->request_count>0) req_processor->register_request(request);

            // put thread to sleep for remaining time
            auto end_time = steady_clock::now();
            auto sleep_time = int(QUANTIZATION_INTERVAL * 1000) - duration_cast<milliseconds>(end_time - start_time).count();
            // std::cout << "sleep time: " << sleep_time << std::endl;
            if(sleep_time > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            }
        } if (stop_flag) {
            break;
        }
    }
}

std::srand(7);
void Simulator::run(int seed) {
    /*
        Avaialble model list:
        * efficientnetb0
        * resnet18
        * vit16
    */

    std::map<std::string, std::vector<std::pair<rate_type, std::pair<double, std::vector<double>>>>> schedules;
    
    schedules["efficientnetb0"] = {
        {rate_type::flat, {5, {0}}},
        {rate_type::flat, {115, {1}}}
        // {rate_type::ramp, {5, {-10}}}
    };

    schedules["resnet18"] = {
        {rate_type::flat, {5, {0}}},
        {rate_type::flat, {115, {1}}}
        // {rate_type::flat, {40, {10}}},
        // {rate_type::exponential_decay, {20, {0.8}}}
    };

    schedules["vit16"] = {
        {rate_type::flat, {5, {0}}},
        {rate_type::flat, {115, {1}}}
        // {rate_type::flat, {30, {10}}},
        // {rate_type::exponential_decay, {20, {0.8}}}
    };

    int min = 0;
    int max = 50;
    schedules["efficientnetb0"][1].second.second[0] *= (std::rand() % (max - min + 1)) + min;
    schedules["resnet18"][1].second.second[0] *= (std::rand() % (max - min + 1)) + min;
    schedules["vit16"][1].second.second[0] *= (std::rand() % (max - min + 1)) + min;

    // reset flags
    stop_flag = false;
    done_flag = false;

    // Launch threads for each model
    std::vector<std::thread> threads;
    for(const auto& [model_name, processor]: request_processors) {
        auto request = std::make_shared<InferenceRequest>(model_name, 1);
        threads.emplace_back(&Simulator::dynamic_request_rate_generator, this, processor, std::ref(schedules[model_name]), request);
    }

    // Wait for thread to finish
    for(auto &t:threads) {
        t.join();
    }

    // set flag
    done_flag = true;
}