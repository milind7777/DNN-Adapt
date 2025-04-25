#ifndef NODERUNNER_H
#define NODERUNNER_H

#include "SchedulerInterface.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <memory>
#include <cuda_runtime.h>

class ORTRunner {
    // This class runs the onnx runtime, each instance will have a ort session, path to onnx file, cuda stream associated with it
    int x;
};

class NodeRunner {
    // This class manages complete execution on 1 whole GPU

    // what should the node runner contain?
    // 1. node object with session list
    // 2. list of ort runtime sessions and cuda streams for each model in the session list
    //    * This should be created before schedule launch and recreated on schedule update
    // 3. gpu id
    // 4. object to hold the updated node schedule
    // 5. bool flag to represent there is pending update, needs to be thread safe
    // 6. mutex for above thread safety

    // what should the node runner expose?
    // 1. updateNode(node): To update the schedule for the node runner
    // 2. start(): To start running the current schedule list

public:
    void updateNode(Node updated_node) {
        _update.lock();
        new_node = updated_node;
        pedningUpdate = true;
        _update.unlock();
    }

    void start() {
        if(!_running) _runner_thread = std::thread(&NodeRunner::run, this);
    }

    void stop() {
        _running = false;
        if(_runner_thread.joinable()) {
            _runner_thread.join();
        }
    }

private:
    std::mutex _update;
    bool pedningUpdate;
    Node new_node;

    void run() {
        _running = true;
        while(_running) {
            if(!_running) break;

            // loop through all session in the schedule

            // check if update is needded
            _update.lock();
            if(pedningUpdate) {
                // update node with new_node
            } _update.unlock();
        }
    }

    std::atomic<bool> _running = false;
    std::thread _runner_thread;

    int gpu_id;
    Node node;
    std::vector<std::pair<std::shared_ptr<ORTRunner>, cudaStream_t>> ort_list;
};

#endif