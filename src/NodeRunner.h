#ifndef NODERUNNER_H
#define NODERUNNER_H

#include "SchedulerInterface.h"
#include <iostream>
#include <thread>

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
        update.lock();
        new_node = updated_node;
        isUpdate = true;
        update.unlock();
    }

private:
    std::mutex update;
    bool isUpdate;
    Node new_node;
};

class ORTRunner {
    // This class runs the onnx runtime, each instance will have a ort session, path to onnx file, cuda stream associated with it
    int x;
};

#endif