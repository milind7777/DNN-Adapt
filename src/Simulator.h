#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <map>
#include <string>
#include "RequestProcessor.h"

class Simulator {
private:
    std::map<std::string, std::shared_ptr<RequestProcessor>> request_processors;

public:
    Simulator(std::map<std::string, std::shared_ptr<RequestProcessor>> _request_processors): request_processors(_request_processors) {};
    void run();
};

#endif