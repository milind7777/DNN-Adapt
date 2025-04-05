#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <map>
#include <string>
#include "RequestProcessor.h"

class Simulator {
private:
    const std::map<std::string, RequestProcessor*> &request_processors;

public:
    Simulator(std::map<std::string, RequestProcessor*> &_request_processors): request_processors(_request_processors) {};
    void run();
};

#endif