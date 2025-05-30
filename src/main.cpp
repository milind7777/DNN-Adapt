//
//  main.cpp
//  DNN-Adapt
//
//  Created by Milind Varma Penumathsa on 4/2/25.
//
#include <getopt.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <map>
#include <thread>
#include <sys/mman.h>
#include "Simulator.h"
#include "RequestProcessor.h"
#include "SchedulerInterface.h"
#include "nexus.h"
#include "NodeRunner.h"
#include "imageInput.h"
#include "Logger.h"
#include "Executor.h"
#include "Profiler.h"
#include "grpc_scheduler.h"

const int SLOTS_PER_GPU = 3;
int SIMULATION_LEN = 60; // in seconds

bool pathExists(const std::string &path) {
    return std::filesystem::exists(path);
}

bool endsWith(const std::string &str, const std::string &pattern) {
    if(pattern.length() > str.length()) {
        return false;
    }

    return str.rfind(pattern) == str.length() - pattern.length();
}

void run_profiling() {
    // run profiling on all models for given batch size range
    // go through models folder and find onnx files to run profiling on
    // batch size can range from 1 to 1024 for all models
    // Initilaize profiler with onnx file path and max batch size to test till
    
    // Initialize the logger
    if (!Logger::getInstance().initialize("logs", "profiling", Logger::Level::INFO, Logger::Level::TRACE)) {
        std::cerr << "Failed to initialize logging system. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    // Initialize mmap
    auto mappedBin = mmap_image_bin_file("data/images/batch_input_nchw.bin");

    const std::string models_folder = "models";
    const int max_batch_size = 1024;
    for(const auto& entry:std::filesystem::directory_iterator(models_folder)) {
        if(entry.is_regular_file() && entry.path().extension() == ".onnx") {
            std::string model_path = entry.path().string();
            std::string model_name = entry.path().stem().string();

            // Initialize and run profiler on this model name and path
            auto profiler = new Profiler(model_name, model_path);
            profiler->profile(1024, 0);
        }
    }

    // free mmap and logger
    Logger::getInstance().shutdown();
    munmap(mappedBin.data_ptr, mappedBin.file_size);
}

int main(int argc, char * argv[]) {
    // argument processing
    int option_index = 0;
    static struct option long_options[] = {
        {"model_repo", required_argument, nullptr, 'm'},
        {"run_name", required_argument, nullptr, 'r'},
        {"profile", no_argument, nullptr, 'p'},
        {0, 0, 0, 0}
    };
    
    int opt;
    std::string model_dir = "";
    std::string run_name = "";
    while((opt = getopt_long_only(argc, argv, "m:r:p:", long_options, &option_index)) != -1) {
        switch(opt) {
            case 'm':
                model_dir = optarg;
                break;
            case 'r':
                run_name = optarg;
                break;
            case 'p':
                run_profiling();
                return 0;
            default:
                std::cerr << "Usage: " << argv[0] << " --model_repo <dir> --run_name <name>\n";
                exit(1);
        }
    }

    if (model_dir.empty()) {
        std::cerr << "Error: --model_repo, -m is required\n";
        exit(EXIT_FAILURE);
    }
    
    if (run_name.empty()) {
        std::cerr << "Error: --run_name, -r is required\n";
        exit(EXIT_FAILURE);
    }

    // Initialize the logger
    if (!Logger::getInstance().initialize("logs", run_name, Logger::Level::TRACE, Logger::Level::TRACE)) {
        std::cerr << "Failed to initialize logging system. Exiting.\n";
        exit(EXIT_FAILURE);
    }
    std::cout << "Logging system initialized. Log directory: logs\n";
    
    // Get the main logger
    auto logger = Logger::getInstance().getLogger("main");
    if (!logger) {
        std::cerr << "Failed to get logger. Exiting.\n";
        exit(EXIT_FAILURE);
    }
    
    LOG_DEBUG(logger, "DNN-Adapt starting with run_name: {}, model_repo: {}", run_name, model_dir);
    logger->flush(); // Force flush after important messages

    if(!pathExists(model_dir)) {
        LOG_CRITICAL(logger, "Model repo path provided - {}, does not exist!", model_dir);
        //logger->flush();
        exit(1);
    }
 
    // generate mapping from model names to onnx files
    std::map<std::string, std::string> models;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        auto file_name = entry.path().filename().string();
        if(endsWith(file_name, ".onnx")) {
            auto model_name = file_name.substr(0, file_name.find('.'));
            models[model_name] = entry.path();
        }
    }

    std::vector<std::string> modelNames;
    for(auto model:models) {
        modelNames.push_back(model.first);
    }

    // populate SLO latencies for each model in ms
    std::map<std::string, double> latencies;
    latencies["vit16"] = 1500.0;
    latencies["resnet18"] = 1000.0;
    latencies["efficientnetb0"] = 1000.0;

    // generate mmap for image bin file
    auto mappedBin = mmap_image_bin_file("data/images/batch_input_nchw.bin");
    LOG_DEBUG(logger, "Mapped bin file: {}, size: {}", (void*)mappedBin.data_ptr, mappedBin.file_size);
    //logger->flush();

    
    // Initialize system GPU info
    // TO DO: can be automated?
    std::vector<std::shared_ptr<Gpu>> gpuList;
    auto gpu1 = std::make_shared<Gpu>("A6000", 48);
    auto gpu2 = std::make_shared<Gpu>("A6000", 48);
    gpuList.push_back(gpu1);
    gpuList.push_back(gpu2);

    // path to profiling folder
    std::string profilingFolder = "models/profiles/system";

    // Initialize request processors
    LOG_DEBUG(logger, "Initializing request processors");
    std::map<std::string, std::shared_ptr<RequestProcessor>> request_processors;
    for(auto [model_name, _]:models) {
        request_processors[model_name] = std::make_shared<RequestProcessor>(model_name, latencies[model_name]);
    }

    // Initialize simulator
    auto sim = std::make_shared<Simulator>(request_processors);

    // Initialize executor
    LOG_DEBUG(logger, "Initializing executor");
    auto executor = std::make_shared<Executor>(models, gpuList, request_processors, sim, latencies, profilingFolder);

    // Start grpc server
    RunServer(executor);

    LOG_DEBUG(logger, "Exiting main program");
    logger->flush();
    
    // Short delay to ensure flush completes
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    Logger::getInstance().shutdown();
    munmap(mappedBin.data_ptr, mappedBin.file_size);
    return 0;
}
