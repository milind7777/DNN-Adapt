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
#include  "Logger.h"

bool pathExists(const std::string &path) {
    return std::filesystem::exists(path);
}

bool endsWith(const std::string &str, const std::string &pattern) {
    if(pattern.length() > str.length()) {
        return false;
    }

    return str.rfind(pattern) == str.length() - pattern.length();
}

int main(int argc, char * argv[]) {
    // int gpu_id = 0;
    // size_t total_bytes = 64 * 1024 * 1024;

    // cudaError_t err = cudaSetDevice(gpu_id);
    // if (err != cudaSuccess) {
    //     std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << "\n";
    //     return 1;
    // }

    // float* gpu_ptr = nullptr;
    // err = cudaMalloc((void**)&gpu_ptr, total_bytes);
    // if (err != cudaSuccess) {
    //     std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
    //     return 1;
    // }

    // std::cout << "cudaMalloc succeeded.\n";
    // cudaFree(gpu_ptr);
    // return 0;

    
    int option_index = 0;
    static struct option long_options[] = {
        {"model_repo", required_argument, nullptr, 'm'},
        {"run_name", required_argument, nullptr, 'r'},
        {0, 0, 0, 0}
    };
    
    int opt;
    std::string model_dir = "";
    std::string run_name = "";
    while((opt = getopt_long_only(argc, argv, "m:r:", long_options, &option_index)) != -1) {
        switch(opt) {
            case 'm':
                model_dir = optarg;
                break;
            case 'r':
                run_name = optarg;
                break;
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
    if (!Logger::getInstance().initialize("logs", run_name, Logger::Level::INFO, Logger::Level::TRACE)) {
        std::cerr << "Failed to initialize logging system. Exiting.\n";
        exit(EXIT_FAILURE);
    }
    
    // Get the main logger
    auto logger = Logger::getInstance().getLogger("main");
    if (!logger) {
        std::cerr << "Failed to get logger. Exiting.\n";
        exit(EXIT_FAILURE);
    }
    
    LOG_INFO(logger, "DNN-Adapt starting with run_name: {}, model_repo: {}", run_name, model_dir);
    //logger->flush(); // Force flush after important messages

    if(!pathExists(model_dir)) {
        LOG_CRITICAL(logger, "Model repo path provided - {}, does not exist!", model_dir);
        //logger->flush();
        exit(1);
    }
 
    std::map<std::string, std::string> models;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        auto file_name = entry.path().filename().string();
        if(endsWith(file_name, ".onnx")) {
            auto model_name = file_name.substr(0, file_name.find('.'));
            models[model_name] = entry.path();
        }
    }

    // testing mmap for bin file
    auto mappedBin = mmap_image_bin_file("data/images/batch_input_nchw.bin");
    LOG_INFO(logger, "Mapped bin file: {}, size: {}", (void*)mappedBin.data_ptr, mappedBin.file_size);
    //logger->flush();

    
    std::vector<std::shared_ptr<Gpu>> gpuList;
    auto gpu1 = std::make_shared<Gpu>("A6000", 48);
    auto gpu2 = std::make_shared<Gpu>("A6000", 48);
    gpuList.push_back(gpu1);
    gpuList.push_back(gpu2);

    std::vector<std::string> modelNames;
    for(auto model:models) {
        modelNames.push_back(model.first);
    }

    std::string profilingFolder = "models/profiles/sample";

    NexusScheduler* test = new NexusScheduler(gpuList, modelNames, profilingFolder);
    
    std::vector<std::shared_ptr<Session>> sessionList;
    auto s1 = std::make_shared<Session>("vit16", 2000, 200);
    // auto s2 = std::make_shared<Session>("resnet18", 2000, 20);
    // auto s3 = std::make_shared<Session>("efficientnetb0", 500, 20);
    sessionList.push_back(s1);
    // sessionList.push_back(s2);
    // sessionList.push_back(s3);

    LOG_INFO(logger, "Running generation");

    auto nodeList = test->generate_schedule(sessionList);
    for(int i=0;i<nodeList.size();i++) {
        auto node = nodeList[i];
        //LOG_INFO(logger, "Node {}: GPU: {}, Duty Cycle: {}", i+1, node->gpu.gpu_type, node->duty_cycle);
        //LOG_INFO(logger, "Session List: ");
        LOG_INFO(logger, "NODE NUMBER: {}", i+1);
        node->pretty_print();
    }

    // Initialize request processors
    std::map<std::string, std::shared_ptr<RequestProcessor>> request_processors;
    for(auto [model_name, _]:models) {
        request_processors[model_name] = std::make_shared<RequestProcessor>(model_name);
    }

    std::vector<std::shared_ptr<NodeRunner>> runner_list;
    for(int i=0;i<nodeList.size();i++) {
        auto node_runner = std::make_shared<NodeRunner>(nodeList[i], i, request_processors);
        runner_list.push_back(node_runner);
    }

    for(int i=0;i<runner_list.size();i++) {
        auto runner = runner_list[i];
        runner->start();
    }

    // for(int i=0;i<runner_list.size();i++) {
    //     auto runner = runner_list[i];
    //     if(runner->_runner_thread.joinable()) {
    //         runner->_runner_thread.join();
    //     }
    // }

    // start simulator thread
    Simulator sim(request_processors);
    std::thread sim_thread(&Simulator::run, &sim);

    // manual check for dynamic request generator and get request rate
    // int total_req_count[3] = {0};
    // int new_req_count[3]   = {0};
    // int request_rate[3]    = {0};
    // for(int i=0;i<30;i++) {
    //     int j = 0;
    //     for(const auto &[model_name, processor]: request_processors) {
    //         total_req_count[j] = processor->get_size();
    //         processor->form_batch(total_req_count[j]);
    //         request_rate[j] = processor->get_request_rate();
    //         j++;
    //     }

    //     for(int k=0;k<3;k++) {
    //         std::cout << "<" << request_rate[k] << "> ";
    //     } std::cout << std::endl;

    //     std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // }
    
    // Wait for thread to complete
    sim_thread.join();

    LOG_INFO(logger, "Exiting main program");
    logger->flush();
    
    // Short delay to ensure flush completes
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    Logger::getInstance().shutdown();
    munmap(mappedBin.data_ptr, mappedBin.file_size);
    return 0;
}
