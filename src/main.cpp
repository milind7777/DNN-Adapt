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
    int option_index = 0;
    static struct option long_options[] = {
        {"model_repo", required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    
    int opt;
    std::string model_dir = "";
    while((opt = getopt_long_only(argc, argv, "m:", long_options, &option_index)) != -1) {
        switch(opt) {
            case 0:
                // std::cout << "option name" << long_options[option_index].name << std::endl;
                model_dir = optarg;
                break;
            case 'm':
                model_dir = optarg;
                break;
            default:
                std::cout << "bad??" << std::endl;
        }
    }

    if(!pathExists(model_dir)) {
        std::cout << "Model repo path provided - " << model_dir << ", does not exist!" << std::endl;
        exit(1);
    }
    
    std::map<std::string, std::string> models;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        // std::cout << entry.path().filename().string() << std::endl;
        auto file_name = entry.path().filename().string();
        if(endsWith(file_name, ".onnx")) {
            auto model_name = file_name.substr(0, file_name.find('.'));
            models[model_name] = entry.path();
        }
    }

    // testing mmap for bin file
    auto mappedBin = mmap_image_bin_file("data/images/batch_input_nchw.bin");
    std::cout << mappedBin.file_size << std::endl;
    
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
    auto s1 = std::make_shared<Session>("vit16", 2000, 20);
    auto s2 = std::make_shared<Session>("resnet18", 2000, 20);
    auto s3 = std::make_shared<Session>("efficientnetb0", 500, 20);
    sessionList.push_back(s1);
    sessionList.push_back(s2);
    sessionList.push_back(s3);

    std::cout << "Running generation\n";
    auto nodeList = test->generate_schedule(sessionList);
    for(int i=0;i<nodeList.size();i++) {
        auto node = nodeList[i];
        std::cout << "NODE NUMBER: " << i+1 << std::endl;
        node->pretty_print();
    }

    std::vector<std::shared_ptr<NodeRunner>> runner_list;
    for(int i=0;i<nodeList.size();i++) {
        auto node_runner = std::make_shared<NodeRunner>(nodeList[i], i);
        runner_list.push_back(node_runner);
    }
    // NodeRunner node_runner = NodeRunner(nodeList[0], 0);

    // // Initialize request processors
    // std::map<std::string, RequestProcessor*> request_processors;
    // for(auto m:models) {
    //     request_processors[m.first] = new RequestProcessor();
    // }

    // // start simulator thread
    // Simulator sim(request_processors);
    // std::thread sim_thread(&Simulator::run, &sim);

    // // manual check for dynamic request generator and get request rate
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
    
    // // Wait for thread to complete
    // sim_thread.join();

    munmap(mappedBin.data_ptr, mappedBin.file_size);
    return 0;
}
