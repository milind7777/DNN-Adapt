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
        {"model_repo", required_argument, 0, 0}
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
    
    

    return 0;
}
