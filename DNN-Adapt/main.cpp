//
//  main.cpp
//  DNN-Adapt
//
//  Created by Milind Varma Penumathsa on 4/2/25.
//
#include <getopt.h>
#include <iostream>
#include <filesystem>

bool pathExists(const std::string& path) {
    return std::filesystem::exists(path);
}

int main(int argc, char * argv[]) {
    
    int option_index = 0;
    static struct option long_options[] = {
        {"model_repo", required_argument, 0, 0}
    };
    
    int opt;
    while((opt = getopt_long_only(argc, argv, "m:", long_options, &option_index)) != -1) {
        switch(opt) {
            case 0:
                std::cout << "long option?" << std::endl;
                std::cout << "option " << long_options[option_index].name << std::endl;
                if (optarg)
                    std::cout << " with arg " << optarg << std::endl;
                std::cout << "exe - " << pathExists(optarg) << std::endl;
                break;
            case 'm':
                std::cout << "short option?" << std::endl;
                std::cout << optarg << std::endl;
                break;
            default:
                std::cout << "bad??" << std::endl;
        }
    }
    
    
    return 0;
}
