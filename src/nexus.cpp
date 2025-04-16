#include "nexus.h"
#include "csv.h"
#include <iostream>
#include <filesystem>

extern const std::string LATENCY_COLUMN;
extern const std::string BATCH_COLUMN;
extern const int MAX_BATCH_SIZE;

NexusScheduler::NexusScheduler(const std::vector<std::shared_ptr<Gpu>>& gpus,
    const std::vector<std::string>& models,
    const std::string& profiling_folder): Scheduler(gpus, models, profiling_folder) {
    
    // load profiling info from .csv files
    loadProfile(profiling_folder);
}

void NexusScheduler::loadProfile(const std::string &folderPath) {
    for(const auto &entry: std::filesystem::directory_iterator(folderPath)) {
        if(entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string filePath = entry.path().string();
            std::string modelName = entry.path().stem().string();

            model_profiles[modelName][LATENCY_COLUMN] = loadCSVFile(filePath, LATENCY_COLUMN); 
        }
    }
}

std::vector<double> NexusScheduler::loadCSVFile(const std::string &filePath, const std::string column) {
    std::vector<double> latencies(MAX_BATCH_SIZE+1, 0.0);
    io::CSVReader<2> csvReader(filePath);
    csvReader.read_header(io::ignore_extra_column, BATCH_COLUMN, LATENCY_COLUMN);

    int batchSize;
    int rowCount = 0;
    double latency;

    while(csvReader.read_row(batchSize, latency)) {
        if(batchSize >= 1 && batchSize <= MAX_BATCH_SIZE) {
            latencies[batchSize] = latency;
            rowCount++;
        } else {
            std::cerr << "Warning: batch_size " << batchSize << " is out of bounds" << std::endl;
        }
    } latencies.resize(rowCount+1);

    return latencies;
}

std::vector<std::shared_ptr<Node>> NexusScheduler::generate_schedule(const std::vector<std::shared_ptr<Session>>& sessions) {
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Session>> residual_sessions;

    //TODO: add check to verify latency profile exists for models in sessions

    scheduleSaturate(sessions, nodes, residual_sessions);
    scheduleResidue(nodes, residual_sessions);
    return nodes;
}

void NexusScheduler::scheduleSaturate(const std::vector<std::shared_ptr<Session>> &sessions, std::vector<std::shared_ptr<Node>> &nodes, std::vector<std::shared_ptr<Session>> &residual_sessions) {
    for(auto session:sessions) {
        auto model_name = session->model_name;
        auto latency_SLO = session->latency;
        auto request_rate = session->request_rate;
        const std::vector<double> &latency_profile = model_profiles[model_name][LATENCY_COLUMN];
        
        // Bi = argmax_{b} (2l_{ki} (b) <= Li)
        // TODO: no check for GPU memory?
        auto itr = std::upper_bound(latency_profile.begin(), latency_profile.end(), latency_SLO/2);
        int max_batch = std::distance(latency_profile.begin(), itr) - 1;

        // Ti = Bi/l_{ki} (Bi)
        double max_throughput = (max_batch/latency_profile[max_batch]) * 1000;

        // Ri = nTi + ri
        int n = request_rate/max_throughput;
        double ri = request_rate - (n * max_throughput);
        
        // add n nodes running single model
        session->batch_size = max_batch;
        auto node = std::make_shared<Node>(
            std::vector<std::pair<std::shared_ptr<Session>, std::pair<double, bool>>>{
                {session, {1.0, 0}}
            }, 
            latency_profile[max_batch]
        );
        for(int i=0;i<n;i++) nodes.push_back(node);

        // get residual load
        if(ri > 0) {
            residual_sessions.push_back(std::make_shared<Session>(model_name, latency_SLO, ri));
        }
    }
}

void NexusScheduler::scheduleResidue(std::vector<std::shared_ptr<Node>> &nodes, std::vector<std::shared_ptr<Session>> &residual_sessions) {
    std::vector<std::pair<std::vector<double>, std::shared_ptr<Session>>> session_occ_list;
    for(auto session:residual_sessions) {
        auto model_name = session->model_name;
        auto latency_SLO = session->latency;
        auto request_rate = session->request_rate;
        const std::vector<double> &latency_profile = model_profiles[model_name][LATENCY_COLUMN];

        // TODO: no check for GPU memory?
        // bi = argmax_{b} (l_{ki} (b) + b/ri <= Li)
        // custom binary search
        int l = 1;
        int r = latency_profile.size()-1;
        double value = latency_SLO / 1000;
        while(l<r) {
            int m = (l+r+1)/2;
            double compute = (latency_profile[m]/1000) + (m/request_rate);
            if(compute <= value) {
                l = m;
            } else {
                r = m-1;
            }
        }
        int max_batch = l;

        // di = bi/ri
        auto duty_cycle = (max_batch/request_rate) * 1000;

        // occi = l_{ki} (b) / di
        auto occ_ratio = latency_profile[max_batch] / duty_cycle;
        session_occ_list.push_back({{occ_ratio, duty_cycle}, session});
        session->batch_size = max_batch;
    }

    std::sort(session_occ_list.begin(), session_occ_list.end(), [](const auto &s1, const auto &s2) {
        return s1.first[0] > s2.first[0];
    });

    double max_occ = 0;
    std::shared_ptr<Node> max_node = nullptr;
    std::vector<std::shared_ptr<Node>> residual_nodes;
    for(auto [v, session]:session_occ_list) {
        auto occ_ratio = v[0];
        auto duty_cycle = v[1];
        auto node_ind = 0;

        for(int i=0;i<residual_nodes.size();i++) {
            auto node = residual_nodes[i];
            auto merged_node = mergeNodes(node, session, occ_ratio, duty_cycle);
            if(merged_node != nullptr) {
                if(double merged_occ = merged_node->getOccupancy(); merged_occ > max_occ) {
                    max_occ = merged_occ;
                    max_node = merged_node;
                    node_ind = i;
                }
            }
        }

        if(max_node != nullptr) {
            residual_nodes[node_ind] = max_node;
        } else {
            residual_nodes.push_back(std::make_shared<Node>(
                std::vector<std::pair<std::shared_ptr<Session>, std::pair<double, bool>>>{
                    {session, {occ_ratio, 0}}
                }, 
                duty_cycle
            ));
        }
    }

    nodes.reserve(nodes.size() + residual_nodes.size());
    nodes.insert(nodes.end(), residual_nodes.begin(), residual_nodes.end());
}

std::shared_ptr<Node> NexusScheduler::mergeNodes(std::shared_ptr<Node> node, std::shared_ptr<Session> session, double occ_ratio, double duty_cycle) {
    auto node_duty_cycle = node->duty_cycle;
    if(node_duty_cycle < duty_cycle) {
        const std::vector<double> &latency_profile = model_profiles[session->model_name][LATENCY_COLUMN];
        // reduce the batch size of session and try to fit in node
        int new_batch_size = (node_duty_cycle/1000) * session->request_rate;

        double session_occ = latency_profile[new_batch_size] / node_duty_cycle;
        double total_occ = node->getOccupancy() + session_occ;
        if(total_occ <= 1) {
            // add session to node
            std::shared_ptr<Node> new_node = std::make_shared<Node>(*node);
            std::shared_ptr<Session> new_session = std::make_shared<Session>(*session);
            new_session->batch_size = new_batch_size;
            new_node->session_list.push_back({new_session, {session_occ, 0}});
            return new_node;
        } else {
            // merge not possible
            return nullptr;
        }
    } else {
        // reduce batch size of all sessions in node
        // TODO: can we refactor to reduce copies?
        std::shared_ptr<Node> new_node = std::make_shared<Node>();
        new_node->duty_cycle = duty_cycle;
        for(auto [s, attr]:node->session_list) {
            const std::vector<double> &latency_profile = model_profiles[s->model_name][LATENCY_COLUMN];
            
            int new_batch_size = (duty_cycle/1000) * s->request_rate;
            double new_session_occ = latency_profile[new_batch_size] / duty_cycle;
            std::shared_ptr<Session> new_session = std::make_shared<Session>(*s);
            new_session->batch_size = new_batch_size;

            new_node->session_list.push_back({new_session, {new_session_occ, 0}});
        }

        // merge the session with new node
        std::shared_ptr<Session> new_session = std::make_shared<Session>(*session);
        new_node->session_list.push_back({new_session, {occ_ratio, 0}});

        if(new_node->getOccupancy() <= 1) {
            return new_node;
        } else {
            return nullptr;
        }
    }

    return nullptr;
}