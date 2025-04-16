#ifndef NEXUS_H
#define NEXUS_H

#include "SchedulerInterface.h"
#include "iostream"
#include <map>

extern const std::string LATENCY_COLUMN;

class NexusScheduler: public Scheduler {
    public:
        NexusScheduler(const std::vector<std::shared_ptr<Gpu>>& gpus,
            const std::vector<std::string>& models,
            const std::string& profiling_folder);

        std::vector<std::shared_ptr<Node>> generate_schedule(const std::vector<std::shared_ptr<Session>>& sessions);

    protected:
        std::map<std::string, std::map<std::string, std::vector<double>>> model_profiles;

    private:
        void loadProfile(const std::string &folderPath);
        std::vector<double> loadCSVFile(const std::string &filePath, const std::string column);
        void scheduleSaturate(const std::vector<std::shared_ptr<Session>> &sessions, std::vector<std::shared_ptr<Node>> &nodes, std::vector<std::shared_ptr<Session>> &residual_sessions);
        void scheduleResidue(std::vector<std::shared_ptr<Node>> &nodes, std::vector<std::shared_ptr<Session>> &residual_sessions);
        std::shared_ptr<Node> mergeNodes(std::shared_ptr<Node> node, std::shared_ptr<Session> session, double occ_ratio, double duty_cycle);
};



#endif