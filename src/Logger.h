#ifndef LOGGER_H
#define LOGGER_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>  
#include "spdlog/spdlog.h"
#include "spdlog/sinks/sink.h"

// A wrapper around spdlog
class Logger {
public:
    
    enum class Level {
        TRACE,
        DEBUG,
        INFO,
        WARN,
        ERROR,
        CRITICAL,
    };
    
    static Logger& getInstance();
    
    // Initialize the logger
    // log_dir Directory, run_name Name of the current run (used for log file naming)
    // console_level Minimum log level, file_level Minimum log level for file output
    // Returns true if initialization was successful, false otherwise
    bool initialize(const std::string& log_dir, const std::string& run_name,
                  Level console_level = Level::TRACE, Level file_level = Level::TRACE);
    
    // Get a logger instance for a specific component
    // component_name Name of the component
    // Returns Logger for the component
    std::shared_ptr<spdlog::logger> getLogger(const std::string& component_name);
    
    static int toSpdlogLevel(Level level);
    
    // Shutdown all loggers and flush all logs
    void shutdown();

    // Add a helper method to flush all loggers
    void flushAll() {
        for (auto& logger_pair : loggers_) {
            if (logger_pair.second) {
                logger_pair.second->flush();
            }
        }
    }

private:
    Logger() = default;
    ~Logger();
    
    // Disable copy and move operations
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    // Create loggers
    bool createLoggers(const std::string& log_dir, const std::string& run_name,
                      Level console_level, Level file_level);
    
    std::vector<spdlog::sink_ptr> createSinks(const std::string& log_dir,
                                           const std::string& run_name,
                                           Level console_level,
                                           Level file_level);
    
    bool initialized_ = false;
    
    std::string log_dir_;
    
    std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> loggers_;
    
    std::vector<spdlog::sink_ptr> default_sinks_;

    std::mutex init_mutex_;
};

// macros 
#define LOG_TRACE(logger, ...) logger->trace(__VA_ARGS__)
#define LOG_DEBUG(logger, ...) logger->debug(__VA_ARGS__)
#define LOG_INFO(logger, ...) logger->info(__VA_ARGS__)
#define LOG_WARN(logger, ...) logger->warn(__VA_ARGS__)
#define LOG_ERROR(logger, ...) do { logger->error(__VA_ARGS__); logger->flush(); } while(0)
#define LOG_CRITICAL(logger, ...) do { logger->critical(__VA_ARGS__); logger->flush(); } while(0)

#endif 