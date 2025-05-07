#include "Logger.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <signal.h>
#include <fstream>
#include <mutex>  
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/basic_file_sink.h"

std::mutex logger_mutex;

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void signal_handler(int signal_num) {
    std::cerr << "Signal " << signal_num << " received, flushing logs...\n";
    spdlog::shutdown(); 
    exit(signal_num);
}

bool Logger::initialize(const std::string& log_dir, const std::string& run_name,
                       Level console_level, Level file_level) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    if (initialized_) {
        return true; 
    }

    try {
        signal(SIGINT, signal_handler);   
        signal(SIGTERM, signal_handler);
        
        // Convert to absolute path
        std::filesystem::path abs_log_dir = std::filesystem::absolute(log_dir);
        log_dir_ = abs_log_dir.string();
        
        // Create log directory if it doesn't exist
        if (!std::filesystem::exists(abs_log_dir)) {
            std::error_code ec;
            std::filesystem::create_directories(abs_log_dir, ec);
            if (ec) {
                std::cerr << "Failed to create log directory: " << ec.message() << std::endl;
                return false;
            }
            std::cout << "Log directory created: " << abs_log_dir.string() << std::endl;
        }

        // Log pattern 
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [pid:%P] [tid:%t] [%^%l%$] %v");

        if (!createLoggers(abs_log_dir.string(), run_name, console_level, file_level)) {
            return false;
        }

        // Global log level
        spdlog::set_level(static_cast<spdlog::level::level_enum>(toSpdlogLevel(Level::TRACE)));

        // Flush logs every second
        spdlog::flush_every(std::chrono::seconds(1));

        initialized_ = true;
        
        auto root_logger = getLogger("root");
        LOG_INFO(root_logger, "Logging system initialized. Log directory: {}", abs_log_dir.string());
        root_logger->flush();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize logger: " << e.what() << std::endl;
        return false;
    }
}

bool Logger::createLoggers(const std::string& log_dir, const std::string& run_name,
                          Level console_level, Level file_level) {
    try {
        default_sinks_ = createSinks(log_dir, run_name, console_level, file_level);
        
        if (default_sinks_.empty()) {
            std::cerr << "No logger sinks created" << std::endl;
            return false;
        }
        
        auto root_logger = std::make_shared<spdlog::logger>("root", default_sinks_.begin(), default_sinks_.end());
        spdlog::register_logger(root_logger);
        
        loggers_["root"] = root_logger;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create loggers: " << e.what() << std::endl;
        return false;
    }
}

std::vector<spdlog::sink_ptr> Logger::createSinks(const std::string& log_dir, 
                                               const std::string& run_name,
                                               Level console_level, 
                                               Level file_level) {
    std::vector<spdlog::sink_ptr> sinks;
    
    try {
        // Console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(toSpdlogLevel(console_level)));
        sinks.push_back(console_sink);
        
        // Create run directory with absolute path
        std::filesystem::path abs_log_dir = std::filesystem::absolute(log_dir);
        std::filesystem::path run_dir = abs_log_dir / run_name;
        
        // Create directory structure
        std::error_code ec;
        if (!std::filesystem::exists(run_dir)) {
            std::filesystem::create_directories(run_dir, ec);
            if (ec) {
                std::cerr << "Failed to create run directory: " << ec.message() << std::endl;
                // Return with just console sink
                return sinks;
            }
        }
        
        std::string log_path = (run_dir / "dnn_adapt.log").string();
        
        try {
            // Using a basic file sink (normal log file)
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path, true);
            file_sink->set_level(static_cast<spdlog::level::level_enum>(toSpdlogLevel(file_level)));
            sinks.push_back(file_sink);
            
            std::cout << "Created log file: " << log_path << std::endl;
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "spdlog exception while creating file sink: " << ex.what() << std::endl;
            // Continue without file sink, at least we'll have console output
        }
        
        return sinks;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create logger sinks: " << e.what() << std::endl;
        return sinks;
    }
}

std::shared_ptr<spdlog::logger> Logger::getLogger(const std::string& component_name) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    auto it = loggers_.find(component_name);
    if (it != loggers_.end()) {
        return it->second;
    }
    
    try {
        auto logger = std::make_shared<spdlog::logger>(component_name, default_sinks_.begin(), default_sinks_.end());
        spdlog::register_logger(logger);
        
        loggers_[component_name] = logger;
        
        return logger;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create logger for component " << component_name << ": " << e.what() << std::endl;
        // Fallback to root 
        return loggers_["root"];
    }
}

int Logger::toSpdlogLevel(Level level) {
    switch (level) {
        case Level::TRACE:
            return spdlog::level::trace;
        case Level::DEBUG:
            return spdlog::level::debug;
        case Level::INFO:
            return spdlog::level::info;
        case Level::WARN:
            return spdlog::level::warn;
        case Level::ERROR:
            return spdlog::level::err;
        case Level::CRITICAL:
            return spdlog::level::critical;
        default:
            return spdlog::level::info;
    }
}

void Logger::shutdown() {
    if (initialized_) {
        try {
            std::lock_guard<std::mutex> lock(logger_mutex);
            
            // Force flush all loggers
            for (auto& logger_pair : loggers_) {
                if (logger_pair.second) {
                    logger_pair.second->flush();
                }
            }
            spdlog::shutdown();
            loggers_.clear();
            default_sinks_.clear();
            initialized_ = false;
        } catch (const std::exception& e) {
            std::cerr << "Error during logger shutdown: " << e.what() << std::endl;
        }
    }
}

Logger::~Logger() {
    shutdown();
}