/**
 * Performance tracker implementation
 */

#include "performance_tracker.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/resource.h>
#include <mach/mach.h>
#endif

PerformanceTracker* g_performance_tracker = nullptr;

PerformanceTracker::PerformanceTracker() 
    : total_cpu_time_us_(0)
    , total_wall_time_us_(0)
    , cpu_tracking_start_(std::chrono::high_resolution_clock::now())
{
}

PerformanceTracker::~PerformanceTracker() {
    if (this == g_performance_tracker) {
        g_performance_tracker = nullptr;
    }
}

void PerformanceTracker::start_stage(const std::string& stage_name) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    active_timers_[stage_name] = std::chrono::high_resolution_clock::now();
}

void PerformanceTracker::end_stage(const std::string& stage_name) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto it = active_timers_.find(stage_name);
    if (it == active_timers_.end()) {
        return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - it->second);
    
    record_stage(stage_name, duration.count());
    active_timers_.erase(it);
}

void PerformanceTracker::record_stage(const std::string& stage_name, uint64_t duration_us) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    StageMetrics& metrics = metrics_[stage_name];
    metrics.stage_name = stage_name;
    metrics.call_count++;
    metrics.total_time_us += duration_us;
    metrics.current_time_us = duration_us;
    metrics.min_time_us = std::min(metrics.min_time_us, duration_us);
    metrics.max_time_us = std::max(metrics.max_time_us, duration_us);
}

StageMetrics PerformanceTracker::get_metrics(const std::string& stage_name) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto it = metrics_.find(stage_name);
    if (it != metrics_.end()) {
        return it->second;
    }
    return StageMetrics();
}

std::map<std::string, StageMetrics> PerformanceTracker::get_all_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void PerformanceTracker::reset() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.clear();
    active_timers_.clear();
    total_cpu_time_us_ = 0;
    total_wall_time_us_ = 0;
    cpu_tracking_start_ = std::chrono::high_resolution_clock::now();
}

void PerformanceTracker::print_summary() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (metrics_.empty()) {
        std::cout << "No performance metrics recorded." << std::endl;
        return;
    }
    
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Stage"
              << std::right << std::setw(10) << "Calls"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(12) << "Total (ms)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& pair : metrics_) {
        const StageMetrics& m = pair.second;
        std::cout << std::left << std::setw(20) << m.stage_name
                  << std::right << std::setw(10) << m.call_count
                  << std::setw(12) << std::fixed << std::setprecision(2) << m.get_avg_time_ms()
                  << std::setw(12) << std::fixed << std::setprecision(2) << m.get_min_time_ms()
                  << std::setw(12) << std::fixed << std::setprecision(2) << m.get_max_time_ms()
                  << std::setw(12) << std::fixed << std::setprecision(2) << (m.total_time_us / 1000.0)
                  << std::endl;
    }
    
    std::cout << std::endl;
}

double PerformanceTracker::get_cpu_utilization() const {
#ifdef __linux__
    // Linux: Use /proc/self/stat
    FILE* stat_file = fopen("/proc/self/stat", "r");
    if (!stat_file) {
        return 0.0;
    }
    
    unsigned long utime, stime;
    fscanf(stat_file, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
           &utime, &stime);
    fclose(stat_file);
    
    // Convert to microseconds (clock ticks to us)
    long clock_ticks = sysconf(_SC_CLK_TCK);
    uint64_t cpu_time_us = ((utime + stime) * 1000000) / clock_ticks;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto wall_time = std::chrono::duration_cast<std::chrono::microseconds>(
        now - cpu_tracking_start_).count();
    
    if (wall_time > 0) {
        return (cpu_time_us * 100.0) / wall_time;
    }
    return 0.0;
    
#elif defined(__APPLE__)
    // macOS: Use task_info
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO,
                                   (task_info_t)&info, &size);
    if (kerr != KERN_SUCCESS) {
        return 0.0;
    }
    
    uint64_t cpu_time_us = (info.user_time.seconds + info.system_time.seconds) * 1000000 +
                           (info.user_time.microseconds + info.system_time.microseconds);
    
    auto now = std::chrono::high_resolution_clock::now();
    auto wall_time = std::chrono::duration_cast<std::chrono::microseconds>(
        now - cpu_tracking_start_).count();
    
    if (wall_time > 0) {
        return (cpu_time_us * 100.0) / wall_time;
    }
    return 0.0;
    
#else
    // Unsupported platform
    return 0.0;
#endif
}
