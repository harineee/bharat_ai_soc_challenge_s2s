/**
 * Performance tracking for pipeline stages
 * Measures latency, CPU utilization, and throughput
 */

#ifndef PERFORMANCE_TRACKER_H
#define PERFORMANCE_TRACKER_H

#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <map>

struct StageMetrics {
    std::string stage_name;
    uint64_t call_count;
    uint64_t total_time_us;  // Microseconds
    uint64_t min_time_us;
    uint64_t max_time_us;
    uint64_t current_time_us;
    
    StageMetrics() 
        : call_count(0)
        , total_time_us(0)
        , min_time_us(UINT64_MAX)
        , max_time_us(0)
        , current_time_us(0)
    {}
    
    double get_avg_time_ms() const {
        return call_count > 0 ? (total_time_us / 1000.0) / call_count : 0.0;
    }
    
    double get_min_time_ms() const {
        return min_time_us == UINT64_MAX ? 0.0 : min_time_us / 1000.0;
    }
    
    double get_max_time_ms() const {
        return max_time_us / 1000.0;
    }
};

class PerformanceTracker {
public:
    PerformanceTracker();
    ~PerformanceTracker();
    
    // Start timing a stage
    void start_stage(const std::string& stage_name);
    
    // End timing a stage
    void end_stage(const std::string& stage_name);
    
    // Record a stage duration directly
    void record_stage(const std::string& stage_name, uint64_t duration_us);
    
    // Get metrics for a stage
    StageMetrics get_metrics(const std::string& stage_name) const;
    
    // Get all metrics
    std::map<std::string, StageMetrics> get_all_metrics() const;
    
    // Reset all metrics
    void reset();
    
    // Print summary to stdout
    void print_summary() const;
    
    // Get current CPU utilization (platform-specific)
    double get_cpu_utilization() const;
    
    // RAII timer helper
    class Timer {
    public:
        Timer(PerformanceTracker* tracker, const std::string& stage_name)
            : tracker_(tracker)
            , stage_name_(stage_name)
            , start_time_(std::chrono::high_resolution_clock::now())
        {
            if (tracker_) {
                tracker_->start_stage(stage_name_);
            }
        }
        
        ~Timer() {
            if (tracker_) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time_);
                tracker_->record_stage(stage_name_, duration.count());
            }
        }
        
    private:
        PerformanceTracker* tracker_;
        std::string stage_name_;
        std::chrono::high_resolution_clock::time_point start_time_;
    };

private:
    mutable std::mutex metrics_mutex_;
    std::map<std::string, StageMetrics> metrics_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> active_timers_;
    
    // CPU utilization tracking
    std::atomic<uint64_t> total_cpu_time_us_;
    std::atomic<uint64_t> total_wall_time_us_;
    std::chrono::high_resolution_clock::time_point cpu_tracking_start_;
};

// Global instance (optional, for convenience)
extern PerformanceTracker* g_performance_tracker;

// Convenience macros
#define PERF_START(stage) PerformanceTracker::Timer _perf_timer(g_performance_tracker, stage)
#define PERF_RECORD(stage, duration_us) \
    if (g_performance_tracker) g_performance_tracker->record_stage(stage, duration_us)

#endif // PERFORMANCE_TRACKER_H
