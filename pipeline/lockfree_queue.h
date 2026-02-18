/**
 * Lock-free SPSC (Single Producer Single Consumer) queue
 * Optimized for low-latency pipeline communication
 */

#ifndef LOCKFREE_QUEUE_H
#define LOCKFREE_QUEUE_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>

template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity)
        : capacity_(capacity)
        , buffer_(std::make_unique<T[]>(capacity))
        , write_pos_(0)
        , read_pos_(0)
    {
        // Ensure capacity is power of 2 for efficient modulo
        if ((capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("Capacity must be power of 2");
        }
    }

    bool push(const T& item) {
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) & (capacity_ - 1);
        
        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer_[current_write] = item;
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        
        if (current_read == write_pos_.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }
        
        item = buffer_[current_read];
        read_pos_.store((current_read + 1) & (capacity_ - 1), std::memory_order_release);
        return true;
    }

    bool empty() const {
        return read_pos_.load(std::memory_order_acquire) == 
               write_pos_.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t write = write_pos_.load(std::memory_order_acquire);
        size_t read = read_pos_.load(std::memory_order_acquire);
        return (write >= read) ? (write - read) : (capacity_ - read + write);
    }

private:
    const size_t capacity_;
    std::unique_ptr<T[]> buffer_;
    alignas(64) std::atomic<size_t> write_pos_; // Cache line aligned
    alignas(64) std::atomic<size_t> read_pos_;  // Cache line aligned
};

#endif // LOCKFREE_QUEUE_H
