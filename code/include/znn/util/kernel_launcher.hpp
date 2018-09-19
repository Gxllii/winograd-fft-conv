//
// Copyright (C) 2018 Aleksandar Zlateski <zlateski@mit.edu>
// Copyright (C) 2018 Zhen Jia <zhenj@princeton.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include "znn/intrin.hpp"
#include "znn/types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <thread>

#if !defined(ZNN_NUM_CORES)
#define ZNN_NUM_CORES 64
#endif

#if !defined(ZNN_USE_CORES)
#define ZNN_USE_CORES ZNN_NUM_CORES
#endif

namespace znn
{

class alignas(64) znn_barrier
{
private:
    alignas(64) int const P;             // barrier threshold.
    alignas(64) std::atomic<int> bar{0}; // counter of threads, faced barrier.
    alignas(64) std::atomic<int> passed{
        0}; // number of barriers, passed by all threads.

public:
    znn_barrier(int p)
        : P(p)
    {
    }

    void wait()
    {
        // memory_order_relaxed is OK b/c of the acquire fence at the bottom of
        // the function
        int passed_old = passed.load(std::memory_order_relaxed);

        // fetch_add is sequentially consistent
        // all threads see bar increase in the same order
        if (bar.fetch_add(1) == (P - 1))
        {
            // the last thread, faced barrier.
            bar = 0;
            // synchronize and store in one operation. memory_order_release
            // ensures bar = 0 is not reordered with passed_old store
            passed.store(passed_old + 1, std::memory_order_release);
        }
        else
        {
            // not the last thread. wait others. some thread will increase
            // passed_old by one.
            while (passed.load(std::memory_order_relaxed) == passed_old)
            {
                _mm_pause();
            };
            // need to synchronize cache with other threads, passed barrier.
            // all memory operations stay below the memory barrier.
            std::atomic_thread_fence(std::memory_order_acquire);
        }
    }
};

class alignas(64) kernel_launcher
{
private:
    alignas(64) znn_barrier even_barrier;
    alignas(64) pthread_barrier_t odd_barrier;
    alignas(64) long_t num_threads_;
    alignas(64) cpu_set_t old_set_;
    alignas(64) std::function<void()>* kernels;

private:
    void even_thread_loop(long_t id, long_t core)
    {
        cpu_set_t old_set;
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t set;
        CPU_ZERO(&set);

        CPU_SET(static_cast<int>(core), &set);
        sched_setaffinity(0, sizeof(set), &set);

        even_barrier.wait();
        // Constructor done

        while (1)
        {
            even_barrier.wait();

            if (kernels == nullptr)
            {
                sched_setaffinity(0, sizeof(old_set), &old_set);
                even_barrier.wait();
                return;
            }
            else if (kernels[id])
            {
                kernels[id]();
            }

            even_barrier.wait();
        }
    }

    void odd_thread_loop(long_t id, long_t core)
    {
        cpu_set_t old_set;
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t set;
        CPU_ZERO(&set);

        CPU_SET(static_cast<int>(core), &set);
        sched_setaffinity(0, sizeof(set), &set);

        pthread_barrier_wait(&odd_barrier);
        // Constructor done

        while (1)
        {
            pthread_barrier_wait(&odd_barrier);

            if (kernels == nullptr)
            {
                sched_setaffinity(0, sizeof(old_set), &old_set);
                pthread_barrier_wait(&odd_barrier);
                return;
            }
            else if (kernels[id])
            {
                kernels[id]();
            }
            pthread_barrier_wait(&odd_barrier);
        }
    }

public:
    kernel_launcher(long_t n_cpus, long_t n_hwt, long_t cpu_scale = 1)
        : even_barrier(n_cpus)
        , num_threads_(n_cpus * n_hwt)
        , kernels(nullptr)
    {
        long_t cpu_offset = ZNN_NUM_CORES - n_cpus * cpu_scale;

        sched_getaffinity(0, sizeof(old_set_), &old_set_);
        pthread_barrier_init(&odd_barrier, NULL, static_cast<int>(n_cpus + 1));

        cpu_set_t set;
        CPU_ZERO(&set);

        CPU_SET(static_cast<int>(cpu_offset), &set);
        sched_setaffinity(0, sizeof(set), &set);

        for (long_t c = 0; c < n_cpus; ++c)
        {
            for (long_t h = 0; h < n_hwt; ++h)
            {
                if (c + h > 0)
                {
                    long_t id = c * n_hwt + h;
                    long_t core =
                        cpu_offset + c * cpu_scale + h * ZNN_NUM_CORES;

                    if (h)
                    {
                        std::thread t(&kernel_launcher::odd_thread_loop, this,
                                      id, core);
                        t.detach();
                    }
                    else
                    {
                        std::thread t(&kernel_launcher::even_thread_loop, this,
                                      id, core);
                        t.detach();
                    }
                }
            }
        }

        even_barrier.wait();
        pthread_barrier_wait(&odd_barrier);
    }

    void launch(std::function<void()>* ks)
    {
        kernels = ks;

        even_barrier.wait();

        if (kernels == nullptr)
        {
            sched_setaffinity(0, sizeof(old_set_), &old_set_);
        }
        else if (kernels[0])
        {
            kernels[0]();
        }

        even_barrier.wait();
    }

    template <bool HT>
    void launch2(std::function<void()>* ks)
    {
        kernels = ks;

        if constexpr (HT)
        {
            pthread_barrier_wait(&odd_barrier);
        }

        even_barrier.wait();

        if (kernels == nullptr)
        {
            sched_setaffinity(0, sizeof(old_set_), &old_set_);
        }
        else if (kernels[0])
        {
            kernels[0]();
        }

        if constexpr (HT)
        {
            pthread_barrier_wait(&odd_barrier);
        }

        even_barrier.wait();
    }

    long_t num_threads() const { return num_threads_; }

    ~kernel_launcher()
    {
        launch2<true>(nullptr);
        pthread_barrier_destroy(&odd_barrier);
    }
};

} // namespace znn
