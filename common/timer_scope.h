//
// Created by saleh on 3/30/25.
//

#pragma once

#include <iostream>
#include <chrono>
#include <functional>

#include "timer_stats.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class timer_scope {
protected:
    std::chrono::system_clock::time_point m_oTimerLast;
    const std::string name;
    const bool m_bIsRoot;
    timer_stats* m_pStats; // to keep things simple, we are not using smart pointers.
    bool report = true;

public:
    timer_scope(const std::string& name);

    timer_scope(timer_stats& parent);

    ~timer_scope();

    template <class StdTimeResolution = std::milli>
    float from_last();

    template <class StdTimeResolution = std::milli>
    float report_from_last(const std::string& msg = "");

    template <class StdTimeResolution = std::milli>
    static inline float for_lambda(const std::function<void()>& operation);

    template <class StdTimeResolution = std::milli>
    static inline float report_for_lambda(const std::function<void()>& operation);
};

template <class StdTimeResolution>
float timer_scope::from_last() {
    auto now = high_resolution_clock::now();
    duration<float, StdTimeResolution> ms = now - m_oTimerLast;
    m_oTimerLast = now;
    return ms.count();
}

template <class StdTimeResolution>
float timer_scope::report_from_last(const std::string& msg) {
    auto t = from_last<StdTimeResolution>();
    std::cout << "Elapsed " << msg << ": " << t << " ." << std::endl;
    return t;
}

template <class StdTimeResolution>
float timer_scope::for_lambda(const std::function<void()>& operation) {
    auto t1 = high_resolution_clock::now();
    operation();
    auto t2 = high_resolution_clock::now();
    duration<float, StdTimeResolution> ms = t2 - t1;
    return ms.count();
}

template <class StdTimeResolution>
float timer_scope::report_for_lambda(const std::function<void()>& operation) {
    auto t = for_lambda<StdTimeResolution>(operation);
    std::cout << "Elapsed: " << t << " ." << std::endl;
    return t;
}
