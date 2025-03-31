//
// Created by saleh on 3/30/25.
//

#include "timer_scope.h"

timer_scope::timer_scope(const std::string& name) : name(name), m_bIsRoot(true) {
    m_oTimerLast = high_resolution_clock::now();
    m_pStats = nullptr;
}

timer_scope::timer_scope(timer_stats& parent) : name(""), m_bIsRoot(false) {
    m_oTimerLast = high_resolution_clock::now();
    m_pStats = &parent;
}

timer_scope::~timer_scope() {
    if (report) {
        if (m_bIsRoot) {
            report_from_last(name);
        }
        else {
            m_pStats->add_sample(from_last());
        }
    }
}
