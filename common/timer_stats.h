//
// Created by saleh on 3/30/25.
//

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

class timer_scope; // Forward declaration
class timer_stats {
    friend class timer_scope;

private:
    const std::string name;
    std::vector<float> samples;
    const std::map<std::string, int> pairs;
    const bool dont_report;

    std::string legalize_filename(const std::string& name) const;
    std::string pairs_to_string() const;
    std::string pairs_to_json() const;
    std::string data_to_json() const;

public:
    timer_stats(const std::string& name, bool dont_report = false);
    timer_stats(const std::string& name, const std::map<std::string, int>& pairs, bool dont_report = false);
    void add_sample(float time);
    size_t count() const;
    float ave() const;
    float max() const;
    float min() const;
    float median() const;
    float variance() const;
    void print() const;
    void save() const;
    ~timer_stats();
};
