// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <deque>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <string>

namespace isaaclab
{

class ManagerBasedRLEnv;

using ObsFunc = std::function<std::vector<float>(ManagerBasedRLEnv*, YAML::Node)>;

struct ObservationTermCfg
{
    YAML::Node params;
    ObsFunc func;
    std::vector<float> clip;
    std::vector<float> scale;
    std::string noise_type;
    float noise_n_min = 0.0f;
    float noise_n_max = 0.0f;
    bool enable_noise = false;
    int history_length = 1;
    bool scale_first = false;

    void reset(std::vector<float> obs)
    {
        for(int i(0); i < history_length; ++i) add(obs);
    }

    void add(std::vector<float> obs)
    {
        if(enable_noise && noise_type == "uniform")
        {
            const float n_min = std::min(noise_n_min, noise_n_max);
            const float n_max = std::max(noise_n_min, noise_n_max);
            static thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(n_min, n_max);
            for(float& v : obs)
            {
                v += dist(rng);
            }
        }

        for(int j = 0; j < obs.size(); ++j)
        {
            if(scale_first) {
                if(!scale.empty()) obs[j] *= scale[j];
                if (!clip.empty()) {
                    obs[j] = std::clamp(obs[j], clip[0], clip[1]);
                }
            } else {
                if (!clip.empty()) {
                    obs[j] = std::clamp(obs[j], clip[0], clip[1]);
                }
                if(!scale.empty()) obs[j] *= scale[j];
            }
        }
        buff_.push_back(obs);

        if (buff_.size() > history_length) buff_.pop_front();
    }

    const std::vector<float> & get(int n) const { return buff_[n]; }

    const std::vector<float> get() const
    {
        std::vector<float> concatenated;
        for (const auto& entry : buff_) {
            concatenated.insert(concatenated.end(), entry.begin(), entry.end());
        }
        return concatenated;
    }

    const std::size_t size() const { return std::accumulate(buff_.begin(), buff_.end(), 0,
        [](std::size_t sum, const auto& v) { return sum + v.size(); }); }

private:
    // Complete circular buffer with most recent entry at the end and oldest entry at the beginning.
    std::deque<std::vector<float>> buff_;
};

};
