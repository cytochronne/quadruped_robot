// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_name_storage.emplace_back(input_name.get());
        }
        rebuild_name_views(input_name_storage, input_names);

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_name_storage.emplace_back(output_name.get());
        }
        rebuild_name_views(output_name_storage, output_names);

        action_output_index = find_name(output_names, "actions");
        if (action_output_index < 0) {
            throw std::runtime_error("ONNX output 'actions' not found.");
        }
        output_shape = output_shapes[action_output_index];

        action.resize(output_shape[1]);
        reset();
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure observation-backed inputs are in obs
        for (const auto& name : input_names) {
            const std::string name_str(name);
            if (state_inputs.find(name_str) != state_inputs.end()) {
                continue;
            }
            if (obs.find(name_str) == obs.end()) {
                throw std::runtime_error("Input name " + name_str + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = state_inputs.count(name_str) ? state_inputs.at(name_str) : obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size()
        );

        for (size_t i = 0; i < output_names.size(); ++i) {
            const std::string name_str(output_names[i]);
            if (name_str == "h_out") {
                update_state_input("h_in", output_tensor[i], output_shapes[i]);
            } else if (name_str == "c_out") {
                update_state_input("c_in", output_tensor[i], output_shapes[i]);
            }
        }

        // Copy output data
        auto floatarr = output_tensor[action_output_index].GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

    void reset() override
    {
        state_inputs.clear();
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name_str(input_names[i]);
            if (name_str == "h_in" || name_str == "c_in") {
                state_inputs[name_str] = std::vector<float>(input_sizes[i], 0.0f);
            }
        }
    }

private:
    static void rebuild_name_views(const std::vector<std::string>& storage, std::vector<const char*>& views)
    {
        views.clear();
        views.reserve(storage.size());
        for (const auto& item : storage) {
            views.push_back(item.c_str());
        }
    }

    static int find_name(const std::vector<const char*>& names, const std::string& target)
    {
        for (size_t i = 0; i < names.size(); ++i) {
            if (target == names[i]) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void update_state_input(const std::string& input_name, Ort::Value& output_tensor, const std::vector<int64_t>& shape)
    {
        int64_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        auto* data = output_tensor.GetTensorMutableData<float>();
        state_inputs[input_name] = std::vector<float>(data, data + size);
    }

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::string> output_name_storage;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
    std::unordered_map<std::string, std::vector<float>> state_inputs;
    int action_output_index = 0;
};
};
