#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <cstring>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];
        auto keyboard_transitions = param::config["FSM"][state_string]["keyboard_transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        if (keyboard_transitions)
        {
            auto keyboard_transition_map = keyboard_transitions.as<std::map<std::string, std::string>>();
            for (const auto& it : keyboard_transition_map)
            {
                const std::string target_fsm = it.first;
                if (!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }
                const int fsm_id = FSMStringMap.right.at(target_fsm);
                const std::string condition = it.second;

                std::string key = condition;
                bool require_on_pressed = false;
                constexpr const char* kPressedSuffix = ".on_pressed";
                if (key.size() > std::strlen(kPressedSuffix) &&
                    key.rfind(kPressedSuffix) == key.size() - std::strlen(kPressedSuffix))
                {
                    key = key.substr(0, key.size() - std::strlen(kPressedSuffix));
                    require_on_pressed = true;
                }

                registered_checks.emplace_back(
                    std::make_pair(
                        [key, require_on_pressed]()->bool
                        {
                            if (!FSMState::keyboard) return false;
                            const bool key_match = FSMState::keyboard->key() == key;
                            return require_on_pressed ? (key_match && FSMState::keyboard->on_pressed) : key_match;
                        },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};
