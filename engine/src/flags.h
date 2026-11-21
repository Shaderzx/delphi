#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>

struct Flags {
    std::map<std::string, std::string> values;
    std::set<std::string> switches;
};

std::optional<Flags> parse_flags(int argc, char** argv, const std::set<std::string>& valued, const std::set<std::string>& switches, std::string& error);
std::optional<uint64_t> whole_flag(const Flags& flags, const std::string& name, uint64_t fallback, uint64_t low, uint64_t high, std::string& error);
std::optional<std::string> choice_flag(const Flags& flags, const std::string& name, const std::set<std::string>& choices, std::string& error);
int refuse(const std::string& message);