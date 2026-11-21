#include "flags.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

std::optional<Flags> parse_flags(int argc, char** argv, const std::set<std::string>& valued, const std::set<std::string>& switches, std::string& error) {
    Flags flags;
    for (int i = 0; i < argc; ++i) {
        std::string flag = argv[i];
        if (switches.count(flag)) {
            flags.switches.insert(flag);
        } else if (!valued.count(flag)) {
            error = "unknown option " + flag;
            return std::nullopt;
        } else if (i + 1 >= argc) {
            error = flag + " needs a value";
            return std::nullopt;
        } else {
            flags.values[flag] = argv[++i];
        }
    }
    return flags;
}

std::optional<uint64_t> whole_flag(const Flags& flags, const std::string& name, uint64_t fallback, uint64_t low, uint64_t high, std::string& error) {
    auto found = flags.values.find(name);
    if (found == flags.values.end()) {
        return fallback;
    }
    const char* text = found->second.c_str();
    char* end = nullptr;
    errno = 0;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || *text == '-' || errno == ERANGE || value < low || value > high) {
        error = name + " must be a whole number from " + std::to_string(low) + " to " + std::to_string(high);
        return std::nullopt;
    }
    return value;
}

std::optional<std::string> choice_flag(const Flags& flags, const std::string& name, const std::set<std::string>& choices, std::string& error) {
    auto found = flags.values.find(name);
    std::string value = found == flags.values.end() ? *choices.begin() : found->second;
    if (!choices.count(value)) {
        error = name + " must be one of";
        for (const auto& choice : choices) {
            error += " " + choice;
        }
        return std::nullopt;
    }
    return value;
}

int refuse(const std::string& message) {
    std::fprintf(stderr, "delphi: %s\n", message.c_str());
    return 2;
}