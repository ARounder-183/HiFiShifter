#pragma once

#include "UstImporter.h"
#include <string_view>

namespace hachi::backend
{
// Format boundary only. No project mutation, renderer inference, singer lookup,
// external audio loading or source writes. Native tick/part structure is retained.
class UstxImporter final
{
public:
    // Strict UTF-8 YAML (JSON-valid YAML is accepted by the same parser).
    // One document, <=16 MiB, <=1,000,000 YAML events, depth <=64;
    // aliases, duplicate/non-scalar map keys and custom tags are rejected.
    // |Score positions/endpoints| <=1,000,000,000 ticks; <=256 tracks,
    // <=10,000 combined voice/wave parts and <=100,000 notes; BPM 1..1000.
    // Generic numeric metadata must be finite and within +/-1e12; warnings
    // are capped at 1,000 plus a truncation notice. Failure returns no partial score.
    [[nodiscard]] static std::optional<UstProject> parse(
        std::string_view utf8, juce::String& error, juce::StringArray& warnings);
    [[nodiscard]] static std::optional<UstProject> read(
        const juce::File& file, juce::String& error, juce::StringArray& warnings);
};
}
