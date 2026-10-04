//------------------------------------------------------------------------------
//! \file       AraProbeDump.cpp
//!             ARA 模型图 dump 的实现（HiFiShifter 探针专用，一次性产物）
//! \project    HiFiShifter ARA2 probe
//!
//! 主要内容：遍历 ARA 模型图并序列化为 JSON。
//! 作用：给出"宿主通过 ARA 到底提供了哪些字段"的一手证据。
//! 特殊说明：见头注释与各函数 doc。整体重写文件（快照语义）。
//------------------------------------------------------------------------------

#include "AraProbeDump.h"

#include "ARA_Library/PlugIn/ARAPlug.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

namespace
{
    /// JSON 字符串转义。ARA 的 name 来自宿主，可能含引号/反斜杠/非 ASCII。
    std::string jsonEscape (const char* text)
    {
        if (text == nullptr)
            return "null";

        std::string out { "\"" };
        for (const char* p = text; *p != '\0'; ++p)
        {
            const unsigned char c { static_cast<unsigned char> (*p) };
            switch (c)
            {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20)
                    {
                        char buf[8];
                        std::snprintf (buf, sizeof (buf), "\\u%04x", c);
                        out += buf;
                    }
                    else
                    {
                        // 原样透传 UTF-8 字节
                        out += static_cast<char> (c);
                    }
                    break;
            }
        }
        out += "\"";
        return out;
    }

    /// 可选字符串属性。ARA 的 OptionalProperty<ARAUtf8String> 可隐式转为指针；
    /// 这里直接用 const char* 承接，避免依赖该 typedef 的可见性。
    template <typename OptionalString>
    std::string jsonOptionalString (const OptionalString& value)
    {
        const char* const raw { value };
        return jsonEscape (raw);
    }

    /// 把相对路径解析成绝对路径（不依赖 std::filesystem —— 本目标按 C++11 编译）。
    std::string absolutePath (const std::string& relative)
    {
        char buffer[MAX_PATH] { };
        const DWORD written { ::GetFullPathNameA (relative.c_str (), MAX_PATH, buffer, nullptr) };
        if (written == 0 || written >= MAX_PATH)
            return relative;
        return std::string { buffer };
    }

    /// 真实存在的输出文件路径。返回空串表示无处可写。
    std::string resolveOutputPath ()
    {
        if (const char* fromEnv = std::getenv ("ARA_PROBE_OUT"))
            if (*fromEnv != '\0')
                return std::string { fromEnv };

        // 未设环境变量时的候选：当前工作目录、以及探针 captures 目录的相对位置。
        const std::vector<std::string> candidates {
            ".\\ara-model.json",
            "..\\..\\..\\..\\..\\captures\\ara-model.json",
            "captures\\ara-model.json",
        };

        for (const auto& candidate : candidates)
        {
            const std::string absolute { absolutePath (candidate) };
            // 只要目录存在就采用；写失败会在调用方静默降级。
            std::string directory { absolute };
            const std::size_t slash { directory.find_last_of ("\\/") };
            if (slash != std::string::npos)
                directory.erase (slash);
            if (directory.empty () || (::GetFileAttributesA (directory.c_str ()) != INVALID_FILE_ATTRIBUTES))
                return absolute;
        }
        return {};
    }

    /// 节流：文档变更回调可能被高频触发，限制写盘频率。
    bool shouldWriteNow ()
    {
        static auto lastWrite { std::chrono::steady_clock::time_point::min () };
        const auto now { std::chrono::steady_clock::now () };
        if (lastWrite != std::chrono::steady_clock::time_point::min ()
            && now - lastWrite < std::chrono::milliseconds { 400 })
            return false;
        lastWrite = now;
        return true;
    }
}

/*******************************************************************************/

void AraProbeDumpToFile (ARA::PlugIn::DocumentController* documentController) noexcept
{
    if (documentController == nullptr || !shouldWriteNow ())
        return;

    try
    {
        auto* const document { documentController->getDocument () };
        if (document == nullptr)
            return;

        std::ostringstream out;
        out << "{\n";
        out << "  \"_probe\": \"HiFiShifter ARA probe dump\",\n";
        out << "  \"documentName\": " << jsonOptionalString (document->getName ()) << ",\n";

        // ---- audioSources ----
        out << "  \"audioSources\": [\n";
        {
            const auto& sources { document->getAudioSources () };
            for (std::size_t i = 0; i < sources.size (); ++i)
            {
                auto* const source { sources[i] };
                out << "    {";
                out << "\"persistentID\": " << jsonEscape (source->getPersistentID ().c_str ());
                out << ", \"name\": " << jsonOptionalString (source->getName ());
                out << ", \"sampleRate\": " << source->getSampleRate ();
                out << ", \"sampleCount\": " << source->getSampleCount ();
                out << ", \"durationSeconds\": " << source->getDuration ();
                out << ", \"channelCount\": " << source->getChannelCount ();
                out << ", \"merits64BitSamples\": " << (source->merits64BitSamples () ? "true" : "false");
                out << ", \"sampleAccessEnabled\": " << (source->isSampleAccessEnabled () ? "true" : "false");
                out << ", \"deactivatedForUndoHistory\": " << (source->isDeactivatedForUndoHistory () ? "true" : "false");
                out << ", \"modificationCount\": " << source->getAudioModifications ().size ();
                out << "}";
                if (i + 1 < sources.size ())
                    out << ",";
                out << "\n";
            }
        }
        out << "  ],\n";

        // ---- musicalContexts + regionSequences ----
        out << "  \"musicalContexts\": [\n";
        {
            const auto& contexts { document->getMusicalContexts () };
            for (std::size_t i = 0; i < contexts.size (); ++i)
            {
                auto* const context { contexts[i] };
                out << "    {";
                out << "\"name\": " << jsonOptionalString (context->getName ());
                out << ", \"orderIndex\": " << context->getOrderIndex ();
                out << ", \"regionSequences\": [";
                const auto& sequences { context->getRegionSequences () };
                for (std::size_t s = 0; s < sequences.size (); ++s)
                {
                    out << "{";
                    out << "\"name\": " << jsonOptionalString (sequences[s]->getName ());
                    out << ", \"orderIndex\": " << sequences[s]->getOrderIndex ();
                    out << ", \"playbackRegionCount\": " << sequences[s]->getPlaybackRegions ().size ();
                    out << "}";
                    if (s + 1 < sequences.size ())
                        out << ", ";
                }
                out << "]}";
                if (i + 1 < contexts.size ())
                    out << ",";
                out << "\n";
            }
        }
        out << "  ],\n";

        // ---- audioModifications ----
        // 注意：modification 挂在 source 下，需经由 source 遍历。
        out << "  \"audioModifications\": [\n";
        {
            std::vector<std::string> entries;
            for (auto* const source : document->getAudioSources ())
            {
                for (auto* const modification : source->getAudioModifications ())
                {
                    std::ostringstream entry;
                    entry << "    {";
                    entry << "\"persistentID\": " << jsonEscape (modification->getPersistentID ().c_str ());
                    entry << ", \"name\": " << jsonOptionalString (modification->getName ());
                    entry << ", \"audioSourcePersistentID\": "
                          << jsonEscape (source->getPersistentID ().c_str ());
                    entry << ", \"playbackRegionCount\": "
                          << modification->getPlaybackRegions ().size ();
                    entry << "}";
                    entries.push_back (entry.str ());
                }
            }
            for (std::size_t i = 0; i < entries.size (); ++i)
            {
                out << entries[i];
                if (i + 1 < entries.size ())
                    out << ",";
                out << "\n";
            }
        }
        out << "  ],\n";

        // ---- playbackRegions ----
        out << "  \"playbackRegions\": [\n";
        {
            std::vector<std::string> entries;
            for (auto* const source : document->getAudioSources ())
            {
                for (auto* const modification : source->getAudioModifications ())
                {
                    for (auto* const region : modification->getPlaybackRegions ())
                    {
                        std::ostringstream entry;
                        entry << "    {";
                        entry << "\"name\": " << jsonOptionalString (region->getName ());

                        // 与 modification 的关系（即"源上的哪个区间"）
                        entry << ", \"audioSourcePersistentID\": "
                              << jsonEscape (source->getPersistentID ().c_str ());
                        entry << ", \"audioModificationPersistentID\": "
                              << jsonEscape (modification->getPersistentID ().c_str ());

                        // modification 时间轴（源内）
                        entry << ", \"startInModificationTime\": "
                              << region->getStartInAudioModificationTime ();
                        entry << ", \"durationInModificationTime\": "
                              << region->getDurationInAudioModificationTime ();

                        // playback 时间轴（时间线上）
                        entry << ", \"startInPlaybackTime\": "
                              << region->getStartInPlaybackTime ();
                        entry << ", \"durationInPlaybackTime\": "
                              << region->getDurationInPlaybackTime ();

                        // 变换标志 —— 拉伸 / 倒放 就藏在这里
                        entry << ", \"isTimestretchEnabled\": "
                              << (region->isTimestretchEnabled () ? "true" : "false");
                        entry << ", \"isTimeStretchReflectingTempo\": "
                              << (region->isTimeStretchReflectingTempo () ? "true" : "false");
                        entry << ", \"hasContentBasedFadeAtHead\": "
                              << (region->hasContentBasedFadeAtHead () ? "true" : "false");
                        entry << ", \"hasContentBasedFadeAtTail\": "
                              << (region->hasContentBasedFadeAtTail () ? "true" : "false");

                        // 颜色字段已刻意省略：ARAColor 属于 ARA 2.0 草案附加项，在本
                        // 目标的编译配置下不可见；且颜色对 HiFiShifter 的渲染映射无关。
                        entry << "}";
                        entries.push_back (entry.str ());
                    }
                }
            }
            for (std::size_t i = 0; i < entries.size (); ++i)
            {
                out << entries[i];
                if (i + 1 < entries.size ())
                    out << ",";
                out << "\n";
            }
        }
        out << "  ]\n";

        out << "}\n";

        const std::string path { resolveOutputPath () };
        if (path.empty ())
            return;

        std::ofstream file { path, std::ios::binary | std::ios::trunc };
        if (!file)
            return;
        file << out.str ();
    }
    catch (...)
    {
        // 探针代码绝不允许把异常抛回宿主。
    }
}
