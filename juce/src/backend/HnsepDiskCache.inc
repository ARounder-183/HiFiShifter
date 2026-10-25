// Shared with hf_backend/hn_cache.py. Included in the NSF anonymous namespace.
// Cache identity is the exact model input, not a filename or filesystem timestamp.
namespace hnCache {
constexpr std::size_t maxBytes = 64 * 1024 * 1024;
constexpr const char* revision = "hnsep-complex-periodic-hann-zero-pad32-v1";

juce::String identity(const std::vector<float>& wave, const juce::MemoryBlock& model,
                      const juce::MemoryBlock& config, int rate, int fft, int hop)
{
    const auto descriptor = juce::String(revision) + "\n" + juce::String(rate) + "\n"
        + juce::String(fft) + "\n" + juce::String(hop) + "\n"
        + juce::String(static_cast<juce::int64>(wave.size())) + "\n"
        + juce::SHA256(model).toHexString() + "\n" + juce::SHA256(config).toHexString() + "\n"
        + juce::SHA256(wave.data(), wave.size() * sizeof(float)).toHexString() + "\n";
    return juce::SHA256(descriptor.toRawUTF8(), descriptor.getNumBytesAsUTF8()).toHexString();
}

juce::File root()
{
    auto path = juce::SystemStats::getEnvironmentVariable("HACHISHIFTER_HNSEP_CACHE", {});
    if (path.isNotEmpty()) return juce::File(path);
    path = juce::SystemStats::getEnvironmentVariable("LOCALAPPDATA", {});
    const auto base = path.isEmpty() ? juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                                   : juce::File(path);
    return base.getChildFile("HachiShifter/cache/hnsep-v1");
}

struct Lock {
    juce::InterProcessLock mutex;
    bool held = false;
    explicit Lock(const juce::String& key) : mutex("HachiShifter-HNSEP-" + key)
    {
        const auto deadline = juce::Time::getMillisecondCounterHiRes() + 120000;
        do {
            checkNsfCancellation();
            if ((held = mutex.enter(50))) break;
        } while (juce::Time::getMillisecondCounterHiRes() < deadline);
    }
    ~Lock() { if (held) mutex.exit(); }
};

std::vector<float> load(const juce::String& key, std::size_t count)
{
    if (count == 0 || count > maxBytes / sizeof(float)) return {};
    const auto file = root().getChildFile(key + ".hnh");
    if (file.getSize() != static_cast<juce::int64>(144 + count * sizeof(float))) return {};
    juce::FileInputStream stream(file);
    char magic[8], storedKey[64], checksum[64];
    if (!stream.openedOk() || stream.read(magic, 8) != 8 || std::memcmp(magic, "HNHARM1\n", 8)
        || stream.read(storedKey, 64) != 64 || juce::String::fromUTF8(storedKey, 64) != key
        || stream.readInt64() != static_cast<juce::int64>(count) || stream.read(checksum, 64) != 64) return {};
    std::vector<float> result(count);
    const auto bytes = count * sizeof(float);
    if (stream.read(result.data(), static_cast<int>(bytes)) != static_cast<int>(bytes)
        || juce::SHA256(result.data(), bytes).toHexString() != juce::String::fromUTF8(checksum, 64)
        || !std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); })) return {};
    file.setLastModificationTime(juce::Time::getCurrentTime());
    return result;
}

void save(const juce::String& key, const std::vector<float>& wave)
{
    if (wave.empty() || wave.size() > maxBytes / sizeof(float)) return;
    const auto folder = root();
    if (folder.createDirectory().failed()) return; // Cache storage is optional.
    const auto target = folder.getChildFile(key + ".hnh");
    const juce::TemporaryFile temporary(target,
        folder.getChildFile(key + "-" + juce::Uuid().toString() + ".tmp"));
    {
        juce::FileOutputStream stream(temporary.getFile());
        const auto hash = juce::SHA256(wave.data(), wave.size() * sizeof(float)).toHexString();
        if (!stream.openedOk() || !stream.write("HNHARM1\n", 8) || !stream.write(key.toRawUTF8(), 64)
            || !stream.writeInt64(static_cast<juce::int64>(wave.size())) || !stream.write(hash.toRawUTF8(), 64)
            || !stream.write(wave.data(), wave.size() * sizeof(float))) return;
        stream.flush();
        if (stream.getStatus().failed()) return;
    }
    if (!temporary.overwriteTargetFileWithTemporary()) return;
    // Only our derived .hnh files participate in the 512 MiB disk budget.
    auto files = folder.findChildFiles(juce::File::findFiles, false, "*.hnh");
    files.removeIf([](const auto& file) {
        const auto name = file.getFileNameWithoutExtension();
        return name.length() != 64 || !name.containsOnly("0123456789abcdef");
    });
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return a.getLastModificationTime() < b.getLastModificationTime();
    });
    juce::int64 total = 0;
    for (const auto& file : files) total += file.getSize();
    for (const auto& file : files) {
        if (total <= 512LL * 1024 * 1024) break;
        if (file == target) continue;
        const auto size = file.getSize();
        if (file.deleteFile()) total -= size;
    }
}
} // namespace hnCache
