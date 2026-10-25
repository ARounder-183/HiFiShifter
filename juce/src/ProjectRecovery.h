#pragma once
#include "ProjectModel.h"
#include "ProjectFileIO.h"
#include <juce_cryptography/juce_cryptography.h>
#include <future>
#include <optional>

namespace hachi
{
class ProjectRecovery final
{
public:
    struct Entry { juce::File file, original; juce::String name, savedAt; };
    struct Lease
    {
        explicit Lease(const juce::File& file)
            : lock("HachiShifter-recovery-" + juce::SHA256(file.getFullPathName().toLowerCase().toUTF8()).toHexString()),
              acquired(lock.enter(0)) {}
        ~Lease() { if (acquired) lock.exit(); }
        juce::InterProcessLock lock;
        bool acquired;
    };
    explicit ProjectRecovery(juce::File directory)
        : root(std::move(directory)), file(root.getChildFile(juce::Uuid().toString()+".hjpx")),
          lease(std::make_unique<Lease>(file)) {}
    ~ProjectRecovery() { finish(); } // Only an approved save/discard removes recovery data.
    juce::File directory() const { return root; }
    juce::File currentFile() const { return file; }
    bool busy() const { return pending.valid(); }
    std::optional<juce::String> poll()
    {
        if (!pending.valid() || pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return {};
        const auto result = pending.get();
        if (result.ok) { lastSaved = result.revision; hasSaved = true; }
        return result.ok ? juce::String{} : result.error;
    }
    bool capture(const ProjectModel& model, const juce::File& original)
    {
        if (pending.valid() || !lease->acquired) return false;
        const auto data = model.snapshot();
        const auto revision = model.revisionNumber();
        const auto destination = file;
        pending = std::async(std::launch::async, [data, revision, original, destination]
        {
            Result result; result.revision=revision;
            try
            {
                const auto created=destination.getParentDirectory().createDirectory();
                if (created.failed()) result.error=created.getErrorMessage();
                else result.ok=ProjectModel::saveRecoverySnapshot(data,destination,original,result.error);
            }
            catch (const std::exception& e) { result.error=juce::String::fromUTF8(e.what()); }
            catch (...) { result.error="Automatic project save failed."; }
            return result;
        });
        return true;
    }
    std::optional<juce::String> tick(const ProjectModel& model, const juce::File& original,
                                    bool dirty, double nowMilliseconds)
    {
        auto result=poll();
        if (nowMilliseconds>=nextAttempt)
        {
            nextAttempt=nowMilliseconds+30000.0;
            if (dirty && (!hasSaved || lastSaved!=model.revisionNumber())) capture(model,original);
        }
        return result;
    }
    void finish() { if (pending.valid()) { pending.wait(); (void)poll(); } }
    void discard()
    {
        finish(); // A stale in-flight writer must never recreate a retired document.
        if (file.getParentDirectory()==root) file.deleteFile();
        if (claimedFile!=juce::File{} && claimedFile.getParentDirectory()==root) claimedFile.deleteFile();
        claimedFile=juce::File{}; claimedLease.reset(); hasSaved=false;
        nextAttempt=juce::Time::getMillisecondCounterHiRes()+30000.0;
    }
    static std::unique_ptr<Lease> tryClaim(const juce::File& file)
    {
        auto result=std::make_unique<Lease>(file);
        return result->acquired ? std::move(result) : nullptr;
    }
    void adopt(const juce::File& source, std::unique_ptr<Lease> sourceLease)
    {
        claimedFile=source; claimedLease=std::move(sourceLease);
    }
    std::vector<Entry> entries() const
    {
        std::vector<Entry> result;
        for (const auto& candidate:root.findChildFiles(juce::File::findFiles,false,"*.hjpx"))
        {
            if (candidate==file || candidate==claimedFile) continue;
            auto claim=tryClaim(candidate);
            if (!claim) continue; // Another editor instance still owns this snapshot.
            juce::MemoryBlock bytes;
            if (!candidate.loadFileAsData(bytes)) continue;
            const auto tree=projectio::validatedTree(bytes);
            if (!tree.hasType("HachiShifterProject") || !tree.hasProperty("recoverySavedAt")) continue;
            result.push_back({candidate,juce::File(tree["recoveryOriginalFile"].toString()),
                              tree["name"].toString(),tree["recoverySavedAt"].toString()});
        }
        std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.savedAt>b.savedAt;});
        return result;
    }
private:
    struct Result { bool ok=false; std::uint64_t revision=0; juce::String error; };
    juce::File root, file, claimedFile;
    std::unique_ptr<Lease> lease, claimedLease;
    std::future<Result> pending;
    bool hasSaved=false;
    std::uint64_t lastSaved=0;
    double nextAttempt=juce::Time::getMillisecondCounterHiRes()+30000.0;
};
}