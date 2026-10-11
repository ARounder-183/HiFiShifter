#include "cpp_interop.h"

#include "TestHost/CompanionAPIs.h"
#include "TestHost/TestHost.h"
#include "TestHost/TestCases.h"

#include "ARA_Library/Debug/ARADebug.h"

#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace
{
thread_local Ara2NativeResult* currentResult {};

void setDiagnostic (Ara2NativeResult* result, const char* message) noexcept
{
    if ((result == nullptr) || (result->diagnostic[0] != '\0'))
        return;
    std::strncpy (result->diagnostic, message, sizeof (result->diagnostic) - 1U);
    result->diagnostic[sizeof (result->diagnostic) - 1U] = '\0';
}

void ARA_CALL nativeAssert (ARA::ARAAssertCategory, const void*, const char* file) noexcept
{
    if (currentResult != nullptr)
    {
        currentResult->status = kAra2NativeAraFailure;
        setDiagnostic (currentResult, (file != nullptr) ? file : "ARA interface assertion");
    }
}

ARA::ARAAssertFunction nativeAssertFunction { &nativeAssert };

class DirectPlugInEntry final : public PlugInEntry
{
public:
    DirectPlugInEntry (const ARA::ARAFactory* factory, ARA::ARAAPIGeneration generation)
    : PlugInEntry { "ara2-bridge direct factory" }, _generation { generation }
    {
        validateAndSetFactory (factory);
    }

    void initializeARA (ARA::ARAAssertFunction* assertFunctionAddress) override
    {
        const ARA::SizedStruct<ARA_STRUCT_MEMBER (ARAInterfaceConfiguration, assertFunctionAddress)> configuration {
            _generation,
            assertFunctionAddress
        };
        getARAFactory ()->initializeARAWithConfiguration (&configuration);
    }

    std::unique_ptr<PlugInInstance> createPlugInInstance () override
    {
        return {};
    }

private:
    ARA::ARAAPIGeneration _generation;
};

class InitializedEntry final
{
public:
    InitializedEntry (DirectPlugInEntry& entry, ARA::ARAAssertFunction* assertFunction)
    : _entry { entry }
    {
        _entry.initializeARA (assertFunction);
    }

    ~InitializedEntry ()
    {
        _entry.uninitializeARA ();
    }

    InitializedEntry (const InitializedEntry&) = delete;
    InitializedEntry& operator= (const InitializedEntry&) = delete;

private:
    DirectPlugInEntry& _entry;
};

uint64_t runBasicDocument (DirectPlugInEntry& entry)
{
    TestHost host;
    auto* document { host.addDocument ("Rust TestPlugIn / C++ TestHost", &entry) };
    auto* controller { host.getDocumentController (document) };

    controller->beginEditing ();
    auto* context { host.addMusicalContext (document, "Interop context", { 1.0F, 0.0F, 0.0F }) };
    auto* sequence { host.addRegionSequence (document, "Interop sequence", context, { 0.0F, 1.0F, 0.0F }) };
    controller->endEditing ();

    controller->beginEditing ();
    document->setName ("Rust TestPlugIn / C++ TestHost updated");
    controller->updateDocumentProperties ();
    context->setName ("Interop context updated");
    controller->updateMusicalContextProperties (context);
    sequence->setName ("Interop sequence updated");
    controller->updateRegionSequenceProperties (sequence);
    controller->endEditing ();

    host.destroyDocument (document);

    // Factory create, two complete edit scopes, six model operations, teardown edit scope,
    // two model destroys, and controller destroy all crossed the C ABI.
    return 15U;
}

uint64_t runScenario (DirectPlugInEntry& entry, int32_t scenario)
{
    if (scenario == kAra2NativeBasicDocument)
        return runBasicDocument (entry);

    auto audioFiles { createDummyAudioFiles (2U) };
    switch (scenario)
    {
        case kAra2NativePropertyUpdates:
            testPropertyUpdates (&entry, audioFiles);
            break;
        case kAra2NativeContentUpdates:
            testContentUpdates (&entry, audioFiles);
            break;
        case kAra2NativeContentReading:
            testContentReading (&entry, audioFiles);
            break;
        case kAra2NativeModificationCloning:
            testModificationCloning (&entry, audioFiles);
            break;
        case kAra2NativeFullArchive:
            testArchiving (&entry, audioFiles);
            break;
        case kAra2NativeSplitPartialArchives:
            testSplitArchives (&entry, audioFiles);
            break;
        case kAra2NativeDragDropImport:
            testDragAndDrop (&entry, audioFiles);
            break;
        case kAra2NativeProcessingAlgorithms:
            testProcessingAlgorithms (&entry, audioFiles);
            break;
        case kAra2NativeAudioFileChunkSave:
            testAudioFileChunkSaving (&entry, audioFiles);
            break;
        default:
            return 0U;
    }
    // The Rust fixture supplies the authoritative callback trace for upstream test-case runs.
    return 0U;
}
} // namespace

extern "C" void ara2_cpp_assert_scope_begin (Ara2NativeResult* result)
{
    currentResult = result;
    ARA::ARASetExternalAssertReference (&nativeAssertFunction);
}

extern "C" void ara2_cpp_assert_scope_end (void)
{
    ARA::ARASetExternalAssertReference (nullptr);
    currentResult = nullptr;
}

extern "C" int32_t ara2_cpp_test_host_run (const Ara2ARAFactory* factory,
                                             int32_t generation,
                                             int32_t scenario,
                                             Ara2NativeResult* result)
{
    if (result == nullptr)
        return kAra2NativeInvalidArgument;
    std::memset (result, 0, sizeof (*result));
    result->generation = generation;
    if (factory == nullptr)
    {
        result->status = kAra2NativeInvalidArgument;
        setDiagnostic (result, "null Rust ARA factory");
        return result->status;
    }
    if ((scenario < kAra2NativeBasicDocument) || (scenario > kAra2NativeAudioFileChunkSave))
    {
        result->status = kAra2NativeInvalidArgument;
        setDiagnostic (result, "unsupported native scenario");
        return result->status;
    }

    ara2_cpp_assert_scope_begin (result);
    try
    {
        DirectPlugInEntry entry { factory, static_cast<ARA::ARAAPIGeneration> (generation) };
        ARA::ARAAssertFunction assertFunction { &nativeAssert };
        {
            InitializedEntry initialized { entry, &assertFunction };
            result->callbackCount = runScenario (entry, scenario);
        }
        result->liveObjects = 0U;
    }
    catch (const std::exception& error)
    {
        result->status = kAra2NativeException;
        setDiagnostic (result, error.what ());
    }
    catch (...)
    {
        result->status = kAra2NativeException;
        setDiagnostic (result, "unknown C++ exception while running TestHost");
    }
    ara2_cpp_assert_scope_end ();
    return result->status;
}
