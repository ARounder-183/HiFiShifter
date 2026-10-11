#pragma once

#include <stdint.h>

#include "ARA_API/ARAInterface.h"

#ifdef __cplusplus
typedef ARA::ARAFactory Ara2ARAFactory;
extern "C" {
#else
typedef ARAFactory Ara2ARAFactory;
#endif

enum Ara2NativeStatus
{
    kAra2NativeOk = 0,
    kAra2NativeInvalidArgument = 1,
    kAra2NativeAraFailure = 2,
    kAra2NativeException = 3
};

enum Ara2NativeScenario
{
    kAra2NativeBasicDocument = 1,
    kAra2NativePropertyUpdates = 2,
    kAra2NativeContentUpdates = 3,
    kAra2NativeContentReading = 4,
    kAra2NativeModificationCloning = 5,
    kAra2NativeFullArchive = 6,
    kAra2NativeSplitPartialArchives = 7,
    kAra2NativeDragDropImport = 8,
    kAra2NativeProcessingAlgorithms = 9,
    kAra2NativeAudioFileChunkSave = 10
};

typedef struct Ara2NativeResult
{
    int32_t status;
    int32_t generation;
    uint64_t callbackCount;
    uint64_t liveObjects;
    char diagnostic[512];
} Ara2NativeResult;

const Ara2ARAFactory* ara2_cpp_test_plugin_factory (Ara2NativeResult* result);

void ara2_cpp_assert_scope_begin (Ara2NativeResult* result);
void ara2_cpp_assert_scope_end (void);

int32_t ara2_cpp_test_host_run (const Ara2ARAFactory* factory,
                                int32_t generation,
                                int32_t scenario,
                                Ara2NativeResult* result);

#ifdef __cplusplus
}
#endif
