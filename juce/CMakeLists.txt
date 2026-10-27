cmake_minimum_required(VERSION 3.24)

# Public version changes require an explicit user request; small updates use docs/updates.md.
project(HachiShifterNext VERSION 0.27 LANGUAGES C CXX)

option(HACHI_ENABLE_ASAN "Enable AddressSanitizer for native crash diagnosis" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
# USTX is YAML; use the bounded native parser rather than a partial YAML reader.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml_cpp
    URL https://codeload.github.com/jbeder/yaml-cpp/tar.gz/refs/tags/0.8.0
    URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
# yaml-cpp 0.8.0 declares an old minimum; CMake 4 needs an explicit compatibility
# floor. Keep it scoped to this dependency, not JUCE or the application.
function(hachi_make_yaml_available)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_MakeAvailable(yaml_cpp)
endfunction()
hachi_make_yaml_available()
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.8
    GIT_SHALLOW TRUE
    # Populate first; apply the native helper-tool compatibility patch before
    # adding JUCE. This directory intentionally does not contain CMakeLists.txt.
    SOURCE_SUBDIR hachi-deferred-configure
)
FetchContent_MakeAvailable(JUCE)
# JUCE's helper-tool bootstrap starts a second CMake configure. On native
# Windows that child must receive this same Clang-CL/xwin toolchain; otherwise
# it silently picks MSYS2 cc.exe/c++.exe and JUCE rejects it as MinGW.
if(WIN32 AND CMAKE_TOOLCHAIN_FILE)
    get_filename_component(HACHI_TOOLCHAIN_FILE_ABS
        "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE)
    set(HACHI_JUCEAIDE_FILE
        "${juce_SOURCE_DIR}/extras/Build/juceaide/CMakeLists.txt")
    file(READ "${HACHI_JUCEAIDE_FILE}" HACHI_JUCEAIDE_SOURCE)
    if(NOT HACHI_JUCEAIDE_SOURCE MATCHES "HachiShifter passes the native toolchain")
        string(REPLACE
            "\"-DCMAKE_BUILD_TYPE=Custom\""
            "\"-DCMAKE_BUILD_TYPE=Custom\"\n            \"-DCMAKE_TOOLCHAIN_FILE=${HACHI_TOOLCHAIN_FILE_ABS}\" # HachiShifter passes the native toolchain"
            HACHI_JUCEAIDE_SOURCE "${HACHI_JUCEAIDE_SOURCE}")
        file(WRITE "${HACHI_JUCEAIDE_FILE}" "${HACHI_JUCEAIDE_SOURCE}")
    endif()
endif()
# Clang-CL 22 no longer accepts JUCE 8.0.8's pointer-to-array to
# initializer_list conversion in AudioPluginInstance. Use JUCE's existing
# array conversion helper explicitly; this is ABI-neutral and also works with
# MSVC and older Clang-CL releases.
if(WIN32)
    set(HACHI_AUDIO_PLUGIN_FILE
        "${juce_SOURCE_DIR}/modules/juce_audio_processors/processors/juce_AudioPluginInstance.h")
    file(READ "${HACHI_AUDIO_PLUGIN_FILE}" HACHI_AUDIO_PLUGIN_SOURCE)
    if(NOT HACHI_AUDIO_PLUGIN_SOURCE MATCHES "HachiShifter array-layout compatibility")
        string(REPLACE
            "AudioPluginInstance (const short channelLayoutList[numLayouts][2]) : AudioProcessor (channelLayoutList) {}"
            "AudioPluginInstance (const short (&channelLayoutList)[numLayouts][2]) : AudioProcessor (channelLayoutList) {} // HachiShifter array-layout compatibility"
            HACHI_AUDIO_PLUGIN_SOURCE "${HACHI_AUDIO_PLUGIN_SOURCE}")
        file(WRITE "${HACHI_AUDIO_PLUGIN_FILE}" "${HACHI_AUDIO_PLUGIN_SOURCE}")
    endif()
    set(HACHI_AUDIO_PROCESSOR_FILE
        "${juce_SOURCE_DIR}/modules/juce_audio_processors/processors/juce_AudioProcessor.h")
    file(READ "${HACHI_AUDIO_PROCESSOR_FILE}" HACHI_AUDIO_PROCESSOR_SOURCE)
    if(NOT HACHI_AUDIO_PROCESSOR_SOURCE MATCHES "HachiShifter array-layout constructor")
        string(REPLACE
            "    AudioProcessor (const BusesProperties& ioLayouts);"
            "    AudioProcessor (const BusesProperties& ioLayouts);\n    // HachiShifter array-layout constructor\n    template <size_t numLayouts>\n    AudioProcessor (const short (&layouts)[numLayouts][2])\n        : AudioProcessor (busesPropertiesFromLayoutArray (layoutListToArray (layouts))) {}"
            HACHI_AUDIO_PROCESSOR_SOURCE "${HACHI_AUDIO_PROCESSOR_SOURCE}")
        file(WRITE "${HACHI_AUDIO_PROCESSOR_FILE}" "${HACHI_AUDIO_PROCESSOR_SOURCE}")
    endif()
endif()
add_subdirectory("${juce_SOURCE_DIR}" "${juce_BINARY_DIR}" EXCLUDE_FROM_ALL)

# JUCE 8.0.8 creates Windows peers with Direct2D before application code can
# choose an engine. Start with software rendering to avoid driver-dependent
# hangs during DocumentWindow construction (reproduced with Wine/WSLg).
if(WIN32)
    set(HACHI_WINDOWING_FILE "${juce_SOURCE_DIR}/modules/juce_gui_basics/native/juce_Windowing_windows.cpp")
    file(READ "${HACHI_WINDOWING_FILE}" HACHI_WINDOWING_SOURCE)
    set(HACHI_WINDOWING_ORIGINAL "${HACHI_WINDOWING_SOURCE}")
    string(REPLACE
        "new HWNDComponentPeer { *this, styleFlags, (HWND) parentHWND, false, 1 }"
        "new HWNDComponentPeer { *this, styleFlags, (HWND) parentHWND, false, 0 }"
        HACHI_WINDOWING_SOURCE "${HACHI_WINDOWING_SOURCE}")
    if(NOT HACHI_WINDOWING_SOURCE MATCHES "parentHWND, false, 0")
        message(FATAL_ERROR "JUCE Windows software-peer patch no longer matches")
    endif()
    # Restore the original site in caches patched by older releases, then
    # subscribe only Direct2D peers. Software fallback must not wait on DXGI.
    string(REPLACE
        "(void) currentMonitor; // HachiShifter software rendering: no DXGI wait thread"
        "VBlankDispatcher::getInstance()->updateDisplay (*this, currentMonitor);"
        HACHI_WINDOWING_SOURCE "${HACHI_WINDOWING_SOURCE}")
    if(NOT HACHI_WINDOWING_SOURCE MATCHES "HachiShifter selective VBlank")
        string(REPLACE
            "VBlankDispatcher::getInstance()->updateDisplay (*this, currentMonitor);"
            "if (renderContext != nullptr && getCurrentRenderingEngine() != 0) VBlankDispatcher::getInstance()->updateDisplay (*this, currentMonitor); // HachiShifter selective VBlank"
            HACHI_WINDOWING_SOURCE "${HACHI_WINDOWING_SOURCE}")
    endif()
    if(NOT HACHI_WINDOWING_SOURCE MATCHES "HachiShifter engine refresh")
        string(REPLACE
            "renderContext = contextDescriptors[e].construct (*this);"
            "renderContext = contextDescriptors[e].construct (*this);\n        if (e == 0) { if (auto* dispatcher = VBlankDispatcher::getInstanceWithoutCreating()) dispatcher->removeListener (*this); }\n        else updateCurrentMonitorAndRefreshVBlankDispatcher (ForceRefreshDispatcher::yes); // HachiShifter engine refresh"
            HACHI_WINDOWING_SOURCE "${HACHI_WINDOWING_SOURCE}")
    endif()
    if(NOT HACHI_WINDOWING_SOURCE MATCHES "HachiShifter selective VBlank" OR
       NOT HACHI_WINDOWING_SOURCE MATCHES "HachiShifter engine refresh")
        message(FATAL_ERROR "JUCE rendering engine scheduling patch no longer matches")
    endif()
    # The peer-level onVBlank used to flush GDI's deferred repaint queue.
    # Without DXGI notifications, enqueue WM_PAINT directly. Windows still
    # coalesces invalid rectangles; do not synchronously paint from callbacks.
    string(REPLACE
        "deferredRepaints.add (area);"
        "auto invalidRect = D2DUtilities::toRECT (area); InvalidateRect (peer.getHWND(), &invalidRect, FALSE); // HachiShifter WM_PAINT scheduling"
        HACHI_WINDOWING_SOURCE "${HACHI_WINDOWING_SOURCE}")
    if(NOT HACHI_WINDOWING_SOURCE MATCHES "HachiShifter WM_PAINT scheduling")
        message(FATAL_ERROR "JUCE GDI repaint patch no longer matches")
    endif()
    if(NOT HACHI_WINDOWING_ORIGINAL STREQUAL HACHI_WINDOWING_SOURCE)
        file(WRITE "${HACHI_WINDOWING_FILE}.hachi-patch-tmp" "${HACHI_WINDOWING_SOURCE}")
        file(RENAME "${HACHI_WINDOWING_FILE}.hachi-patch-tmp" "${HACHI_WINDOWING_FILE}")
    endif()
    set(HACHI_VBLANK_FILE "${juce_SOURCE_DIR}/modules/juce_gui_basics/native/juce_VBlank_windows.cpp")
    file(READ "${HACHI_VBLANK_FILE}" HACHI_VBLANK_SOURCE)
    set(HACHI_VBLANK_ORIGINAL "${HACHI_VBLANK_SOURCE}")
    string(REPLACE "for (;;)" "while (! threadShouldExit())"
        HACHI_VBLANK_SOURCE "${HACHI_VBLANK_SOURCE}")
    if(NOT HACHI_VBLANK_ORIGINAL STREQUAL HACHI_VBLANK_SOURCE)
        file(WRITE "${HACHI_VBLANK_FILE}.hachi-patch-tmp" "${HACHI_VBLANK_SOURCE}")
        file(RENAME "${HACHI_VBLANK_FILE}.hachi-patch-tmp" "${HACHI_VBLANK_FILE}")
    endif()
endif()

# GAME, FCPE and NSF-HiFiGAN use the official ONNX Runtime C++ API.  The runtime is part of
# the application package, while the considerably larger model files remain in
# the optional models/ directory.  Linux packages use CPU inference; Windows
# packages include the official DirectML provider and retain CPU fallback.
option(HACHI_ENABLE_ONNX_ANALYSIS "Enable GAME/FCPE/NSF-HiFiGAN ONNX inference" ON)
if(HACHI_ENABLE_ONNX_ANALYSIS AND (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    set(HACHI_ORT_VERSION "1.22.0")
    if(WIN32)
        set(HACHI_ORT_ARCHIVE
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${HACHI_ORT_VERSION}")
        set(HACHI_ORT_HASH "SHA256=29f9872d786236b79aa83f94482f3a17c14297e4833768d6d0ed4883ee732e60")
        set(HACHI_ORT_INCLUDE_REL "build/native/include")
        set(HACHI_ORT_LIBRARY_REL "runtimes/win-x64/native/onnxruntime.lib")
        set(HACHI_ORT_RUNTIME_REL "runtimes/win-x64/native/onnxruntime.dll")
        set(HACHI_ORT_PACKAGE_NAME "onnxruntime.dll")
        FetchContent_Declare(directml_runtime
            URL "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4"
            URL_HASH "SHA256=4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(directml_runtime)
    else()
        set(HACHI_ORT_ARCHIVE
            "https://github.com/microsoft/onnxruntime/releases/download/v${HACHI_ORT_VERSION}/onnxruntime-linux-x64-${HACHI_ORT_VERSION}.tgz")
        set(HACHI_ORT_HASH "")
        set(HACHI_ORT_INCLUDE_REL "include")
        set(HACHI_ORT_LIBRARY_REL "lib/libonnxruntime.so")
        set(HACHI_ORT_RUNTIME_REL "lib/libonnxruntime.so.${HACHI_ORT_VERSION}")
        set(HACHI_ORT_PACKAGE_NAME "libonnxruntime.so.1")
    endif()
    if(HACHI_ORT_HASH)
        FetchContent_Declare(onnxruntime URL "${HACHI_ORT_ARCHIVE}"
            URL_HASH "${HACHI_ORT_HASH}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    else()
        FetchContent_Declare(onnxruntime URL "${HACHI_ORT_ARCHIVE}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()
    FetchContent_MakeAvailable(onnxruntime)
    add_library(hachi_onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(hachi_onnxruntime PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/${HACHI_ORT_RUNTIME_REL}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/${HACHI_ORT_INCLUDE_REL}")
    if(WIN32)
        set_target_properties(hachi_onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/${HACHI_ORT_LIBRARY_REL}")
    endif()
endif()

# WORLD is vendored so model-free builds have a genuinely independent WORLD
# analysis/synthesis path instead of routing the WORLD selector through the
# common phase-vocoder fallback.
set(WORLD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WORLD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/world EXCLUDE_FROM_ALL)

# LLSM2 is integrated as its speech-model algorithm library only; none of the
# moresampler command-line frontend is built or packaged.
add_subdirectory(third_party/ciglet EXCLUDE_FROM_ALL)
add_subdirectory(third_party/libllsm2 EXCLUDE_FROM_ALL)
target_compile_definitions(ciglet PUBLIC FP_TYPE=float)
target_compile_definitions(llsm PUBLIC FP_TYPE=float)

juce_add_gui_app(HachiShifterNext
    PRODUCT_NAME "HachiShifter Next"
    COMPANY_NAME "funnymdzz"
    BUNDLE_ID "io.github.funnymdzz.hachishifter.next"
    VERSION "${PROJECT_VERSION}"
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
    ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon-small.png"
    NEEDS_CURL FALSE
    NEEDS_WEB_BROWSER FALSE
)

# This JUCE revision tracks the icon, but omits Info.txt from the generated
# Windows resource dependencies. Keep Explorer's EXE version in sync on updates.
if(WIN32)
    get_target_property(HACHI_JUCE_CODE HachiShifterNext JUCE_GENERATED_SOURCES_DIRECTORY)
    get_target_property(HACHI_JUCE_INFO HachiShifterNext JUCE_INFO_FILE)
    add_custom_command(OUTPUT "${HACHI_JUCE_CODE}/HachiShifterNext_resources.rc"
        APPEND DEPENDS "${HACHI_JUCE_INFO}")
endif()

juce_add_binary_data(NoteDanceAssets
    NAMESPACE NoteDanceData
    HEADER_NAME NoteDanceData.h
    SOURCES
        resources/note-dance/danceF1.png
        resources/note-dance/danceEight.png
        resources/note-dance/danceF2.png
        resources/note-dance/danceQ.png
)
target_link_libraries(HachiShifterNext PRIVATE NoteDanceAssets)

target_sources(HachiShifterNext PRIVATE
    src/Main.cpp
    src/MainComponent.cpp
    src/MainProjectMaterialRegistration.cpp
    src/AccompanimentTempo.cpp
    src/AccompanimentTempo.h
    src/AccompanimentTempoPanel.h
    src/Hamood.cpp
    src/MainComponentHamood.cpp
    src/MainComponentMcp.cpp
    src/backend/LiveMcpBridge.cpp
    src/MainComponentDiffSinger.cpp
    src/MainComponent.h
    src/AudioEngine.cpp
    src/AudioEngine.h
    src/ProjectModel.cpp
    src/ProjectModel.h
    src/Pinyin.cpp
    src/Pinyin.h
    src/PianoRollComponent.cpp
    src/PianoRollComponent.h
    src/NativePitchVoicingDisplay.cpp
    src/NativePitchVoicingDisplay.h
    src/TrackListComponent.cpp
    src/TrackListComponent.h
    src/TimelineComponent.cpp
    src/TimelineComponent.h
    src/Theme.cpp
    src/Theme.h
    src/I18n.cpp
    src/I18n.h
    src/Pinyin.cpp
    src/Pinyin.h
    src/PinyinTable.inc
    src/SettingsComponent.cpp
    src/SettingsComponent.h
    src/SampleSettings.cpp
    src/PronunciationTiming.cpp
    src/PronunciationTiming.h
    src/OttoMaterialIO.cpp
    src/OttoMaterialIO.h
    src/OttoMaterialMigration.cpp
    src/OttoMaterialMigration.h
    src/MaterialPronunciationCatalog.cpp
    src/MaterialPronunciationCatalog.h
    src/CommunityDatasetIO.cpp
    src/CommunityDatasetIO.h
    src/CommunityDatasetImportDialog.cpp
    src/CommunityDatasetImportDialog.h
    src/ProjectMaterialRegistration.cpp
    src/ProjectMaterialRegistration.h
    src/ProjectMaterialRegistrationFacts.cpp
    src/ProjectMaterialRegistrationFacts.h
    src/ProjectMaterialRegistrationTypes.h
    src/ProjectMaterialRegistrationModel.inc
    src/MaterialAnnotationSession.cpp
    src/MaterialAnnotationSession.h
    src/MaterialAnnotationEditor.cpp
    src/MaterialAnnotationEditor.h
    src/PronunciationCandidateBox.cpp
    src/PronunciationCandidateBox.h
    src/VoicebankSettingsComponent.cpp
    src/OtoWaveformEditorComponent.cpp
    src/SampleSettings.h
    src/AssetManagerComponent.cpp
    src/AssetManagerComponent.h
    src/backend/Mld5Renderer.cpp
    src/backend/Mld5Renderer.h
    src/backend/Mld3Renderer.cpp
    src/backend/Mld3Renderer.h
    src/backend/MelodyneImporter.cpp
    src/backend/MelodyneProvider.cpp
    src/backend/UstImporter.cpp
    src/backend/UstxImporter.cpp
    src/backend/UstxImporter.h
    src/backend/MelodyneImporter.h
    src/backend/MelodyneProvider.h
    src/backend/McpServer.cpp
    src/backend/McpServer.h
    src/backend/RenderService.cpp
    src/backend/RenderService.h
    src/backend/UtauRenderer.cpp
    src/backend/DiffSingerRenderer.cpp
    src/backend/DiffSingerRenderer.h
    src/backend/UtauRenderer.h
    src/backend/UtauOtoOverride.h
    src/backend/AmplitudeEnvelopeCurve.h
    src/backend/WorldRenderer.cpp
    src/backend/WorldRenderer.h
    src/backend/AudioFileReader.cpp
    src/backend/AudioFileReader.h
    src/backend/NativeAnalyzer.cpp
    src/backend/NativeAnalyzer.h
    src/backend/AnalysisService.cpp
    src/backend/OtoAudioAnalysis.cpp
    src/backend/AnalysisService.h
    src/backend/GameAnalyzer.cpp
    src/backend/GameAnalyzer.h
    src/backend/FcpeAnalyzer.cpp
    src/backend/FcpeAnalyzer.h
    src/backend/NsfHifiganRenderer.cpp
    src/backend/NsfHifiganRenderer.h
    src/backend/Llsm2Renderer.cpp
    src/backend/Llsm2Renderer.h
    src/backend/WcsNonLoopFeatures.cpp
    src/backend/WcsNonLoopFeatures.h
    src/backend/OrtExecution.cpp
    src/backend/OrtExecution.h
)

# The xwin/MSVC-ABI build links JUCE against the common-controls v6 import
# library.  Without an embedded activation-context manifest Windows loads the
# legacy v5 COMCTL32 first and aborts at startup with "ordinal 345 not found".
# Keep the manifest as an explicit PE resource because the cross linker runs
# with /manifest:no and therefore does not invoke mt.exe implicitly.
# The explicit manifest resource is only required by the xwin cross build.
# Native Visual Studio builds already receive JUCE's manifest; adding this
# resource there produces a duplicate MANIFEST:1 and fails with CVT1100.
if(WIN32 AND CMAKE_TOOLCHAIN_FILE MATCHES "windows-msvc-xwin\\.cmake$")
    target_sources(HachiShifterNext PRIVATE
        resources/windows/HachiShifterNext.rc
        resources/windows/HachiShifterNext.manifest
    )
endif()

target_compile_definitions(HachiShifterNext PRIVATE
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:HachiShifterNext,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_PLUGINHOST_VST3=1
)

target_include_directories(HachiShifterNext PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/signalsmith-stretch"
)

if(MSVC)
    # The combined diagnostic entry point contains both upstream and local
    # smoke fixtures. Reserve sufficient main-thread stack for its MSVC frame.
    target_link_options(HachiShifterNext PRIVATE /STACK:8388608)
    # Avoid relying on the newer constexpr mutex ABI during startup when an
    # older compatible VC runtime (including Wine's implementation) is loaded.
    target_compile_definitions(HachiShifterNext PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

target_link_libraries(HachiShifterNext PRIVATE
    yaml-cpp::yaml-cpp
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_gui_extra
    juce::juce_data_structures
    juce::juce_cryptography
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
    world::core
    llsm
)

if(TARGET hachi_onnxruntime)
    target_link_libraries(HachiShifterNext PRIVATE hachi_onnxruntime)
    target_compile_definitions(HachiShifterNext PRIVATE HACHI_HAS_ONNX_ANALYSIS=1)
    if(WIN32)
        target_include_directories(HachiShifterNext PRIVATE "${directml_runtime_SOURCE_DIR}/include")
        target_compile_definitions(HachiShifterNext PRIVATE HACHI_HAS_ORT_DML=1)
    else()
        target_compile_definitions(HachiShifterNext PRIVATE HACHI_HAS_ORT_DML=0)
    endif()
    add_custom_command(TARGET HachiShifterNext POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${onnxruntime_SOURCE_DIR}/${HACHI_ORT_RUNTIME_REL}"
            "$<TARGET_FILE_DIR:HachiShifterNext>/${HACHI_ORT_PACKAGE_NAME}")
    if(WIN32)
        add_custom_command(TARGET HachiShifterNext POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime_providers_shared.dll"
                "$<TARGET_FILE_DIR:HachiShifterNext>/onnxruntime_providers_shared.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${directml_runtime_SOURCE_DIR}/bin/x64-win/DirectML.dll"
                "$<TARGET_FILE_DIR:HachiShifterNext>/DirectML.dll")
    endif()
    if(UNIX AND NOT APPLE)
        set_target_properties(HachiShifterNext PROPERTIES
            BUILD_RPATH "$ORIGIN")
    endif()
else()
    target_compile_definitions(HachiShifterNext PRIVATE HACHI_HAS_ONNX_ANALYSIS=0 HACHI_HAS_ORT_DML=0)
endif()

add_custom_command(TARGET HachiShifterNext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:HachiShifterNext>/engines/diffsinger"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/bridge.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/expressions.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/variance_retake.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/compatibility.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/runtime.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/inference.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/timing.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/pronunciation.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/english_g2p.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/requirements.txt"
        "$<TARGET_FILE_DIR:HachiShifterNext>/engines/diffsinger")

# Keep locally prepared analysis weights beside the binary, so builds and
# portable packages use the same default without absolute machine paths.
foreach(analysis_pack game/medium fcpe)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../models/${analysis_pack}")
        add_custom_command(TARGET HachiShifterNext POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/../models/${analysis_pack}"
                "$<TARGET_FILE_DIR:HachiShifterNext>/models/${analysis_pack}")
    endif()
endforeach()
add_custom_command(TARGET HachiShifterNext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/diffsinger/pronunciation-data"
        "$<TARGET_FILE_DIR:HachiShifterNext>/engines/diffsinger/pronunciation-data")

# UTAU voicebanks commonly store oto.ini in CP932/Shift-JIS.  iconv is part of
# libc on Linux and a system library on macOS; Windows uses MultiByteToWideChar.
if(NOT WIN32)
    find_package(Iconv)
    if(Iconv_FOUND)
        target_link_libraries(HachiShifterNext PRIVATE Iconv::Iconv)
        target_compile_definitions(HachiShifterNext PRIVATE HACHI_HAS_ICONV=1)
    endif()
endif()

if(MSVC)
    target_compile_options(HachiShifterNext PRIVATE /utf-8 /W4)
    if(HACHI_ENABLE_ASAN)
        target_compile_options(HachiShifterNext PRIVATE /fsanitize=address /Zi /GL-)
        target_link_options(HachiShifterNext PRIVATE /INCREMENTAL:NO)
    endif()
    # Keep a linker address map beside native builds.  It is small, is not
    # packaged with the application, and makes field crash addresses from
    # Windows Event Viewer actionable even for optimised Release binaries.
    target_link_options(HachiShifterNext PRIVATE
        "/MAP:${CMAKE_BINARY_DIR}/HachiShifterNext-$<CONFIG>.map")
else()
    target_compile_options(HachiShifterNext PRIVATE -Wall -Wextra -Wpedantic)
endif()
