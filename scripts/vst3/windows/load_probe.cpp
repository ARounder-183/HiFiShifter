// Windows VST3 black-box load probe.
//
// This is intentionally independent of the VST3 SDK: it only needs the three
// module entry points and the first few factory/component vtable slots.  The
// probe is run in a separate process by verify.ps1 with a hard timeout, so a
// deadlocked GetPluginFactory fails the gate instead of hanging CI.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string_view>

namespace {
using tresult = std::int32_t;
constexpr tresult kResultOk = 0;

struct PClassInfo {
    unsigned char cid[16];
    std::int32_t cardinality;
    char category[32];
    char name[64];
};

using GetPluginFactoryFn = void* (__stdcall*)();
using InitDllFn = bool (__stdcall*)();
using ExitDllFn = bool (__stdcall*)();
using ReleaseFn = std::uint32_t (__stdcall*)(void*);
using CountClassesFn = std::int32_t (__stdcall*)(void*);
using GetClassInfoFn = tresult (__stdcall*)(void*, std::int32_t, PClassInfo*);
using CreateInstanceFn = tresult (__stdcall*)(void*, const char*, const char*, void**);

// IComponent IID, in the Windows VST3 COM/GUID byte layout:
// {E831FF31-F2D5-4301-928E-BBEE25697802}
const unsigned char kIComponentIid[16] = {
    0x31, 0xFF, 0x31, 0xE8, 0xD5, 0xF2, 0x01, 0x43,
    0x92, 0x8E, 0xBB, 0xEE, 0x25, 0x69, 0x78, 0x02,
};

void** vtbl_of(void* object) {
    return *reinterpret_cast<void***>(object);
}

ReleaseFn release_of(void* object) {
    return reinterpret_cast<ReleaseFn>(vtbl_of(object)[2]);
}

std::string_view fixed_string(const char* data, std::size_t size) {
    std::size_t length = 0;
    while (length < size && data[length] != '\0') ++length;
    return std::string_view(data, length);
}

void print_last_error(const char* what) {
    std::fprintf(stderr, "FAIL: %s (GetLastError=%lu)\n", what,
                 static_cast<unsigned long>(GetLastError()));
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: load_probe <path-to-HiFiShifter.vst3> [--factory-first]\n");
        return 2;
    }
    const bool factory_first = argc > 2 && std::wcscmp(argv[2], L"--factory-first") == 0;

    HMODULE module = nullptr;
    InitDllFn init = nullptr;
    ExitDllFn exit_fn = nullptr;
    GetPluginFactoryFn get_factory = nullptr;
    void* factory = nullptr;
    void* component = nullptr;
    int result = 1;

    auto release_object = [](void* object) {
        if (object) release_of(object)(object);
    };
    // Tear down in the order a real host uses: release the objects first,
    // then ExitDll, then unload the library.  Calling ExitDll while the
    // component is still alive would pin a sequence production never exercises
    // (the shim's gate clears engine state inside ExitDll, so a later Release
    // would touch cleared engine state), and a future engine change could then
    // crash only inside this probe.
    // Returns false only when ExitDll itself ran and returned false.
    auto cleanup = [&]() -> bool {
        if (component) {
            release_object(component);
            component = nullptr;
        }
        if (factory) {
            release_object(factory);
            factory = nullptr;
        }
        bool exit_ok = true;
        if (exit_fn) {
            exit_ok = exit_fn();
            exit_fn = nullptr;
        }
        if (module) {
            FreeLibrary(module);
            module = nullptr;
        }
        return exit_ok;
    };

    module = LoadLibraryW(argv[1]);
    if (!module) {
        print_last_error("LoadLibraryW(HiFiShifter.vst3)");
        goto done;
    }

    init = reinterpret_cast<InitDllFn>(GetProcAddress(module, "InitDll"));
    exit_fn = reinterpret_cast<ExitDllFn>(GetProcAddress(module, "ExitDll"));
    get_factory = reinterpret_cast<GetPluginFactoryFn>(GetProcAddress(module, "GetPluginFactory"));
    if (!init || !exit_fn || !get_factory) {
        std::fprintf(stderr, "FAIL: missing InitDll/ExitDll/GetPluginFactory export\n");
        goto done;
    }

    if (!factory_first) {
        if (!init()) {
            std::fprintf(stderr, "FAIL: InitDll returned false\n");
            goto done;
        }
    }

    factory = get_factory();
    if (!factory) {
        std::fprintf(stderr, "FAIL: GetPluginFactory returned null\n");
        goto done;
    }

    if (factory_first) {
        if (!init()) {
            std::fprintf(stderr, "FAIL: InitDll after GetPluginFactory returned false\n");
            goto done;
        }
    }

    {
        void** factory_vtbl = vtbl_of(factory);
        auto count_classes = reinterpret_cast<CountClassesFn>(factory_vtbl[4]);
        auto get_class_info = reinterpret_cast<GetClassInfoFn>(factory_vtbl[5]);
        auto create_instance = reinterpret_cast<CreateInstanceFn>(factory_vtbl[6]);

        const std::int32_t class_count = count_classes(factory);
        std::printf("countClasses: %d\n", class_count);
        if (class_count != 3) {
            std::fprintf(stderr, "FAIL: expected 3 classes, got %d\n", class_count);
            goto done;
        }

        PClassInfo info{};
        const tresult info_result = get_class_info(factory, 0, &info);
        if (info_result != kResultOk) {
            std::fprintf(stderr, "FAIL: getClassInfo(0) returned %d\n", info_result);
            goto done;
        }
        const std::string_view category = fixed_string(info.category, sizeof(info.category));
        const std::string_view name = fixed_string(info.name, sizeof(info.name));
        std::printf("class[0]: %.*s / %.*s\n",
                    static_cast<int>(category.size()), category.data(),
                    static_cast<int>(name.size()), name.data());
        if (category.find("Audio Module Class") == std::string_view::npos ||
            name != "HiFiShifter") {
            std::fprintf(stderr, "FAIL: unexpected class[0] category/name\n");
            goto done;
        }

        const tresult create_result = create_instance(
            factory, reinterpret_cast<const char*>(info.cid),
            reinterpret_cast<const char*>(kIComponentIid), &component);
        if (create_result != kResultOk || !component) {
            std::fprintf(stderr, "FAIL: createInstance(component) returned %d, object=%p\n",
                         create_result, component);
            goto done;
        }
        std::printf("createInstance(component): ok\n");
    }

    result = 0;

done:
    // The happy path deliberately has no standalone ExitDll call: cleanup()
    // runs the full host-order teardown (Release objects -> ExitDll ->
    // FreeLibrary).  A false return from ExitDll on the happy path still fails
    // the gate; on failure paths the result is already non-zero.
    if (result == 0 && !cleanup()) {
        std::fprintf(stderr, "FAIL: ExitDll returned false\n");
        result = 1;
    }
    if (result == 0) {
        std::printf("PASS\n");
    }
    return result;
}
