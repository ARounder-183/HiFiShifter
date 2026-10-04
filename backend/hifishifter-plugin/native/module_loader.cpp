// Windows VST3轻量模块入口：绝对路径加载邻接Rust引擎及依赖，不修改宿主PATH/CWD。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>

namespace {
using Factory = void* (__stdcall*)();
using Entry = bool (__stdcall*)();
SRWLOCK gate = SRWLOCK_INIT;
HMODULE engine = nullptr;
Factory factory = nullptr;
Entry initialize = nullptr;
Entry engine_exit = nullptr;
struct Gate {
    Gate() { AcquireSRWLockExclusive(&gate); }
    ~Gate() { ReleaseSRWLockExclusive(&gate); }
};

// 只加载自身bundle的engine，DLL_LOAD_DIR让SoundTouch/DirectML导入不依赖项目目录。
bool load_engine() noexcept {
    Gate lock;
    if (engine) return true;
    try {
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&load_engine), &self)) return false;
        wchar_t buffer[32768];
        DWORD length = GetModuleFileNameW(self, buffer, 32768);
        if (!length || length >= 32768) return false;
        std::wstring path(buffer, length);
        auto slash = path.find_last_of(L"\\/");
        if (slash == std::wstring::npos) return false;
        path.resize(slash + 1);
        path += L"HiFiShifterEngine.dll";
        HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module) { OutputDebugStringW(L"HiFiShifter: adjacent engine/dependency load failed\n"); return false; }
        auto get = reinterpret_cast<Factory>(GetProcAddress(module, "GetPluginFactory"));
        auto init = reinterpret_cast<Entry>(GetProcAddress(module, "InitDll"));
        auto exit = reinterpret_cast<Entry>(GetProcAddress(module, "ExitDll"));
        if (!get || !init || !exit) { FreeLibrary(module); return false; }
        factory = get; initialize = init; engine_exit = exit; engine = module;
        return true;
    } catch (...) { return false; }
}
}

// 工厂对象与所有音频/ARA/GUI接口都由同进程Rust引擎提供，不启动任何辅助应用。
extern "C" __declspec(dllexport) void* __stdcall GetPluginFactory() {
    return load_engine() ? factory() : nullptr;
}
// 宿主显式初始化发生在LoadLibrary之后；不得在DllMain跑ARA或WebView初始化。
extern "C" __declspec(dllexport) bool __stdcall InitDll() {
    return load_engine() && initialize();
}
// 先终止引擎；尚有原生窗口时不释放模块。WebView激活后引擎有自己的进程期pin。
extern "C" __declspec(dllexport) bool __stdcall ExitDll() {
    Gate lock;
    if (!engine) return true;
    if (!engine_exit()) return false;
    HMODULE old = engine;
    engine = nullptr; factory = nullptr; initialize = nullptr; engine_exit = nullptr;
    FreeLibrary(old);
    return true;
}
