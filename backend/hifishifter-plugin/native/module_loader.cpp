// Windows VST3轻量模块入口：绝对路径加载邻接Rust引擎及依赖，不修改宿主PATH/CWD。
//
// 生命周期锁契约（重要）：
//   - gate 是唯一的生命周期锁，保护 engine / factory / initialize / engine_exit / load_failed。
//   - 所有导出入口（GetPluginFactory / InitDll / ExitDll）在函数入口取一次 Gate。
//   - load_engine_locked() 要求调用者已经持有 gate，自己绝不再取锁；否则就是同线程
//     对非递归 SRWLOCK 的二次取锁，会永久阻塞（ac0e4c1c 的回归）。
//   - GetPluginFactory / InitDll 在 Gate 内调用 engine 导出，保证 ExitDll 不会在
//     “读取函数指针”和“调用函数指针”之间清空/卸载 engine。
//   - ExitDll 在 Gate 内清空状态，FreeLibrary 放到 Gate 外，避免在持锁时执行 DllMain。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <string>

namespace {
using Factory = void* (__stdcall*)();
using Entry = bool (__stdcall*)();

SRWLOCK gate = SRWLOCK_INIT;
HMODULE engine = nullptr;
Factory factory = nullptr;
Entry initialize = nullptr;
Entry engine_exit = nullptr;
bool load_failed = false;

struct Gate {
    Gate() { AcquireSRWLockExclusive(&gate); }
    ~Gate() { ReleaseSRWLockExclusive(&gate); }
};

// 仅用于诊断；所有调用都发生在 Gate 内，天然串行。不要在这里做重活。
void log_loader(const wchar_t* message, DWORD error = 0) noexcept {
    wchar_t buffer[512];
    if (error) {
        swprintf_s(buffer, L"HiFiShifter loader: %s (GetLastError=%lu)\n", message, error);
    } else {
        swprintf_s(buffer, L"HiFiShifter loader: %s\n", message);
    }
    OutputDebugStringW(buffer);
}

// 调用者必须持有 gate。本函数绝不再取锁。
bool load_engine_locked() noexcept {
    if (engine) return true;
    if (load_failed) return false;

    try {
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&load_engine_locked), &self)) {
            const DWORD error = GetLastError();
            log_loader(L"GetModuleHandleExW(self) failed", error);
            load_failed = true;
            return false;
        }

        wchar_t buffer[32768];
        const DWORD length = GetModuleFileNameW(self, buffer, 32768);
        if (!length || length >= 32768) {
            const DWORD error = GetLastError();
            log_loader(L"GetModuleFileNameW(self) failed or truncated", error);
            load_failed = true;
            return false;
        }

        std::wstring path(buffer, length);
        const auto slash = path.find_last_of(L"\\/");
        if (slash == std::wstring::npos) {
            log_loader(L"shim path has no directory separator");
            load_failed = true;
            return false;
        }
        path.resize(slash + 1);
        path += L"HiFiShifterEngine.dll";

        HMODULE module = LoadLibraryExW(
            path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module) {
            const DWORD error = GetLastError();
            log_loader(L"LoadLibraryExW(HiFiShifterEngine.dll) failed", error);
            load_failed = true;
            return false;
        }

        auto get = reinterpret_cast<Factory>(GetProcAddress(module, "GetPluginFactory"));
        auto init = reinterpret_cast<Entry>(GetProcAddress(module, "InitDll"));
        auto exit = reinterpret_cast<Entry>(GetProcAddress(module, "ExitDll"));
        if (!get || !init || !exit) {
            log_loader(L"HiFiShifterEngine.dll is missing a VST3 entry point");
            FreeLibrary(module);
            load_failed = true;
            return false;
        }

        factory = get;
        initialize = init;
        engine_exit = exit;
        engine = module;
        return true;
    } catch (...) {
        log_loader(L"unexpected exception while loading HiFiShifterEngine.dll");
        load_failed = true;
        return false;
    }
}
}  // namespace

// 工厂对象与所有音频/ARA/GUI接口都由同进程Rust引擎提供，不启动任何辅助应用。
extern "C" __declspec(dllexport) void* __stdcall GetPluginFactory() {
    Gate lock;
    if (!load_engine_locked() || !factory) return nullptr;
    void* result = factory();
    if (!result) log_loader(L"engine GetPluginFactory returned null");
    return result;
}

// 宿主显式初始化发生在LoadLibrary之后；不得在DllMain跑ARA或WebView初始化。
extern "C" __declspec(dllexport) bool __stdcall InitDll() {
    Gate lock;
    if (!load_engine_locked() || !initialize) return false;
    const bool result = initialize();
    if (!result) log_loader(L"engine InitDll returned false");
    return result;
}

// 先终止引擎；尚有原生窗口时引擎自己会拒绝卸载。
extern "C" __declspec(dllexport) bool __stdcall ExitDll() {
    HMODULE old = nullptr;
    {
        Gate lock;
        if (!engine) {
            load_failed = false;
            return true;
        }
        if (!engine_exit || !engine_exit()) {
            log_loader(L"engine ExitDll failed");
            return false;
        }
        old = engine;
        engine = nullptr;
        factory = nullptr;
        initialize = nullptr;
        engine_exit = nullptr;
        load_failed = false;
    }

    // FreeLibrary 在锁外执行：它可能运行 engine 的 DllMain/静态析构，
    // 不应在持有生命周期锁时调用外部代码。
    if (old && !FreeLibrary(old)) {
        const DWORD error = GetLastError();
        log_loader(L"FreeLibrary(HiFiShifterEngine.dll) failed", error);
        return false;
    }
    return true;
}
