/*
 * HiFiShifter VST3 —— 黑盒加载探针。
 *
 * 用法：probe <path/to/module>            （Linux: HiFiShifter.so / macOS: Contents/MacOS/HiFiShifter）
 *
 * 它只做宿主会做的事：dlopen → 调模块入口 → GetPluginFactory → 读工厂信息与类列表。
 * 刻意**不** import 本项目任何头文件或 Rust 代码：如果它还需要我们的代码才能通过，
 * 那它证明的就不是"宿主能加载"，而是"我们自己能加载自己"。
 *
 * 结构体布局来自 VST3 SDK 3.8.0 的 pluginterfaces/base/ipluginbase.h：
 *   PFactoryInfo { char8 vendor[64]; char8 url[256]; char8 email[128]; int32 flags; }
 *   PClassInfo   { TUID cid; int32 cardinality; char8 category[32]; char8 name[64]; }
 *   IPluginFactory = FUnknown{queryInterface, addRef, release}
 *                  + getFactoryInfo, countClasses, getClassInfo, createInstance
 */

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int32_t tresult;
typedef char char8;
typedef char8 TUID[16];

struct PFactoryInfo {
    char8 vendor[64];
    char8 url[256];
    char8 email[128];
    int32_t flags;
};

struct PClassInfo {
    TUID cid;
    int32_t cardinality;
    char8 category[32];
    char8 name[64];
};

/* IPluginFactory 的 vtable：前三个是 FUnknown 的三个方法。 */
typedef struct {
    tresult (*queryInterface)(void *self, const char *iid, void **out);
    uint32_t (*addRef)(void *self);
    uint32_t (*release)(void *self);
    tresult (*getFactoryInfo)(void *self, struct PFactoryInfo *info);
    int32_t (*countClasses)(void *self);
    tresult (*getClassInfo)(void *self, int32_t index, struct PClassInfo *info);
    tresult (*createInstance)(void *self, const char *cid, const char *iid, void **out);
} IPluginFactoryVtbl;

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "用法: probe <module path>\n");
        return 2;
    }

    void *module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (module == NULL) {
        fprintf(stderr, "dlopen 失败: %s\n", dlerror());
        return 1;
    }
    printf("dlopen: ok\n");

    /* 模块入口：Linux 用 ModuleEntry，macOS 用 bundleEntry，Windows 用 InitDll。
       宿主也是"有哪个调哪个"，这里照做 —— 入口返回 false 一律视为加载失败。 */
    static const char *const entries[] = {"ModuleEntry", "bundleEntry", "InitDll"};
    int entry_seen = 0;
    for (size_t i = 0; i < sizeof entries / sizeof entries[0]; i++) {
        void *symbol = dlsym(module, entries[i]);
        if (symbol == NULL) continue;
        bool ok;
        if (strcmp(entries[i], "InitDll") == 0) {
            bool (*fn)(void) = (bool (*)(void))symbol;
            ok = fn();
        } else {
            /* bundleEntry 的 CFBundleRef 参数本实现不读，传 NULL 即可。 */
            bool (*fn)(void *) = (bool (*)(void *))symbol;
            ok = fn(NULL);
        }
        if (!ok) {
            fprintf(stderr, "入口 %s 返回 false\n", entries[i]);
            return 1;
        }
        printf("entry %s: ok\n", entries[i]);
        entry_seen = 1;
    }
    if (!entry_seen) {
        printf("entry: 没有模块入口（可接受，部分宿主直接查 GetPluginFactory）\n");
    }

    void *(*get_plugin_factory)(void) =
        (void *(*)(void))dlsym(module, "GetPluginFactory");
    if (get_plugin_factory == NULL) {
        fprintf(stderr, "缺少 GetPluginFactory\n");
        return 1;
    }

    void *factory = get_plugin_factory();
    if (factory == NULL) {
        fprintf(stderr, "GetPluginFactory 返回 NULL\n");
        return 1;
    }
    printf("GetPluginFactory: ok\n");

    IPluginFactoryVtbl *vtbl = *(IPluginFactoryVtbl **)factory;

    struct PFactoryInfo info;
    memset(&info, 0, sizeof info);
    if (vtbl->getFactoryInfo(factory, &info) != 0) {
        fprintf(stderr, "getFactoryInfo 失败\n");
        return 1;
    }
    printf("vendor: %s\n", info.vendor);
    printf("url: %s\n", info.url);

    int32_t count = vtbl->countClasses(factory);
    printf("classes: %d\n", count);
    if (count < 1) {
        fprintf(stderr, "工厂没有导出任何类\n");
        return 1;
    }

    int matched = 0;
    for (int32_t i = 0; i < count; i++) {
        struct PClassInfo class_info;
        memset(&class_info, 0, sizeof class_info);
        if (vtbl->getClassInfo(factory, i, &class_info) != 0) {
            fprintf(stderr, "getClassInfo(%d) 失败\n", i);
            return 1;
        }
        printf("  class[%d]: %s / %s\n", i, class_info.category, class_info.name);
        if (strstr(class_info.name, "HiFiShifter") != NULL) matched = 1;
    }

    vtbl->release(factory);

    if (!matched) {
        fprintf(stderr, "类列表里没有 HiFiShifter\n");
        return 1;
    }
    printf("result: ok\n");
    return 0;
}