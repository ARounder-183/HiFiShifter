/*
 * HiFiShifter VST3 — black-box load probe.
 *
 * Usage: probe <path/to/module>          (Linux: HiFiShifter.so / macOS: Contents/MacOS/HiFiShifter)
 *
 * It does only what a host would do: dlopen → call the module entry → GetPluginFactory →
 * read factory info and the class list. It deliberately does **not** import any header or
 * Rust code from this project: if it needed our code to pass, it would prove "we can load
 * ourselves", not "a host can load it".
 *
 * Struct layouts come from VST3 SDK 3.8.0's pluginterfaces/base/ipluginbase.h:
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

/* IPluginFactory vtable: the first three slots are FUnknown's three methods. */
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
        fprintf(stderr, "usage: probe <module path>\n");
        return 2;
    }

    void *module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (module == NULL) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    printf("dlopen: ok\n");

    /* Module entry: Linux uses ModuleEntry, macOS bundleEntry, Windows InitDll.
       A host also calls whichever exists; we do the same — an entry returning false is
       always treated as a load failure. */
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
            /* This implementation never reads bundleEntry's CFBundleRef argument; NULL is fine. */
            bool (*fn)(void *) = (bool (*)(void *))symbol;
            ok = fn(NULL);
        }
        if (!ok) {
            fprintf(stderr, "entry %s returned false\n", entries[i]);
            return 1;
        }
        printf("entry %s: ok\n", entries[i]);
        entry_seen = 1;
    }
    if (!entry_seen) {
        printf("entry: no module entry (acceptable; some hosts go straight to GetPluginFactory)\n");
    }

    void *(*get_plugin_factory)(void) =
        (void *(*)(void))dlsym(module, "GetPluginFactory");
    if (get_plugin_factory == NULL) {
        fprintf(stderr, "missing GetPluginFactory\n");
        return 1;
    }

    void *factory = get_plugin_factory();
    if (factory == NULL) {
        fprintf(stderr, "GetPluginFactory returned NULL\n");
        return 1;
    }
    printf("GetPluginFactory: ok\n");

    IPluginFactoryVtbl *vtbl = *(IPluginFactoryVtbl **)factory;

    struct PFactoryInfo info;
    memset(&info, 0, sizeof info);
    if (vtbl->getFactoryInfo(factory, &info) != 0) {
        fprintf(stderr, "getFactoryInfo failed\n");
        return 1;
    }
    printf("vendor: %s\n", info.vendor);
    printf("url: %s\n", info.url);

    int32_t count = vtbl->countClasses(factory);
    printf("classes: %d\n", count);
    if (count < 1) {
        fprintf(stderr, "the factory exports no classes\n");
        return 1;
    }

    int matched = 0;
    for (int32_t i = 0; i < count; i++) {
        struct PClassInfo class_info;
        memset(&class_info, 0, sizeof class_info);
        if (vtbl->getClassInfo(factory, i, &class_info) != 0) {
            fprintf(stderr, "getClassInfo(%d) failed\n", i);
            return 1;
        }
        printf("  class[%d]: %s / %s\n", i, class_info.category, class_info.name);
        if (strstr(class_info.name, "HiFiShifter") != NULL) matched = 1;
    }

    vtbl->release(factory);

    if (!matched) {
        fprintf(stderr, "HiFiShifter not found in the class list\n");
        return 1;
    }
    printf("result: ok\n");
    return 0;
}