/* CLAP 黑盒加载探针：dlopen -> clap_entry -> init -> plugin-factory -> descriptor. */
/*
 * 只依赖宿主 ABI，不链接产品代码；它验证的是「宿主真的能找到并读这个模块」。
 * C99。Linux/macOS：cc -o probe probe.c -ldl；Windows 对应 verify.ps1 用 LoadLibrary。
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint32_t major, minor, revision; } clap_version_t;
typedef struct { const void *(*get_plugin_count_placeholder)(void); } unused_t;
typedef struct {
    clap_version_t clap_version;
    int (*init)(const char *);
    void (*deinit)(void);
    const void *(*get_factory)(const char *);
} clap_plugin_entry_t;
typedef struct {
    clap_version_t clap_version; const char *id, *name, *vendor, *url, *manual_url,
        *support_url, *version, *description; const char *const *features;
} clap_plugin_descriptor_t;
typedef struct {
    uint32_t (*get_plugin_count)(const void *);
    const clap_plugin_descriptor_t *(*get_plugin_descriptor)(const void *, uint32_t);
    const void *(*create_plugin)(const void *, const void *, const char *);
} clap_plugin_factory_t;

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <path-to-.clap>\n", argv[0]); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    const clap_plugin_entry_t *entry = (const clap_plugin_entry_t *)dlsym(lib, "clap_entry");
    if (!entry) { fprintf(stderr, "missing clap_entry\n"); return 1; }
    printf("clap_version=%u.%u.%u\n", entry->clap_version.major, entry->clap_version.minor, entry->clap_version.revision);
    if (entry->clap_version.major < 1 || !entry->init || !entry->deinit || !entry->get_factory) {
        fprintf(stderr, "invalid clap_entry callbacks\n"); return 1;
    }
    if (!entry->init(argv[1])) { fprintf(stderr, "clap_entry.init failed\n"); return 1; }
    const clap_plugin_factory_t *factory =
        (const clap_plugin_factory_t *)entry->get_factory("clap.plugin-factory");
    if (!factory || !factory->get_plugin_count || !factory->get_plugin_descriptor) {
        fprintf(stderr, "clap.plugin-factory unavailable\n"); entry->deinit(); return 1;
    }
    uint32_t count = factory->get_plugin_count(factory);
    printf("plugin_count=%u\n", count);
    if (count != 1) { fprintf(stderr, "expected exactly one descriptor\n"); entry->deinit(); return 1; }
    const clap_plugin_descriptor_t *desc = factory->get_plugin_descriptor(factory, 0);
    if (!desc || !desc->id || !desc->name || strcmp(desc->id, "com.hifishifter.clap")) {
        fprintf(stderr, "invalid descriptor\n"); entry->deinit(); return 1;
    }
    printf("plugin_id=%s plugin_name=%s version=%s\n", desc->id, desc->name, desc->version);
    if (desc->features) {
        for (const char *const *feature = desc->features; *feature; ++feature) printf("feature=%s\n", *feature);
    }
    const void *ara = entry->get_factory("org.ara-audio.ara.factory/2");
    printf("ara_factory=%s\n", ara ? "present" : "absent");
    entry->deinit();
    return 0;
}
