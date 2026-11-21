/* host_iid_probe.c — decides "can the plugin obtain the host's native extension using the
 * **this-platform** IID layout".
 *
 *   cc -O1 -o host_iid_probe host_iid_probe.c -ldl
 *   ./host_iid_probe <HiFiShifter.so> [workdir]
 *
 * [Why this exists] VST3 IIDs are compared **byte for byte**, and INLINE_UID expands into
 * two layouts depending on COM_COMPATIBLE: Windows uses COM/GUID, Linux / macOS use
 * big-endian per word. If the plugin encodes queryInterface's IID wrong, the host's answer
 * is "null pointer" — no crash, no error, just a single available=false line in the log.
 * Such a defect is completely invisible on Windows; only a criterion that **really runs the
 * handshake on Unix** can catch it.
 *
 * Criteria (all four must hit):
 *   1. the log contains "[reaper-host] QI succeeded"
 *   2. the log contains "native host extension available=true"
 *   3. the log's "iid layout:" matches this platform (Unix must be big-endian)
 *   4. the log contains "host application (vst3 name): SynthHost" (proof that the
 *      IHostApplication reference was really obtained — the editor/processor message
 *      channel depends on it)
 *
 * [Why the host is implemented here] using a real DAW as the criterion would never run in
 * CI. Only the two interfaces the plugin **actually calls** during the handshake are
 * implemented: IHostApplication (getName) and IReaperHostApplication (getReaperApi).
 * No view is built and X11 is not touched, so it runs without a display server.
 *
 * Exit codes: 0 = pass; 1 = plugin log does not satisfy the criteria; 2 = usage/load failure.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>

typedef int32_t tresult;
#define OK 0
#define NO_INTERFACE (-2147467262)
#define NOT_IMPLEMENTED (-2147467263)

/* ── IIDs: four 32-bit words + platform layout ───────────────────────────── */

typedef struct { uint32_t w[4]; } IidWords;

static const IidWords IID_HOST      = {{0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5}};
static const IidWords IID_REAPER_HOST= {{0x79655E36, 0x77EE4267, 0xA573FEF7, 0x4912C27C}};
static const IidWords IID_COMPONENT = {{0xE831FF31, 0xF2D54301, 0x928EBBEE, 0x25697802}};

/* FUnknown {00000000-0000-0000-C000-000000000046}: both layouts happen to coincide here. */
static const unsigned char IID_FUNKNOWN[16] = {
    0,0,0,0, 0,0,0,0, 0xC0,0,0,0, 0,0,0,0x46
};

/* [Isomorphic to vst3.rs::platform_iid] Windows uses the COM layout, other platforms big-endian per word. */
static void platform_iid(IidWords words, unsigned char out[16]) {
#if defined(_WIN32)
    uint32_t l1 = words.w[0], l2 = words.w[1], l3 = words.w[2], l4 = words.w[3];
    out[0]  = (unsigned char)(l1 & 0xFF);         out[1]  = (unsigned char)((l1 >> 8) & 0xFF);
    out[2]  = (unsigned char)((l1 >> 16) & 0xFF); out[3]  = (unsigned char)((l1 >> 24) & 0xFF);
    out[4]  = (unsigned char)((l2 >> 16) & 0xFF); out[5]  = (unsigned char)((l2 >> 24) & 0xFF);
    out[6]  = (unsigned char)(l2 & 0xFF);         out[7]  = (unsigned char)((l2 >> 8) & 0xFF);
    out[8]  = (unsigned char)((l3 >> 24) & 0xFF); out[9]  = (unsigned char)((l3 >> 16) & 0xFF);
    out[10] = (unsigned char)((l3 >> 8) & 0xFF);  out[11] = (unsigned char)(l3 & 0xFF);
    out[12] = (unsigned char)((l4 >> 24) & 0xFF); out[13] = (unsigned char)((l4 >> 16) & 0xFF);
    out[14] = (unsigned char)((l4 >> 8) & 0xFF);  out[15] = (unsigned char)(l4 & 0xFF);
#else
    for (int i = 0; i < 4; i++) {
        out[i * 4 + 0] = (unsigned char)((words.w[i] >> 24) & 0xFF);
        out[i * 4 + 1] = (unsigned char)((words.w[i] >> 16) & 0xFF);
        out[i * 4 + 2] = (unsigned char)((words.w[i] >> 8) & 0xFF);
        out[i * 4 + 3] = (unsigned char)(words.w[i] & 0xFF);
    }
#endif
}

/* ── Host objects ────────────────────────────────────────────────────────── */

static uint32_t ref_add(void *self) { (void)self; return 2; }
static uint32_t ref_release(void *self) { (void)self; return 1; }

static const char *app_version(void) { return "7.99/synthetic"; }

/* IReaperHostApplication: getReaperApi / getReaperParent / extended. */
static void *rh_api(void *self, const char *name) {
    (void)self;
    if (name && strcmp(name, "GetAppVersion") == 0) return (void *)&app_version;
    return NULL;
}
static void *rh_parent(void *self, uint32_t which) { (void)self; (void)which; return NULL; }
static void *rh_extended(void *self, uint32_t call, void *a, void *b, void *c) {
    (void)self; (void)call; (void)a; (void)b; (void)c; return NULL;
}
static tresult rh_query(void *self, const char *iid, void **out) {
    unsigned char want[16];
    platform_iid(IID_REAPER_HOST, want);
    if (!memcmp(iid, IID_FUNKNOWN, 16) || !memcmp(iid, want, 16)) {
        *out = self;
        return OK;
    }
    *out = NULL;
    return NO_INTERFACE;
}
static void *rh_vtbl[] = {
    (void *)rh_query, (void *)ref_add, (void *)ref_release,
    (void *)rh_api, (void *)rh_parent, (void *)rh_extended,
};
static void *rh_object[1] = { (void *)rh_vtbl };

/* IHostApplication: getName / createInstance; also hands out the REAPER interface above by IID. */
static tresult ha_get_name(void *self, char *name) {
    (void)self;
    snprintf(name, 128, "SynthHost");
    return OK;
}
static tresult ha_create(void *self, const char *cid, const char *iid, void **out) {
    (void)self; (void)cid; (void)iid;
    *out = NULL;
    return NOT_IMPLEMENTED;
}
static tresult ha_query(void *self, const char *iid, void **out) {
    unsigned char host_iid[16], reaper_iid[16];
    platform_iid(IID_HOST, host_iid);
    platform_iid(IID_REAPER_HOST, reaper_iid);
    if (!memcmp(iid, IID_FUNKNOWN, 16) || !memcmp(iid, host_iid, 16)) {
        *out = self;
        return OK;
    }
    if (!memcmp(iid, reaper_iid, 16)) {
        *out = rh_object;
        return OK;
    }
    *out = NULL;
    return NO_INTERFACE;
}
static void *ha_vtbl[] = {
    (void *)ha_query, (void *)ref_add, (void *)ref_release,
    (void *)ha_get_name, (void *)ha_create,
};
static void *ha_object[1] = { (void *)ha_vtbl };

/* ── Plugin entry ────────────────────────────────────────────────────────── */

#define VT(obj) (*(void ***)(obj))
#define CALL(obj, slot, type, ...) ((type)(VT(obj)[slot]))(__VA_ARGS__)

typedef tresult (*CreateInstanceFn)(void *, const char *, const char *, void **);
typedef tresult (*CountFn)(void *);
typedef tresult (*ClassInfoFn)(void *, int32_t, void *);
typedef tresult (*SetHostContextFn)(void *, void *);
typedef tresult (*InitializeFn)(void *, void *);
typedef uint32_t (*ReleaseFn)(void *);
typedef struct {
    unsigned char cid[16];
    int32_t cardinality;
    char category[32];
    char name[64];
} PClassInfo;

/* Read an entire file into memory (logs are small; one read is the simplest). */
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) { fclose(f); return NULL; }
    char *buffer = (char *)calloc((size_t)size + 1, 1);
    if (buffer) {
        size_t read = fread(buffer, 1, (size_t)size, f);
        buffer[read] = '\0';
    }
    fclose(f);
    return buffer;
}

static void hex16(const unsigned char *bytes, char out[33]) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[i * 2] = digits[(bytes[i] >> 4) & 0xF];
        out[i * 2 + 1] = digits[bytes[i] & 0xF];
    }
    out[32] = 0;
}

/* Print the tail of the plugin's own log.
 *
 * [Why it is printed even when createInstance fails] "the factory answered kNoInterface" has
 * two entirely different causes: it does not recognize the IID, or it recognizes it but
 * cannot build the object (Processor::create() returns the same code when ARA runtime
 * initialization fails). The return value alone cannot tell them apart, but the plugin
 * log's "factory createInstance cid=... iid=..." line and what follows separates them
 * directly.
 */
static void dump_log(const char *path) {
    char *log = slurp(path);
    if (!log) {
        printf("  (no plugin log: %s)\n", path);
        return;
    }
    size_t length = strlen(log);
    size_t from = length > 3000 ? length - 3000 : 0;
    printf("\n---- plugin log (tail) ----\n%s\n--------------------------\n", log + from);
    free(log);
}

static int failures = 0;
static void expect(const char *haystack, const char *needle, const char *what) {
    if (haystack && strstr(haystack, needle)) {
        printf("  ok   %s\n", what);
    } else {
        printf("  FAIL %s (log does not contain \"%s\")\n", what, needle);
        failures++;
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        printf("usage: host_iid_probe <HiFiShifter.so> [workdir]\n");
        return 2;
    }
    const char *module_path = argv[1];
    char workdir[512];
    if (argc >= 3) {
        snprintf(workdir, sizeof workdir, "%s", argv[2]);
        mkdir(workdir, 0700); /* already existing is fine */
    } else {
        snprintf(workdir, sizeof workdir, "/tmp/hfs-iid-XXXXXX");
        if (!mkdtemp(workdir)) {
            printf("mkdtemp failed\n");
            return 2;
        }
    }

    /* [Why an exact PLUGIN_LOG file] the log file name's shape is plugin-<format>-<host>.log
     * and changes with format and host; the probe only cares "does this one file contain
     * the handshake lines" and should not chase a naming scheme. An explicit file path also
     * disables in-memory buffering, so handshake lines are not lost in memory when the
     * process exits early. */
    char probe_log_path[1100];
    snprintf(probe_log_path, sizeof probe_log_path, "%s/plugin-iid-probe.log", workdir);
    setenv("HIFISHIFTER_PLUGIN_LOG", probe_log_path, 1);
    setenv("HIFISHIFTER_LOG_VERBOSITY", "medium", 1);

    void *module = dlopen(module_path, RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        printf("dlopen failed: %s\n", dlerror());
        return 2;
    }
    bool (*module_entry)(void *) = (bool (*)(void *))dlsym(module, "ModuleEntry");
    if (module_entry) module_entry(module);

    void *(*get_factory)(void) = (void *(*)(void))dlsym(module, "GetPluginFactory");
    if (!get_factory) { printf("no GetPluginFactory\n"); return 2; }
    void *factory = get_factory();
    if (!factory) { printf("factory is null\n"); return 2; }

    /* setHostContext: since VST3 3.7 the host hands over the context; the plugin already
     * tries to grab the native extension here. */
    ((SetHostContextFn)VT(factory)[9])(factory, ha_object);

    int32_t count = CALL(factory, 4, CountFn, factory);
    unsigned char component_cid[16] = {0};
    for (int32_t index = 0; index < count; index++) {
        PClassInfo info;
        memset(&info, 0, sizeof info);
        if (CALL(factory, 5, ClassInfoFn, factory, index, &info) != OK) continue;
        if (strstr(info.category, "Audio Module")) memcpy(component_cid, info.cid, 16);
    }
    if (!component_cid[0]) { printf("no Audio Module class\n"); return 2; }

    unsigned char component_iid[16];
    platform_iid(IID_COMPONENT, component_iid);
    char log_path[1100];
    snprintf(log_path, sizeof log_path, "%s", probe_log_path);
    void *component = NULL;
    tresult result = ((CreateInstanceFn)VT(factory)[6])(
        factory, (const char *)component_cid, (const char *)component_iid, &component);
    if (result != OK || !component) {
        char cid_hex[33], iid_hex[33];
        hex16(component_cid, cid_hex);
        hex16(component_iid, iid_hex);
        printf("createInstance failed: %d\n  cid=%s\n  iid=%s (this platform's layout)\n", result, cid_hex, iid_hex);
        dump_log(log_path);
        return 1;
    }

    /* initialize is where the plugin grabs the host application interfaces: IHostApplication
     * and the native extension are both bound here. */
    ((InitializeFn)VT(component)[3])(component, ha_object);

    /* Wait for the log to land (the explicit directory is unbuffered, but host calls are async). */
    usleep(300 * 1000);

    CALL(component, 2, ReleaseFn, component);

    char *log = slurp(log_path);
    if (!log) {
        printf("  FAIL no log: %s\n", log_path);
        return 1;
    }

    printf("[vst3:iid] %s\n", module_path);
    expect(log, "[reaper-host] QI succeeded", "native extension handshake succeeded");
    expect(log, "native host extension available=true", "native extension available");
    expect(log, "host application (vst3 name): SynthHost", "host application interface usable (getName)");
#if defined(_WIN32)
    expect(log, "iid layout: com-guid", "IID layout matches the platform (com-guid)");
#else
    expect(log, "iid layout: big-endian", "IID layout matches the platform (big-endian)");
#endif

    if (failures) {
        printf("\n---- plugin log (tail) ----\n");
        size_t length = strlen(log);
        size_t from = length > 3000 ? length - 3000 : 0;
        printf("%s\n", log + from);
        printf("--------------------------\n");
    }
    free(log);
    return failures ? 1 : 0;
}
