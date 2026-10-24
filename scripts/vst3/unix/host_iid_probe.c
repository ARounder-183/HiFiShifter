/* host_iid_probe.c —— 判定"插件能否按**本平台**的 IID 布局拿到宿主的原生扩展"。
 *
 *   cc -O1 -o host_iid_probe host_iid_probe.c -ldl
 *   ./host_iid_probe <HiFiShifter.so> [工作目录]
 *
 * 【为什么需要它】VST3 的 IID 是**逐字节**比较的，而 INLINE_UID 按 COM_COMPATIBLE
 * 展开成两种布局：Windows 是 COM/GUID，Linux / macOS 是每字大端。插件若把
 * queryInterface 的 IID 编错，宿主的回答是"空指针"——不崩、不报错，日志里只剩一行
 * available=false。这类缺陷在 Windows 上完全看不出来，只能靠一条**在 Unix 上真的跑
 * 一遍握手**的判据挡住。
 *
 * 判据（四条全中才算过）：
 *   1. 日志里有 "[reaper-host] QI succeeded"
 *   2. 日志里有 "native host extension available=true"
 *   3. 日志里的 "iid layout:" 与本平台一致（Unix 必须是 big-endian）
 *   4. 日志里有 "host application (vst3 name): SynthHost"（证明 IHostApplication
 *      那一份引用真的拿到了 —— 编辑器与处理器的消息通道靠它）
 *
 * 【为什么自己实现宿主】用真 DAW 当判据就跑不进 CI。这里只实现插件握手时**真的会
 * 调用**的两个接口：IHostApplication（getName）与 IReaperHostApplication
 * （getReaperApi）。不建视图、不碰 X11，所以没有显示环境也能跑。
 *
 * 退出码：0 = 通过；1 = 插件日志不满足判据；2 = 用法/加载失败。
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

/* ── IID：四个 32 位字 + 平台布局 ─────────────────────────────────────────── */

typedef struct { uint32_t w[4]; } IidWords;

static const IidWords IID_HOST      = {{0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5}};
static const IidWords IID_REAPER_HOST= {{0x79655E36, 0x77EE4267, 0xA573FEF7, 0x4912C27C}};
static const IidWords IID_COMPONENT = {{0xE831FF31, 0xF2D54301, 0x928EBBEE, 0x25697802}};

/* FUnknown {00000000-0000-0000-C000-000000000046}：两种布局在这里恰好相同。 */
static const unsigned char IID_FUNKNOWN[16] = {
    0,0,0,0, 0,0,0,0, 0xC0,0,0,0, 0,0,0,0x46
};

/* 【与 vst3.rs::platform_iid 同构】Windows 走 COM 布局，其余平台每字大端。 */
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

/* ── 宿主对象 ─────────────────────────────────────────────────────────────── */

static uint32_t ref_add(void *self) { (void)self; return 2; }
static uint32_t ref_release(void *self) { (void)self; return 1; }

static const char *app_version(void) { return "7.99/synthetic"; }

/* IReaperHostApplication：getReaperApi / getReaperParent / extended。 */
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

/* IHostApplication：getName / createInstance；同时按 IID 交出上面的 REAPER 接口。 */
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

/* ── 插件入口 ─────────────────────────────────────────────────────────────── */

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

/* 读整个文件到内存（日志很小，一次读完最省事）。 */
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

static int failures = 0;
static void expect(const char *haystack, const char *needle, const char *what) {
    if (haystack && strstr(haystack, needle)) {
        printf("  ok   %s\n", what);
    } else {
        printf("  FAIL %s（日志里找不到 \"%s\"）\n", what, needle);
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
        mkdir(workdir, 0700); /* 已存在也没关系 */
    } else {
        snprintf(workdir, sizeof workdir, "/tmp/hfs-iid-XXXXXX");
        if (!mkdtemp(workdir)) {
            printf("mkdtemp failed\n");
            return 2;
        }
    }

    /* 【为什么用 LOG_DIR 而不是 PLUGIN_LOG】日志文件名会随宿主名改写；给一个**目录**
     * 让"按宿主分文件"照常发生，探针只认目录里那一个文件。显式路径还会关掉缓冲，
     * 于是握手行不会因为进程提前退出而丢在内存里。 */
    setenv("HIFISHIFTER_LOG_DIR", workdir, 1);
    setenv("HIFISHIFTER_LOG", "info", 1);

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

    /* setHostContext：VST3 3.7 起宿主会把上下文交出来；插件在这里就试着拿原生扩展。 */
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
    void *component = NULL;
    tresult result = ((CreateInstanceFn)VT(factory)[6])(
        factory, (const char *)component_cid, (const char *)component_iid, &component);
    if (result != OK || !component) { printf("createInstance failed: %d\n", result); return 2; }

    /* initialize 是插件拿宿主应用接口的地方：IHostApplication 与原生扩展都在这里绑。 */
    ((InitializeFn)VT(component)[3])(component, ha_object);

    /* 等日志落地（显式目录不缓冲，但宿主调用是异步的）。 */
    usleep(300 * 1000);

    CALL(component, 2, ReleaseFn, component);

    char log_path[1100];
    snprintf(log_path, sizeof log_path, "%s/plugin-unknown.log", workdir);
    char *log = slurp(log_path);
    if (!log) {
        printf("  FAIL 没有日志：%s\n", log_path);
        return 1;
    }

    printf("[vst3:iid] %s\n", module_path);
    expect(log, "[reaper-host] QI succeeded", "原生扩展握手成功");
    expect(log, "native host extension available=true", "原生扩展可用");
    expect(log, "host application (vst3 name): SynthHost", "宿主应用接口可用（getName）");
#if defined(_WIN32)
    expect(log, "iid layout: com-guid", "IID 布局与平台一致（com-guid）");
#else
    expect(log, "iid layout: big-endian", "IID 布局与平台一致（big-endian）");
#endif

    if (failures) {
        printf("\n---- 插件日志（尾部）----\n");
        size_t length = strlen(log);
        size_t from = length > 3000 ? length - 3000 : 0;
        printf("%s\n", log + from);
        printf("--------------------------\n");
    }
    free(log);
    return failures ? 1 : 0;
}
