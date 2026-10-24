/* host_harness.c —— 一个**最小的 VST3 宿主**，只做 REAPER 打开插件窗口时做的事：
 *
 *   createInstance(component) → initialize
 *   createInstance(controller) → initialize
 *   createView("editor") → isPlatformTypeSupported("X11EmbedWindowID")
 *   attached(<真实 X11 窗口>, "X11EmbedWindowID") → onSize → 等几秒 → removed
 *
 * 【为什么需要它】真 REAPER 无法在这里自动驱动，而"窗口是不是真的嵌进去了"只能由
 * 一个**真实宿主**回答。这个 harness 就是方案 §13 里那个 P0 探针的宿主侧：
 * 它把 X11 窗口交出去，剩下的看插件自己的日志（[gui] 那几行现在是 WARN 级）。
 *
 * 用法：GDK_BACKEND=x11 ./host_harness <HiFiShifter.so> [等待秒数]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <X11/Xlib.h>

typedef int32_t tresult;
typedef char FIDString[16];
#define K_RESULT_OK 0

typedef struct { int32_t left, top, right, bottom; } ViewRect;

/* FUnknown / 工厂 / 组件 / 控制器 / 视图的 IID（与 backend/.../vst3.rs 里的常量一致） */
/* GUID 布局：第 1 个字小端、第 2 个字拆成两个 u16 各自大端、第 3/4 个字大端。
 * 与 backend/hifishifter-plugin/src/vst3.rs 的 uid_guid() 逐字节一致。 */
static const unsigned char IID_COMPONENT[16]      = {0x31,0xFF,0x31,0xE8, 0xD5,0xF2, 0x01,0x43, 0x92,0x8E, 0xBB,0xEE, 0x25,0x69,0x78,0x02};
static const unsigned char IID_EDIT_CONTROLLER[16]= {0xE3,0xBB,0xD7,0xDC, 0x42,0x77, 0x8D,0x44, 0xA8,0x74, 0xAA,0xCC, 0x97,0x9C,0x75,0x9E};

/* 一个最小的 IHostApplication：只为让插件拿到宿主名（日志按宿主分文件）。
 * vtable: QI / addRef / release / getName / createInstance */
/* IHostApplication 的 IID：58E595CC-DB2D-4969-8BFA-C7CE7E8CAAC0 */
static const unsigned char IID_IHOST_APPLICATION[16] = {0xCC,0x95,0xE5,0x58, 0xDB,0x2D, 0x49,0x69, 0x8B,0xFA, 0xC7,0xCE,0x7E,0x8C,0xAA,0xC0};
static const unsigned char IID_FUNKNOWN_B[16]        = {0x00,0x00,0x00,0x00, 0x00,0x00, 0x00,0x00, 0xC0,0x00,0x00,0x00, 0x00,0x00,0x00,0x46};
static tresult ha_query(void *t, const char *iid, void **obj) {
    if (!memcmp(iid, IID_IHOST_APPLICATION, 16) || !memcmp(iid, IID_FUNKNOWN_B, 16)) {
        *obj = t;
        return 0;
    }
    *obj = NULL;
    return -2147467262; /* kNoInterface */
}
static uint32_t ha_add(void *t) { (void)t; return 2; }
static uint32_t ha_release(void *t) { (void)t; return 1; }
static tresult ha_get_name(void *t, char *name) { (void)t; snprintf(name, 128, "SyntheticHost"); return 0; }
static tresult ha_create(void *t, const char *cid, const char *iid, void **obj) {
    (void)t; (void)cid; (void)iid; *obj = NULL; return -2147467263; /* kNotImplemented */
}
static void *ha_vtbl[] = { (void *)ha_query, (void *)ha_add, (void *)ha_release,
                           (void *)ha_get_name, (void *)ha_create };
static void *ha_object[] = { (void *)ha_vtbl };

/* PClassInfo: cid[16] + cardinality + category[32] + name[64] */
typedef struct {
    unsigned char cid[16];
    int32_t cardinality;
    char category[32];
    char name[64];
} PClassInfo;

typedef struct {
    unsigned char cid[16];
    char name[64];
    char vendor[64];
    char url[256];
    char email[128];
} PFactoryInfo;

#define VT(obj) (*(void ***)(obj))
#define CALL(obj, slot, type, ...) ((type)(VT(obj)[slot]))(__VA_ARGS__)

typedef void *(*CreateInstanceFn)(void *, const char *, const char *, void **);
typedef tresult (*CountFn)(void *);
typedef tresult (*ClassInfoFn)(void *, int32_t, PClassInfo *);
typedef tresult (*FactoryInfoFn)(void *, PFactoryInfo *);
typedef tresult (*InitializeFn)(void *, void *);
typedef tresult (*ControllerCidFn)(void *, unsigned char *);
typedef void *(*CreateViewFn)(void *, const char *);
typedef tresult (*PlatformFn)(void *, const char *);
typedef tresult (*AttachedFn)(void *, void *, const char *);
typedef tresult (*SizeFn)(void *, ViewRect *);
typedef uint32_t (*ReleaseFn)(void *);
typedef tresult (*RemovedFn)(void *);

static void hex(const unsigned char *b) { for (int i = 0; i < 16; i++) printf("%02x", b[i]); }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) { printf("usage: host_harness <HiFiShifter.so> [seconds]\n"); return 2; }
    int seconds = argc > 2 ? atoi(argv[2]) : 8;

    void *module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!module) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    printf("dlopen: ok\n");

    tresult (*entry)(void *) = (tresult (*)(void *))dlsym(module, "ModuleEntry");
    if (entry) printf("ModuleEntry: %d\n", entry(module));

    void *(*get_factory)(void) = (void *(*)(void))dlsym(module, "GetPluginFactory");
    if (!get_factory) { printf("no GetPluginFactory\n"); return 1; }
    void *factory = get_factory();
    if (!factory) { printf("factory is null\n"); return 1; }
    printf("factory: %p\n", factory);

    PFactoryInfo fi; memset(&fi, 0, sizeof fi);
    CALL(factory, 3, FactoryInfoFn, factory, &fi);
    printf("vendor=%s\n", fi.vendor);
    int32_t count = CALL(factory, 4, CountFn, factory);
    printf("classes=%d\n", count);

    unsigned char component_cid[16] = {0};
    for (int32_t i = 0; i < count; i++) {
        PClassInfo ci; memset(&ci, 0, sizeof ci);
        if (CALL(factory, 5, ClassInfoFn, factory, i, &ci) != K_RESULT_OK) continue;
        printf("  class[%d] %s / %s cid=", i, ci.category, ci.name); hex(ci.cid); printf("\n");
        if (strstr(ci.category, "Audio Module")) memcpy(component_cid, ci.cid, 16);
    }
    if (!component_cid[0]) { printf("no Audio Module class\n"); return 1; }

    void *component = NULL;
    tresult r = ((CreateInstanceFn)VT(factory)[6])(factory, (const char *)component_cid,
                                                  (const char *)IID_COMPONENT, &component);
    printf("createInstance(component) -> %d obj=%p\n", r, component);
    if (r != K_RESULT_OK || !component) return 1;
    printf("component.initialize(host) -> %d\n", CALL(component, 3, InitializeFn, component, ha_object));

    unsigned char controller_cid[16] = {0};
    CALL(component, 5, ControllerCidFn, component, controller_cid);
    printf("controller cid="); hex(controller_cid); printf("\n");

    void *controller = NULL;
    r = ((CreateInstanceFn)VT(factory)[6])(factory, (const char *)controller_cid,
                                           (const char *)IID_EDIT_CONTROLLER, &controller);
    printf("createInstance(controller) -> %d obj=%p\n", r, controller);
    if (r != K_RESULT_OK || !controller) return 1;
    printf("controller.initialize(host) -> %d\n", CALL(controller, 3, InitializeFn, controller, ha_object));

    void *view = CALL(controller, 17, CreateViewFn, controller, "editor");
    printf("createView(\"editor\") -> %p\n", view);
    if (!view) return 1;

    printf("isPlatformTypeSupported(\"X11EmbedWindowID\") -> %d\n",
           CALL(view, 3, PlatformFn, view, "X11EmbedWindowID"));
    printf("isPlatformTypeSupported(\"WaylandSurfaceID\") -> %d\n",
           CALL(view, 3, PlatformFn, view, "WaylandSurfaceID"));

    /* 一个普通 X11 窗口 —— 这正是宿主（REAPER）交给插件的那种东西 */
    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) { printf("cannot open display\n"); return 1; }
    Window host = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, 900, 600, 0, 0, 0x101010);
    XStoreName(dpy, host, "HiFiShifter synthetic host");
    XMapWindow(dpy, host);
    XSync(dpy, False);
    printf("host X11 window: 0x%lx\n", (unsigned long)host);

    printf("attached -> %d\n", CALL(view, 4, AttachedFn, view, (void *)(unsigned long)host, "X11EmbedWindowID"));
    ViewRect rect = {0, 0, 900, 600};
    printf("onSize -> %d\n", CALL(view, 10, SizeFn, view, &rect));

    printf("pumping %d s (look at the plugin log for [gui] lines)\n", seconds);
    for (int i = 0; i < seconds * 10; i++) {
        while (XPending(dpy)) { XEvent e; XNextEvent(dpy, &e); }
        usleep(100000);
    }

    printf("removed -> %d\n", CALL(view, 5, RemovedFn, view));
    CALL(view, 2, ReleaseFn, view);
    CALL(controller, 2, ReleaseFn, controller);
    CALL(component, 2, ReleaseFn, component);
    printf("done\n");
    return 0;
}
