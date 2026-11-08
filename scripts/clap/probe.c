/* CLAP black-box probe: dlopen -> entry -> factory -> instance -> process -> state. */
/*
 * C99 for Linux/macOS only. Windows uses scripts/clap/windows/verify.ps1.
 * The probe links no product code; it exercises the host-facing ABI.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint32_t major, minor, revision; } clap_version_t;
typedef struct { const void *(*get_plugin_count_placeholder)(void); } unused_t;
typedef struct {
    clap_version_t clap_version;
    bool (*init)(const char *);
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


typedef struct {
    const clap_plugin_descriptor_t *desc;
    void *plugin_data;
    bool (*init)(const void *);
    void (*destroy)(const void *);
    bool (*activate)(const void *, double, uint32_t, uint32_t);
    void (*deactivate)(const void *);
    bool (*start_processing)(const void *);
    void (*stop_processing)(const void *);
    void (*reset)(const void *);
    int (*process)(const void *, const void *);
    const void *(*get_extension)(const void *, const char *);
    void (*on_main_thread)(const void *);
} clap_plugin_t;

typedef struct { uint32_t major, minor, revision; } clap_version_t2;
typedef struct {
    clap_version_t clap_version; void *host_data; const char *name, *vendor, *url, *version;
    const void *(*get_extension)(const void *, const char *);
    void (*request_restart)(const void *);
    void (*request_process)(const void *);
    void (*request_callback)(const void *);
} clap_host_t;

typedef struct {
    float **data32; double **data64; uint32_t channel_count, latency; uint64_t constant_mask;
} clap_audio_buffer_t;
typedef struct {
    int64_t steady_time; uint32_t frames_count; const void *transport;
    const clap_audio_buffer_t *audio_inputs; clap_audio_buffer_t *audio_outputs;
    uint32_t audio_inputs_count, audio_outputs_count;
    const void *in_events, *out_events;
} clap_process_t;

/* ctx 必须排在函数指针前面：CLAP 的 stream 结构以 ctx 开头（clap/stream.h）。
   少了这个字段，插件会从错误的偏移读 write —— 症状是 state.save 永远返回 false，
   而"保存失败"与"插件没有 state 扩展"在宿主日志里长得一模一样。 */
typedef struct { void *ctx; int64_t (*write)(const void *, const void *, uint64_t); } clap_ostream_t;
typedef struct { void *ctx; int64_t (*read)(const void *, void *, uint64_t); } clap_istream_t;
typedef struct {
    bool (*save)(const void *, const void *);
    bool (*load)(const void *, const void *);
} clap_plugin_state_t;

/* ARA：entry 级工厂表与实例级扩展（ARACLAP.h 的 stable ID）。
   【为什么判据要在这里】宿主在扫描阶段就用 `get_plugin_id` 把这张表与 descriptor
   配对；两者只要差一个字符，插件永远不会被 ARA 认领，而宿主日志里不会有任何提示。 */
typedef struct {
    uint32_t (*get_factory_count)(const void *);
    const void *(*get_ara_factory)(const void *, uint32_t);
    const char *(*get_plugin_id)(const void *, uint32_t);
} clap_ara_factory_t;
typedef struct {
    const void *(*get_factory)(const void *);
    const void *(*bind_to_document_controller)(const void *, const void *, uint32_t, uint32_t);
} clap_ara_plugin_extension_t;

static unsigned char saved_state[4096];
static unsigned char first_state[4096];
static size_t saved_len = 0, loaded_len = 0;

/* 假宿主实现：clap.log 把插件日志收进下面的数组，用来证明 logger 真的接上了。 */
typedef struct { void (*log)(const void *, int32_t, const char *); } clap_host_log_t;
typedef struct { uint32_t (*get)(const void *); } clap_plugin_latency_t;
typedef struct { uint32_t (*get)(const void *); } clap_plugin_tail_t;
static char host_log[8192]; static size_t host_log_len = 0;
static void host_log_write(const void *host, int32_t severity, const char *msg) {
    (void)host; (void)severity;
    size_t n = strlen(msg);
    if (host_log_len + n + 1 >= sizeof(host_log)) return;
    memcpy(host_log + host_log_len, msg, n); host_log_len += n;
    host_log[host_log_len++] = '\n';
}
static clap_host_log_t host_log_ext = { host_log_write };
static const void *host_get_extension(const void *host, const char *id) {
    (void)host;
    if (strcmp(id, "clap.log") == 0) return &host_log_ext;
    return NULL;
}
static clap_host_t host = { {1,1,9}, NULL, "probe", "HiFiShifter", "https://example.invalid", "1",
    host_get_extension, NULL, NULL, NULL };

/* 每次只接受 7 字节：clap/stream.h 明确允许宿主限制单次读写的字节数，
   插件必须循环推进。用这个假宿主复现"短写丢尾部 / 短读把状态截断"。 */
#define STREAM_CHUNK 7
static int64_t stream_write(const void *stream, const void *buffer, uint64_t size) {
    (void)stream;
    uint64_t count = size < STREAM_CHUNK ? size : STREAM_CHUNK;
    if (saved_len + count > sizeof(saved_state)) return -1;
    memcpy(saved_state + saved_len, buffer, (size_t)count); saved_len += (size_t)count;
    return (int64_t)count;
}
static int64_t stream_read(const void *stream, void *buffer, uint64_t size) {
    (void)stream;
    size_t remaining = saved_len - loaded_len;
    uint64_t count = size < STREAM_CHUNK ? size : STREAM_CHUNK;
    if (count > remaining) count = remaining;
    if (count) memcpy(buffer, saved_state + loaded_len, (size_t)count);
    loaded_len += (size_t)count; return (int64_t)count;
}

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
    bool saw_ara_supported = false;
    if (desc->features) {
        for (const char *const *feature = desc->features; *feature; ++feature) {
            printf("feature=%s\n", *feature);
            if (strcmp(*feature, "ara:supported") == 0) saw_ara_supported = true;
        }
        /* 阶段 2：descriptor 必须声明 ara:supported（宿主据此知道可以从 ARA 入口插入），
           但**绝不**声明 ara:required —— 那会让不支持 ARA 的宿主直接拒绝建立实例。
           两种宿主都要能插上，这是本插件的基本前提。 */
        for (const char *const *feature = desc->features; *feature; ++feature) {
            if (strcmp(*feature, "ara:required") == 0) {
                fprintf(stderr, "the descriptor must not require ARA: %s\n", *feature);
                entry->deinit(); return 1;
            }
        }
    }
    if (!saw_ara_supported) {
        fprintf(stderr, "the descriptor does not advertise ara:supported\n");
        entry->deinit(); return 1;
    }
    /* 工厂只能报一个插件；get_plugin_descriptor(1) 必须返回 null，否则
       宿主扫描索引会越界。 */
    if (factory->get_plugin_descriptor(factory, 1) != NULL) {
        fprintf(stderr, "get_plugin_descriptor(1) should return null\n");
        entry->deinit(); return 1;
    }
    /* ARA 工厂表：与 descriptor 同索引、同 id、同一份二进制。 */
    const clap_ara_factory_t *ara_factory =
        (const clap_ara_factory_t *)entry->get_factory("org.ara-audio.ara.factory/2");
    if (!ara_factory || !ara_factory->get_factory_count || !ara_factory->get_ara_factory ||
        !ara_factory->get_plugin_id) {
        fprintf(stderr, "the entry publishes no ARA factory table\n"); entry->deinit(); return 1;
    }
    if (ara_factory->get_factory_count(ara_factory) != 1) {
        fprintf(stderr, "the ARA factory table must describe exactly one association\n");
        entry->deinit(); return 1;
    }
    const char *ara_plugin_id = ara_factory->get_plugin_id(ara_factory, 0);
    if (!ara_plugin_id || strcmp(ara_plugin_id, desc->id) != 0) {
        fprintf(stderr, "ARA plugin id mismatch: %s != %s\n",
            ara_plugin_id ? ara_plugin_id : "(null)", desc->id);
        entry->deinit(); return 1;
    }
    const void *ara_table = ara_factory->get_ara_factory(ara_factory, 0);
    if (!ara_table || ara_factory->get_ara_factory(ara_factory, 0) != ara_table) {
        fprintf(stderr, "the ARA factory pointer must be stable across queries\n");
        entry->deinit(); return 1;
    }
    printf("ara_factory=present plugin_id=%s\n", ara_plugin_id);

    const clap_plugin_t *plugin = (const clap_plugin_t *)factory->create_plugin(
        factory, (const void *)&host, "com.hifishifter.clap");
    if (!plugin || !plugin->init || !plugin->destroy || !plugin->activate || !plugin->process) {
        fprintf(stderr, "create_plugin returned an incomplete plug-in\n"); return 1;
    }
    if (!plugin->init(plugin)) { fprintf(stderr, "plugin.init failed\n"); return 1; }
    /* 实例级 ARA 扩展必须给出**同一张**工厂表的指针：宿主用它把实例绑到文档控制器。 */
    const clap_ara_plugin_extension_t *ara_ext = (const clap_ara_plugin_extension_t *)
        plugin->get_extension(plugin, "org.ara-audio.ara.pluginextension/2");
    if (!ara_ext || !ara_ext->get_factory || !ara_ext->bind_to_document_controller) {
        fprintf(stderr, "the instance exposes no ARA plugin extension\n");
        plugin->destroy(plugin); entry->deinit(); return 1;
    }
    if (ara_ext->get_factory(plugin) != ara_table) {
        fprintf(stderr, "the instance ARA factory differs from the entry table\n");
        plugin->destroy(plugin); entry->deinit(); return 1;
    }
    /* REAPER 6.71+ 用 2022 私有草案 ID 做实例级发现，但那条草案的宿主实现在需要
       真正渲染时会崩在自身代码里（见 hifishifter_clap::ara 模块文档）。因此这里断言
       的是**必须缺席**：应答它等于把宿主的空指针缺陷引到用户面前。 */
    const clap_ara_plugin_extension_t *reaper_ext = (const clap_ara_plugin_extension_t *)
        plugin->get_extension(plugin, "com.celemony.ara.plugin_extension.draft/1");
    if (reaper_ext != NULL) {
        fprintf(stderr, "the REAPER private draft ARA extension must not be answered\n");
        plugin->destroy(plugin); entry->deinit(); return 1;
    }
    printf("ara_plugin_extension=present ara_reaper_legacy=absent\n");
    if (!plugin->activate(plugin, 48000.0, 1, 512)) { fprintf(stderr, "activate failed\n"); return 1; }
    if (plugin->start_processing && !plugin->start_processing(plugin)) { fprintf(stderr, "start_processing failed\n"); return 1; }
    float left[8] = {1,2,3,4,5,6,7,8}, right[8] = {8,7,6,5,4,3,2,1};
    float out_left[8] = {0}, out_right[8] = {0};
    float *in_planes[2] = {left, right}, *out_planes[2] = {out_left, out_right};
    clap_audio_buffer_t input = {in_planes, NULL, 2, 0, 0};
    clap_audio_buffer_t output = {out_planes, NULL, 2, 0, 0};
    clap_process_t process = {0, 8, NULL, &input, &output, 1, 1, NULL, NULL};
    if (plugin->process(plugin, &process) != 1) { fprintf(stderr, "process failed\n"); return 1; }
    for (int i = 0; i < 8; ++i) if (out_left[i] != left[i] || out_right[i] != right[i]) {
        fprintf(stderr, "process output mismatch at %d\n", i); return 1;
    }
    printf("process=ok state_extension=");
    const clap_plugin_state_t *state_ext =
        (const clap_plugin_state_t *)plugin->get_extension(plugin, "clap.state");
    /* 【为什么必须要求它存在】打印 "absent" 再返回 0 等于把"没有 state 扩展"
       当成合格产物：宿主保存工程时不会写任何字节，重开后编辑全丢，而自检
       依然是绿的。这一条判据把"能力缺席"变成硬失败。 */
    if (!state_ext || !state_ext->save || !state_ext->load) {
        fprintf(stderr, "clap.state is missing or incomplete\n"); return 1;
    }
    {
        clap_ostream_t ostream = {NULL, stream_write}; clap_istream_t istream = {NULL, stream_read};
        if (!state_ext->save(plugin, &ostream) || saved_len == 0) { fprintf(stderr, "state.save failed\n"); return 1; }
        size_t first_len = saved_len;
        /* 同一实例连续保存必须给出同一份字节：宿主会反复写工程。
           先把第一次结果拷出来，再让第二次写回同一个缓冲区做逐字比较。 */
        memcpy(first_state, saved_state, first_len);
        saved_len = 0;
        if (!state_ext->save(plugin, &ostream) || saved_len != first_len ||
            memcmp(first_state, saved_state, first_len) != 0) {
            fprintf(stderr, "state.save is not deterministic\n"); return 1;
        }
        loaded_len = 0;
        if (!state_ext->load(plugin, &istream) || loaded_len != saved_len) { fprintf(stderr, "state.load failed\n"); return 1; }
        printf("round-trip(%zu bytes)\n", saved_len);
    }
    /* 其余扩展必须如实返回：未接线的扩展返回 null 是允许的，但声明过的
       render / audio-ports / timer-support 三类宿主能力不能凭空消失。 */
    printf("render_extension=%s\n", plugin->get_extension(plugin, "clap.render") ? "present" : "absent");
    printf("audio_ports_extension=%s\n", plugin->get_extension(plugin, "clap.audio-ports") ? "present" : "absent");
    printf("timer_support_extension=%s\n", plugin->get_extension(plugin, "clap.timer-support") ? "present" : "absent");
    printf("gui_extension=%s\n", plugin->get_extension(plugin, "clap.gui") ? "present" : "absent");
    const clap_plugin_latency_t *latency = (const clap_plugin_latency_t *)plugin->get_extension(plugin, "clap.latency");
    const clap_plugin_tail_t *tail = (const clap_plugin_tail_t *)plugin->get_extension(plugin, "clap.tail");
    printf("latency_extension=%s latency=%u\n", latency && latency->get ? "present" : "absent",
        latency && latency->get ? latency->get(plugin) : 999);
    printf("tail_extension=%s tail=%u\n", tail && tail->get ? "present" : "absent",
        tail && tail->get ? tail->get(plugin) : 999);
    if (!latency || !latency->get || latency->get(plugin) != 0) { fprintf(stderr, "clap.latency must report 0\n"); return 1; }
    if (!tail || !tail->get || tail->get(plugin) != 0) { fprintf(stderr, "clap.tail must report 0\n"); return 1; }
    printf("host_log_bytes=%zu\n", host_log_len);
    if (host_log_len == 0) { fprintf(stderr, "clap.log was never delivered to the host\n"); return 1; }
    /* clap.gui 必须存在：宿主靠它建窗口、嵌前端。阶段 1 曾把它整个拿掉，
       结果是 REAPER 扫描到插件却只给出一块空白窗口。 */
    if (!plugin->get_extension(plugin, "clap.gui")) {
        fprintf(stderr, "clap.gui extension is missing; the host cannot show the editor\n"); return 1;
    }
    if (plugin->stop_processing) plugin->stop_processing(plugin);
    plugin->deactivate(plugin);
    plugin->destroy(plugin);

    entry->deinit();
    return 0;
}
