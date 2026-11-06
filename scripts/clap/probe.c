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

static unsigned char saved_state[4096];
static unsigned char first_state[4096];
static size_t saved_len = 0, loaded_len = 0;

static clap_host_t host = { {1,1,9}, NULL, "probe", "HiFiShifter", "https://example.invalid", "1",
    NULL, NULL, NULL, NULL };

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
    if (desc->features) {
        for (const char *const *feature = desc->features; *feature; ++feature) printf("feature=%s\n", *feature);
        /* 阶段 1 是 Capture-only：descriptor 绝不声明任一 ARA 能力，也绝不要求 ARA。
           宿主若看到 ara:required，会直接拒绝为这个插件建立实例。 */
        for (const char *const *feature = desc->features; *feature; ++feature) {
            if (strncmp(*feature, "ara:", 4) == 0) {
                fprintf(stderr, "unexpected ARA feature: %s\n", *feature);
                entry->deinit(); return 1;
            }
        }
    }
    /* 工厂只能报一个插件；get_plugin_descriptor(1) 必须返回 null，否则
       宿主扫描索引会越界。 */
    if (factory->get_plugin_descriptor(factory, 1) != NULL) {
        fprintf(stderr, "get_plugin_descriptor(1) should return null\n");
        entry->deinit(); return 1;
    }
    const clap_plugin_t *plugin = (const clap_plugin_t *)factory->create_plugin(
        factory, (const void *)&host, "com.hifishifter.clap");
    if (!plugin || !plugin->init || !plugin->destroy || !plugin->activate || !plugin->process) {
        fprintf(stderr, "create_plugin returned an incomplete plug-in\n"); return 1;
    }
    if (!plugin->init(plugin)) { fprintf(stderr, "plugin.init failed\n"); return 1; }
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
    /* 阶段 1 明确不返回 clap.gui：交一个假的嵌入视图比不提供 GUI 更危险。 */
    if (plugin->get_extension(plugin, "clap.gui")) {
        fprintf(stderr, "clap.gui must not be advertised by a Capture-only shell\n"); return 1;
    }
    if (plugin->stop_processing) plugin->stop_processing(plugin);
    plugin->deactivate(plugin);
    plugin->destroy(plugin);
    const void *ara = entry->get_factory("org.ara-audio.ara.factory/2");
    printf("ara_factory=%s\n", ara ? "present" : "absent");
    entry->deinit();
    return 0;
}
