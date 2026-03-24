/**
 * sstretch-c.h — C wrapper for Signalsmith Stretch (MIT)
 *
 * 为 Rust FFI 提供纯 C 接口，封装 signalsmith::stretch::SignalsmithStretch<float>。
 * 支持离线批量拉伸和实时流式拉伸两种模式。
 */
#ifndef SSTRETCH_C_H
#define SSTRETCH_C_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 不透明句柄，指向内部的 SignalsmithStretch 实例。
 */
typedef void* SStretchState;

/**
 * 创建一个新的 stretcher 实例。
 *
 * @param sample_rate  采样率 (Hz)
 * @param channels     声道数 (1 或 2)
 * @return 句柄, 失败返回 NULL
 */
SStretchState sstretch_new(unsigned int sample_rate, unsigned int channels);

/**
 * 销毁 stretcher 实例并释放所有内存。
 */
void sstretch_delete(SStretchState state);

/**
 * 重置内部状态（清空缓冲区）。
 */
void sstretch_reset(SStretchState state);

/**
 * 设置音高偏移（半音）。0 = 不变调。
 */
void sstretch_set_transpose_semitones(SStretchState state, double semitones);

/**
 * 设置音高偏移（乘数）。1.0 = 不变调, 2.0 = 升八度。
 */
void sstretch_set_transpose_factor(SStretchState state, double factor);

/**
 * 获取输入延迟（帧数）。
 */
int sstretch_input_latency(SStretchState state);

/**
 * 获取输出延迟（帧数）。
 */
int sstretch_output_latency(SStretchState state);

/**
 * 流式实时处理：输入交错 PCM，输出交错 PCM。
 *
 * 时间拉伸比通过 in_frames 与 out_frames 的比值隐式控制：
 *   time_ratio ≈ out_frames / in_frames
 *
 * @param state              句柄
 * @param input_interleaved  输入交错 PCM (长度 = in_frames * channels)
 * @param in_frames          输入帧数
 * @param output_interleaved 输出交错 PCM (长度 = out_frames * channels, 由调用方分配)
 * @param out_frames         输出帧数
 * @return 0 成功, -1 失败
 */
int sstretch_process_interleaved(
    SStretchState state,
    const float* input_interleaved,
    unsigned int in_frames,
    float* output_interleaved,
    unsigned int out_frames
);

/**
 * 离线批量拉伸：将整段交错 PCM 按 time_ratio 拉伸。
 *
 * 内部自动处理延迟补偿、flush 等。
 *
 * @param state              句柄
 * @param input_interleaved  输入交错 PCM
 * @param in_frames          输入帧数
 * @param output_interleaved 输出缓冲区 (长度 = out_frames * channels, 由调用方分配)
 * @param out_frames         期望输出帧数
 * @param time_ratio         时间拉伸比 (output_duration / input_duration)
 * @return 实际输出帧数, -1 表示失败
 */
int sstretch_process_offline(
    SStretchState state,
    const float* input_interleaved,
    unsigned int in_frames,
    float* output_interleaved,
    unsigned int out_frames,
    double time_ratio
);

/**
 * 刷新残余输出（用于流式处理结束时）。
 *
 * @param state              句柄
 * @param output_interleaved 输出缓冲区
 * @param out_frames         期望输出帧数
 * @return 实际输出帧数
 */
int sstretch_flush(
    SStretchState state,
    float* output_interleaved,
    unsigned int out_frames
);

#ifdef __cplusplus
}
#endif

#endif /* SSTRETCH_C_H */
