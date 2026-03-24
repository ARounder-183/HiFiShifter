/**
 * sstretch-c.cpp — C wrapper 实现
 *
 * 将 signalsmith::stretch::SignalsmithStretch<float> 的 C++ API
 * 通过 C 函数暴露给 Rust FFI。
 */
#include "sstretch-c.h"
#include "signalsmith-stretch/signalsmith-stretch.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

/// 内部状态结构，持有 stretcher 实例和配置。
struct SStretchInternal {
    signalsmith::stretch::SignalsmithStretch<float> stretch;
    int channels;
    unsigned int sample_rate;

    // 通道主序的临时缓冲区（避免每次分配）
    std::vector<std::vector<float>> in_ch;
    std::vector<std::vector<float>> out_ch;

    // 指针数组，用于传递给 process()
    std::vector<float*> in_ptrs;
    std::vector<float*> out_ptrs;
    std::vector<const float*> in_ptrs_const;

    SStretchInternal(unsigned int sr, int ch)
        : channels(ch)
        , sample_rate(sr)
        , in_ch(ch)
        , out_ch(ch)
        , in_ptrs(ch)
        , out_ptrs(ch)
        , in_ptrs_const(ch)
    {}

    /// 确保通道缓冲区至少有 n 帧
    void ensure_in(int n) {
        for (int c = 0; c < channels; ++c) {
            if ((int)in_ch[c].size() < n) in_ch[c].resize(n, 0.0f);
        }
    }
    void ensure_out(int n) {
        for (int c = 0; c < channels; ++c) {
            if ((int)out_ch[c].size() < n) out_ch[c].resize(n, 0.0f);
        }
    }

    /// 反交错：交错 PCM → 通道主序
    void deinterleave(const float* interleaved, int frames) {
        ensure_in(frames);
        for (int f = 0; f < frames; ++f) {
            for (int c = 0; c < channels; ++c) {
                in_ch[c][f] = interleaved[f * channels + c];
            }
        }
        for (int c = 0; c < channels; ++c) {
            in_ptrs[c] = in_ch[c].data();
            in_ptrs_const[c] = in_ch[c].data();
        }
    }

    /// 交错：通道主序 → 交错 PCM
    void interleave(float* interleaved, int frames) {
        for (int f = 0; f < frames; ++f) {
            for (int c = 0; c < channels; ++c) {
                interleaved[f * channels + c] = out_ch[c][f];
            }
        }
    }

    /// 准备输出指针
    void prepare_out_ptrs(int frames) {
        ensure_out(frames);
        for (int c = 0; c < channels; ++c) {
            out_ptrs[c] = out_ch[c].data();
        }
    }
};

extern "C" {

SStretchState sstretch_new(unsigned int sample_rate, unsigned int channels) {
    if (channels == 0 || channels > 2 || sample_rate == 0) return nullptr;

    auto* s = new (std::nothrow) SStretchInternal(sample_rate, (int)channels);
    if (!s) return nullptr;

    // 使用默认预设配置
    s->stretch.presetDefault((int)channels, (float)sample_rate);
    return (SStretchState)s;
}

void sstretch_delete(SStretchState state) {
    if (!state) return;
    delete (SStretchInternal*)state;
}

void sstretch_reset(SStretchState state) {
    if (!state) return;
    auto* s = (SStretchInternal*)state;
    s->stretch.reset();
}

void sstretch_set_transpose_semitones(SStretchState state, double semitones) {
    if (!state) return;
    auto* s = (SStretchInternal*)state;
    s->stretch.setTransposeSemitones((float)semitones);
}

void sstretch_set_transpose_factor(SStretchState state, double factor) {
    if (!state) return;
    auto* s = (SStretchInternal*)state;
    s->stretch.setTransposeFactor((float)factor);
}

int sstretch_input_latency(SStretchState state) {
    if (!state) return 0;
    auto* s = (SStretchInternal*)state;
    return s->stretch.inputLatency();
}

int sstretch_output_latency(SStretchState state) {
    if (!state) return 0;
    auto* s = (SStretchInternal*)state;
    return s->stretch.outputLatency();
}

int sstretch_process_interleaved(
    SStretchState state,
    const float* input_interleaved,
    unsigned int in_frames,
    float* output_interleaved,
    unsigned int out_frames
) {
    if (!state || !input_interleaved || !output_interleaved) return -1;
    if (in_frames == 0 && out_frames == 0) return 0;

    auto* s = (SStretchInternal*)state;

    // 反交错输入
    s->deinterleave(input_interleaved, (int)in_frames);

    // 准备输出缓冲
    s->prepare_out_ptrs((int)out_frames);

    // 调用 stretch.process()
    // 第一个参数是 channel-major 输入（const float*[]），
    // 第三个参数是 channel-major 输出（float*[]）
    s->stretch.process(
        s->in_ptrs_const.data(),
        (int)in_frames,
        s->out_ptrs.data(),
        (int)out_frames
    );

    // 交错输出
    s->interleave(output_interleaved, (int)out_frames);

    return 0;
}

int sstretch_process_offline(
    SStretchState state,
    const float* input_interleaved,
    unsigned int in_frames,
    float* output_interleaved,
    unsigned int out_frames,
    double time_ratio
) {
    if (!state || !input_interleaved || !output_interleaved) return -1;
    if (in_frames == 0) {
        std::memset(output_interleaved, 0, out_frames * sizeof(float) * ((SStretchInternal*)state)->channels);
        return (int)out_frames;
    }

    auto* s = (SStretchInternal*)state;
    int ch = s->channels;

    // 重置状态，确保干净处理
    s->stretch.reset();

    // 音高保持不变（pitch scale = 1.0），由调用方在外部设置
    // 时间拉伸通过 inputSamples / outputSamples 比值控制

    // 反交错输入
    s->deinterleave(input_interleaved, (int)in_frames);

    int inputLatency = s->stretch.inputLatency();
    int outputLatency = s->stretch.outputLatency();

    // 总输出 = 期望帧数 + outputLatency (pre-roll)
    int totalOutput = (int)out_frames + outputLatency;

    // 总输入 = 输入帧数 + inputLatency (尾部静音 flush)
    int totalInput = (int)in_frames + inputLatency;

    // 分块处理
    // 每次输入 blockIn 帧，输出 blockOut 帧，保持比例 ≈ time_ratio
    const int BLOCK = 1024;

    int inputConsumed = 0;
    int outputProduced = 0;

    // 准备扩展输入缓冲（追加 inputLatency 帧的静音）
    for (int c = 0; c < ch; ++c) {
        s->in_ch[c].resize(totalInput, 0.0f);
        // 尾部已经是 0（静音），用于 flush
    }

    // 收集所有输出
    std::vector<std::vector<float>> all_out(ch);
    for (int c = 0; c < ch; ++c) {
        all_out[c].reserve(totalOutput + BLOCK);
    }

    while (inputConsumed < totalInput || outputProduced < totalOutput) {
        // 计算本块输入帧数
        int remainIn = totalInput - inputConsumed;
        int blockIn = std::min(BLOCK, remainIn);

        // 计算对应输出帧数，保持比例
        int blockOut;
        if (blockIn > 0 && remainIn > 0) {
            // 按比例计算
            double progress = (double)inputConsumed / totalInput;
            int expectedOut = (int)std::round(progress * totalOutput);
            int nextExpectedOut = (int)std::round((double)(inputConsumed + blockIn) / totalInput * totalOutput);
            blockOut = nextExpectedOut - expectedOut;
            blockOut = std::max(1, blockOut);
        } else {
            // 输入已耗尽，仅产出
            blockOut = std::min(BLOCK, totalOutput - outputProduced);
            blockIn = 0;
        }

        if (blockOut <= 0 && blockIn <= 0) break;

        // 设置输入指针（偏移到当前位置）
        std::vector<const float*> inPtrs(ch);
        for (int c = 0; c < ch; ++c) {
            inPtrs[c] = s->in_ch[c].data() + inputConsumed;
        }

        // 准备输出缓冲
        std::vector<std::vector<float>> tmpOut(ch);
        std::vector<float*> outPtrs(ch);
        for (int c = 0; c < ch; ++c) {
            tmpOut[c].resize(blockOut, 0.0f);
            outPtrs[c] = tmpOut[c].data();
        }

        s->stretch.process(inPtrs.data(), blockIn, outPtrs.data(), blockOut);

        for (int c = 0; c < ch; ++c) {
            all_out[c].insert(all_out[c].end(), tmpOut[c].begin(), tmpOut[c].end());
        }

        inputConsumed += blockIn;
        outputProduced += blockOut;
    }

    // flush 残余
    {
        int flushFrames = outputLatency;
        std::vector<std::vector<float>> tmpOut(ch);
        std::vector<float*> outPtrs(ch);
        for (int c = 0; c < ch; ++c) {
            tmpOut[c].resize(flushFrames, 0.0f);
            outPtrs[c] = tmpOut[c].data();
        }
        s->stretch.flush(outPtrs.data(), flushFrames);
        for (int c = 0; c < ch; ++c) {
            all_out[c].insert(all_out[c].end(), tmpOut[c].begin(), tmpOut[c].end());
        }
    }

    // 跳过 outputLatency 的 pre-roll，取 out_frames 帧
    int skip = outputLatency;
    int available = (int)all_out[0].size() - skip;
    int copyFrames = std::min((int)out_frames, std::max(0, available));

    // 交错输出
    std::memset(output_interleaved, 0, out_frames * ch * sizeof(float));
    for (int f = 0; f < copyFrames; ++f) {
        for (int c = 0; c < ch; ++c) {
            output_interleaved[f * ch + c] = all_out[c][skip + f];
        }
    }

    return copyFrames;
}

int sstretch_flush(
    SStretchState state,
    float* output_interleaved,
    unsigned int out_frames
) {
    if (!state || !output_interleaved) return -1;
    if (out_frames == 0) return 0;

    auto* s = (SStretchInternal*)state;

    s->prepare_out_ptrs((int)out_frames);
    s->stretch.flush(s->out_ptrs.data(), (int)out_frames);
    s->interleave(output_interleaved, (int)out_frames);

    return (int)out_frames;
}

} // extern "C"
