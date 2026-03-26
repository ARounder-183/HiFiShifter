//! VstProcessingStage：将 VST 插件封装为 ProcessingStage。
//!
//! 实现 `ProcessingStage` trait，使 VST 效果器可以无缝接入
//! `ProcessorChain` 的 Stage 链中。
//!
//! 音频处理流程：
//! 1. 接收上一 Stage 输出的单声道 PCM
//! 2. 转换为立体声（duplicate mono → L/R）
//! 3. 按 block_size 分块送入 VST 插件处理
//! 4. 将输出合并为单声道返回

use std::sync::{Arc, Mutex};

use crate::renderer::chain::{ProcessingStage, StageContext};
use super::plugin_instance::VstPluginInstance;

/// 将单个 VST 插件实例封装为 ProcessingStage。
pub struct VstProcessingStage {
    /// 插件实例（线程安全封装）。
    pub instance: Arc<Mutex<VstPluginInstance>>,
    /// Stage 唯一 ID（通常是 "vst_{instance_id}"）。
    pub stage_id: String,
    /// Stage 显示名称（插件名称）。
    pub display_name: String,
}

impl VstProcessingStage {
    pub fn new(
        instance: Arc<Mutex<VstPluginInstance>>,
        stage_id: String,
        display_name: String,
    ) -> Self {
        Self {
            instance,
            stage_id,
            display_name,
        }
    }
}

impl ProcessingStage for VstProcessingStage {
    fn id(&self) -> &str {
        &self.stage_id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let mut inst = self.instance.lock().map_err(|e| format!("VST lock failed: {}", e))?;

        if inst.bypassed {
            return Ok(input_pcm);
        }

        let sample_rate = ctx.clip_ctx.sample_rate as f32;

        // 确保插件使用正确的采样率
        if (inst.sample_rate - sample_rate).abs() > 1.0 {
            inst.set_sample_rate(sample_rate);
        }

        let block_size = inst.block_size.max(64);
        let num_inputs = inst.num_inputs.max(1) as usize;
        let num_outputs = inst.num_outputs.max(1) as usize;
        let total_samples = input_pcm.len();

        // ── 准备输入缓冲区 ────────────────────────────────────────────────
        // 输入是单声道 PCM，需要根据插件的输入通道数进行适配
        let mut input_channels: Vec<Vec<f32>> = Vec::with_capacity(num_inputs);
        for ch in 0..num_inputs {
            if ch == 0 {
                // 第一个通道使用原始单声道数据
                input_channels.push(input_pcm.clone());
            } else {
                // 额外通道复制第一通道（mono → stereo duplicate）
                input_channels.push(input_pcm.clone());
            }
        }

        // ── 准备输出缓冲区 ────────────────────────────────────────────────
        let mut output_channels: Vec<Vec<f32>> = Vec::with_capacity(num_outputs);
        for _ in 0..num_outputs {
            output_channels.push(vec![0.0f32; total_samples]);
        }

        // ── 分块处理 ──────────────────────────────────────────────────────
        let mut offset = 0;
        while offset < total_samples {
            let chunk_len = (total_samples - offset).min(block_size);

            // 切出输入块
            let input_chunks: Vec<Vec<f32>> = input_channels
                .iter()
                .map(|ch| ch[offset..offset + chunk_len].to_vec())
                .collect();

            // 准备输出块
            let mut output_chunks: Vec<Vec<f32>> = (0..num_outputs)
                .map(|_| vec![0.0f32; chunk_len])
                .collect();

            // VST 处理
            inst.process(&input_chunks, &mut output_chunks);

            // 复制输出
            for (ch_idx, chunk) in output_chunks.iter().enumerate() {
                if ch_idx < output_channels.len() {
                    output_channels[ch_idx][offset..offset + chunk_len]
                        .copy_from_slice(&chunk[..chunk_len]);
                }
            }

            offset += chunk_len;
        }

        // ── 输出合并为单声道 ──────────────────────────────────────────────
        // 如果有多个输出通道，取平均值
        let output_mono = if num_outputs == 1 {
            output_channels.into_iter().next().unwrap_or(input_pcm)
        } else {
            let mut mono = vec![0.0f32; total_samples];
            let inv = 1.0 / num_outputs as f32;
            for ch in &output_channels {
                for (i, &sample) in ch.iter().enumerate() {
                    if i < mono.len() {
                        mono[i] += sample * inv;
                    }
                }
            }
            mono
        };

        Ok(output_mono)
    }
}

// ─── 轨道 FX 链 Stage 构建 ──────────────────────────────────────────────────

/// 从轨道的 VstChainConfig 构建 ProcessingStage 列表。
///
/// 优先从 `VstPluginRegistry.instances` 中复用已有实例（与编辑器 GUI 共享同一
/// `Arc<Mutex<VstPluginInstance>>`），使得编辑器中的参数调整能实时影响音频处理。
/// 仅在注册表中找不到实例时才从磁盘加载新实例。
pub fn build_vst_stages_for_track(
    track_id: &str,
    chain_config: &super::VstChainConfig,
    registry: &super::VstPluginRegistry,
    sample_rate: f32,
    block_size: usize,
) -> Vec<Box<dyn ProcessingStage>> {
    let mut stages: Vec<Box<dyn ProcessingStage>> = Vec::new();

    for (idx, plugin_state) in chain_config.plugins.iter().enumerate() {
        if plugin_state.bypassed {
            continue;
        }

        let path = &plugin_state.plugin_path;
        if !path.exists() {
            eprintln!(
                "[vst_host::stage] Plugin not found, skipping: {}",
                path.display()
            );
            continue;
        }

        // 生成与 vst_open_editor / build_vst_stages_map 一致的实例 ID
        let instance_id = format!("{}:{}:{}", track_id, idx, plugin_state.plugin_uid);

        // 优先从注册表复用已有实例（与编辑器 GUI 共享）
        let existing = {
            let instances = registry.instances.lock().unwrap_or_else(|e| e.into_inner());
            instances.get(&instance_id).cloned()
        };

        let instance = if let Some(inst) = existing {
            // 确保采样率正确
            {
                let mut locked = inst.lock().unwrap_or_else(|e| e.into_inner());
                if (locked.sample_rate - sample_rate).abs() > 1.0 {
                    locked.set_sample_rate(sample_rate);
                }
            }
            inst
        } else {
            // 注册表中没有，从磁盘加载新实例
            let instance_result = match plugin_state.format {
                super::VstFormat::Vst2 => {
                    super::plugin_host::load_vst2(path, sample_rate, block_size as i64)
                }
                super::VstFormat::Vst3 => {
                    super::plugin_host::load_vst3(path, sample_rate as f64, block_size as i32)
                }
            };

            match instance_result {
                Ok(inst) => {
                    // 恢复插件状态
                    if let Some(ref chunk) = plugin_state.chunk_data {
                        let mut locked = inst.lock().unwrap_or_else(|e| e.into_inner());
                        if let Err(e) = locked.set_chunk(chunk) {
                            eprintln!(
                                "[vst_host::stage] Failed to restore chunk for '{}': {}",
                                plugin_state.plugin_uid, e
                            );
                        }
                    }

                    // 存入注册表以便后续复用
                    {
                        let mut instances = registry.instances.lock().unwrap_or_else(|e| e.into_inner());
                        instances.insert(instance_id.clone(), inst.clone());
                    }

                    inst
                }
                Err(e) => {
                    eprintln!(
                        "[vst_host::stage] Failed to load plugin '{}': {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            }
        };

        let stage_id = format!("vst_{}_{}", plugin_state.plugin_uid, idx);
        let name = instance
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .name
            .clone();

        stages.push(Box::new(VstProcessingStage::new(
            instance,
            stage_id,
            name,
        )));
    }

    stages
}
