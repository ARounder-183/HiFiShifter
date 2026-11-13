//! HNSEP 长音频的有界分块计划。
//!
//! 分块边界按 SEGMENT_FRAMES × HOP = 32 × 512 = 16,384 个 44.1kHz 样本对齐。
//! 每个 core 为约 8 秒输出，左右各带约 2 秒上下文；相邻 core 有 16,384 样本
//! 的交叉淡化区。计划只负责几何，不执行推理。

pub const HNSEP_CHUNK_ALIGN_SAMPLES: usize = 32 * 512;
pub const HNSEP_CHUNK_CORE_SAMPLES: usize =
    (8 * 44_100 / HNSEP_CHUNK_ALIGN_SAMPLES) * HNSEP_CHUNK_ALIGN_SAMPLES;
pub const HNSEP_CHUNK_CONTEXT_SAMPLES: usize =
    (2 * 44_100_usize).div_ceil(HNSEP_CHUNK_ALIGN_SAMPLES) * HNSEP_CHUNK_ALIGN_SAMPLES;
pub const HNSEP_CHUNK_CROSSFADE_SAMPLES: usize = HNSEP_CHUNK_ALIGN_SAMPLES;
pub const HNSEP_FULL_MAX_SAMPLES: usize = 30 * 44_100;
pub const HNSEP_CHUNK_MAX_COUNT: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnsepChunk {
    pub core_start: usize,
    pub core_end: usize,
    pub input_start: usize,
    pub input_end: usize,
}

pub fn plan_chunks(total_samples: usize) -> Vec<HnsepChunk> {
    if total_samples == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut core_start = 0usize;
    while core_start < total_samples {
        if out.len() >= HNSEP_CHUNK_MAX_COUNT {
            return Vec::new();
        }
        let core_end = (core_start + HNSEP_CHUNK_CORE_SAMPLES).min(total_samples);
        let input_start = core_start.saturating_sub(HNSEP_CHUNK_CONTEXT_SAMPLES);
        let input_end = (core_end + HNSEP_CHUNK_CONTEXT_SAMPLES).min(total_samples);
        out.push(HnsepChunk {
            core_start,
            core_end,
            input_start,
            input_end,
        });
        if core_end >= total_samples {
            break;
        }
        let next = core_end.saturating_sub(HNSEP_CHUNK_CROSSFADE_SAMPLES);
        core_start = if next <= core_start {
            core_start + HNSEP_CHUNK_ALIGN_SAMPLES
        } else {
            next
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_whole_input_and_are_bounded() {
        for total in [1usize, 16_383, 16_384, 44_100 * 8, 44_100 * 227] {
            let chunks = plan_chunks(total);
            assert!(!chunks.is_empty(), "total={total}");
            assert_eq!(chunks[0].input_start, 0);
            assert_eq!(chunks.last().unwrap().input_end, total);
            assert_eq!(chunks.last().unwrap().core_end, total);
            let mut covered = 0usize;
            for chunk in &chunks {
                assert!(chunk.core_end > chunk.core_start);
                assert!(chunk.input_end > chunk.input_start);
                assert!(
                    chunk.input_end - chunk.input_start
                        <= HNSEP_CHUNK_CORE_SAMPLES + 2 * HNSEP_CHUNK_CONTEXT_SAMPLES
                );
                assert!(chunk.core_start <= covered, "core must not leave a hole");
                covered = covered.max(chunk.core_end);
            }
            assert_eq!(covered, total);
        }
    }

    #[test]
    fn empty_and_absurd_inputs_are_bounded() {
        assert!(plan_chunks(0).is_empty());
        assert!(plan_chunks(usize::MAX / 2).is_empty());
    }

    #[test]
    fn a_long_clip_is_split_into_many_bounded_windows() {
        let chunks = plan_chunks(44_100 * 227);
        assert!(chunks.len() > 20);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.input_end - chunk.input_start
                <= 12 * 44_100 + HNSEP_CHUNK_ALIGN_SAMPLES * 2));
    }
}
