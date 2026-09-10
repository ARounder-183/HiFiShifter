/*
 * cache_registry.rs - 统一缓存注册表与失效编排（P3-2）。
 *
 * 主要内容：
 * - `cache_registry::invalidate_clip`：按策略位失效某个 clip 的全部相关缓存。
 * - `REGISTRY`：所有"按 clip_id 失效"的缓存的唯一清单。
 *
 * 为什么需要它：
 * 失效逻辑此前是 `invalidate_clip_all_caches` 里的一串手写块，而**新增缓存时
 * 必须记得回到这个函数里加一段**。实践已经证明这条约定守不住：
 * `formant_cache::invalidate_formant_cache_for_clip` 有独立入口，却没有被并入
 * 编排函数，于是调用方只能在每个该失效的地方**手工配对**一次。当前 4 处配对
 * 是对的，但另有 6 处调用 `invalidate_clip_all_caches` 的地方没有配对 —— 也就是
 * 说，同样的删除/替换片段的操作，走不同命令会得到不同的失效覆盖。
 *
 * `FormantCache` 的 key 包含完整渲染参数，因此多数情况"漏失效"只会浪费内存而
 * 不会出错；但 key **不含源文件内容指纹**，源文件被同名替换后旧条目仍会被命中
 * —— 这正是这些手工配对想要防住的情况，也正是漏配对会变成真 bug 的地方。
 *
 * 改为注册表后，"新增缓存"变成在 `REGISTRY` 里加一行，且策略位显式声明它在哪些
 * 失效场景中生效，不再依赖调用方记得配对。
 */

/// 失效策略：结构性变更（源范围/轨道/速率/长度/工程切换/导入等）。
///
/// 旧渲染结果不可安全复用，且快照可能经旧 key 命中陈旧 PCM。
pub(crate) const POLICY_HARD: u8 = 1 << 0;

/// 失效策略：仅音高编辑。
///
/// 保留"整 clip 渲染"类缓存以支持无缝垫音（其 key 已含完整参数，参数真变化会
/// 自然产生新 hash，不会错误复用）；但按段合成缓存与 pending key 必须失效。
pub(crate) const POLICY_PITCH_EDIT: u8 = 1 << 1;

/// 注册表条目：一个"可按 clip_id 失效"的缓存。
pub(crate) struct CacheEntry {
    /// 诊断日志用的名称。
    pub(crate) name: &'static str,
    /// 参与哪些失效策略（策略位或）。
    pub(crate) policies: u8,
    /// 失效该 clip 的条目，返回**被移除的条目数**（0 = 本来就无条目）。
    pub(crate) invalidate_clip: fn(&str) -> usize,
}

// ─── 各缓存的失效适配器 ──────────────────────────────────────────────────────
// 统一为 `fn(&str) -> usize`：移除条目数。没有条目被移除时返回 0，编排函数据此
// 只对"真的有东西被删掉"的缓存打日志，避免日志噪音。

fn invalidate_pending_keys(clip_id: &str) -> usize {
    usize::from(crate::synth_clip_cache::remove_pending_rendered_key(clip_id))
}

fn invalidate_synth_clip(clip_id: &str) -> usize {
    let mut cache = crate::synth_clip_cache::global_synth_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let before = cache.len();
    cache.invalidate(clip_id);
    before.saturating_sub(cache.len())
}

fn invalidate_rendered_clip(clip_id: &str) -> usize {
    let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let before = cache.len();
    cache.invalidate(clip_id);
    before.saturating_sub(cache.len())
}

fn invalidate_tension_rendered_clip(clip_id: &str) -> usize {
    let mut cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let before = cache.len();
    cache.invalidate(clip_id);
    before.saturating_sub(cache.len())
}

fn invalidate_breath_noise(clip_id: &str) -> usize {
    let mut cache = crate::synth_clip_cache::global_breath_noise_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let before = cache.len();
    cache.invalidate(clip_id);
    before.saturating_sub(cache.len())
}

/// 共振峰缓存此前**不在**编排函数内，靠调用方手工配对；并入后所有
/// `invalidate_clip_all_caches` 调用点自动获得正确覆盖（见本文件头部说明）。
fn invalidate_formant(clip_id: &str) -> usize {
    let mut cache = crate::formant_cache::global_formant_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let before = cache.len();
    cache.invalidate(clip_id);
    before.saturating_sub(cache.len())
}

fn invalidate_hifigan_chunks(clip_id: &str) -> usize {
    // 分块缓存按 clip_id 索引；源文件被替换后旧推理输出不再有效。
    crate::renderer::hifigan::invalidate_chunk_cache_for_clip(clip_id);
    // 该缓存不对外暴露"移除了几条"，因此只报告是否可能被触及。
    0
}

// ─── 注册表 ─────────────────────────────────────────────────────────────────

pub(crate) const REGISTRY: &[CacheEntry] = &[
    CacheEntry {
        name: "pending_rendered_keys",
        policies: POLICY_HARD | POLICY_PITCH_EDIT,
        invalidate_clip: invalidate_pending_keys,
    },
    CacheEntry {
        name: "SynthClipCache",
        policies: POLICY_HARD | POLICY_PITCH_EDIT,
        invalidate_clip: invalidate_synth_clip,
    },
    CacheEntry {
        name: "RenderedClipCache",
        policies: POLICY_HARD,
        invalidate_clip: invalidate_rendered_clip,
    },
    CacheEntry {
        name: "TensionRenderedClipCache",
        policies: POLICY_HARD,
        invalidate_clip: invalidate_tension_rendered_clip,
    },
    CacheEntry {
        name: "BreathNoiseCache",
        policies: POLICY_HARD,
        invalidate_clip: invalidate_breath_noise,
    },
    CacheEntry {
        name: "FormantCache",
        policies: POLICY_HARD,
        invalidate_clip: invalidate_formant,
    },
    CacheEntry {
        name: "HifiganChunkCache",
        policies: POLICY_HARD,
        invalidate_clip: invalidate_hifigan_chunks,
    },
];

/// 按 `policy` 失效 `clip_id` 相关的全部缓存。
///
/// 返回 `(有内容被移除的缓存数, 移除条目总数)`。
pub(crate) fn invalidate_clip(clip_id: &str, policy: u8) -> (usize, usize) {
    let mut touched = 0usize;
    let mut removed = 0usize;

    for entry in REGISTRY {
        if entry.policies & policy == 0 {
            continue;
        }
        let n = (entry.invalidate_clip)(clip_id);
        if n > 0 {
            touched += 1;
            removed += n;
            debug_eprintln!(
                "[cache:invalidate] clip_id={clip_id} {} removed {n} entr{}",
                entry.name,
                if n == 1 { "y" } else { "ies" }
            );
        }
    }

    (touched, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_entry_declares_at_least_one_policy() {
        for entry in REGISTRY {
            assert_ne!(
                entry.policies & (POLICY_HARD | POLICY_PITCH_EDIT),
                0,
                "cache `{}` participates in no invalidation policy and would never be invalidated",
                entry.name
            );
        }
    }

    #[test]
    fn cache_names_are_unique() {
        let mut seen = HashSet::new();
        for entry in REGISTRY {
            assert!(
                seen.insert(entry.name),
                "duplicate cache name `{}` in registry",
                entry.name
            );
        }
    }

    #[test]
    fn pitch_edit_policy_is_a_strict_subset_of_hard() {
        // 音高编辑保留"整 clip 渲染"类缓存（垫音需要），但绝不应**多**失效
        // 出 hard 之外的东西 —— 否则语义反了。
        for entry in REGISTRY {
            if entry.policies & POLICY_PITCH_EDIT != 0 {
                assert!(
                    entry.policies & POLICY_HARD != 0,
                    "cache `{}` is invalidated on pitch edit but not on structural changes",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn formant_cache_is_covered_by_structural_invalidation() {
        // 这条断言锁住 P3-2 修复的具体缺陷：FormantCache 曾因未并入编排函数
        // 而依赖调用方手工配对，导致 6 处调用点覆盖不全。
        let entry = REGISTRY
            .iter()
            .find(|e| e.name == "FormantCache")
            .expect("FormantCache must be registered");
        assert_ne!(
            entry.policies & POLICY_HARD,
            0,
            "FormantCache must be invalidated on structural changes"
        );
    }

    #[test]
    fn invalidating_an_unknown_clip_is_a_no_op() {
        let (touched, removed) = invalidate_clip("__no_such_clip__", POLICY_HARD);
        assert_eq!(removed, 0, "unknown clip must remove no entries");
        // pending_rendered_keys 返回 0，分块缓存恒返回 0，其余缓存无条目 → 0。
        assert_eq!(touched, 0);
    }

    #[test]
    fn pitch_edit_invalidates_a_smaller_set_than_hard() {
        let hard: Vec<&str> = REGISTRY
            .iter()
            .filter(|e| e.policies & POLICY_HARD != 0)
            .map(|e| e.name)
            .collect();
        let pitch: Vec<&str> = REGISTRY
            .iter()
            .filter(|e| e.policies & POLICY_PITCH_EDIT != 0)
            .map(|e| e.name)
            .collect();

        assert!(pitch.len() < hard.len(), "pitch edit must be the narrower policy");
        // 整 clip 渲染缓存必须保留，否则音高编辑会丢掉无缝垫音。
        assert!(!pitch.contains(&"RenderedClipCache"));
        assert!(pitch.contains(&"SynthClipCache"));
        assert!(pitch.contains(&"pending_rendered_keys"));
    }
}
