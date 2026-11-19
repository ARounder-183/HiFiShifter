//! item state chunk 里**一条 take 的字段区**：按 take GUID 定位，单 take / 多 take 同一套解析。
//!
//! # 为什么要这一层
//!
//! REAPER 的 item state chunk 有两种形态：
//!
//! * **单 take**：take 的字段（`GUID` / `SOFFS` / `<SOURCE>` / `<PITCHENV>` …）直接挂在
//!   `<ITEM>` 下（深度 1）；
//! * **多 take**：每条 take 之前有一行裸的 `TAKE` 标记，其余结构不变 —— 于是同一个深度上
//!   有 N 份 `GUID`、N 个 `<SOURCE>`、N 个 `<PITCHENV>`。
//!
//! 两条结构写回（`take_pitch` 与 `source_direction`）此前都只实现了第一种形态，对第二种
//! 一律返回 "the host item has more than one take"。而 take FX 恰好**最常**挂在多 take
//! item 的某一条上 —— 于是"把音高线转成 take 包络"和"倒放"在 take FX 下直接失败。
//!
//! 这一层把 chunk 切成"每一条 take 的字段区"（[`TakeSection`]），并保证所有定位都**先按
//! GUID 认出那一条**、再在它的字段区里找块。凡是指定的 take 找不到唯一匹配就报错，绝不
//! 退回"第一个" —— 猜错一条 take 等于改坏别人的音频。
//!
//! # 纪律：只定位，不重写
//!
//! 本模块只回答"哪几行属于这条 take / 这个块在哪"，改写仍然由调用方逐行原样输出其余内容。
//! 把"解析"与"重写"分开，是"除目标块之外逐字节保留"这条纪律能被单测钉住的前提。

use super::item_chunk::MAX_ITEM_CHUNK_BYTES;

/// 一条 take 的字段区（行下标半开区间 `[start, end)`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TakeSection {
    pub start: usize,
    pub end: usize,
}

/// 块（`<KIND …>` 到配对的 `>`）的行下标，闭区间。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub start: usize,
    pub end: usize,
}

/// 校验 chunk 的基本形状并切出全部 take 的字段区。
pub(super) fn sections(lines: &[&str]) -> Result<Vec<TakeSection>, String> {
    let mut stack: Vec<&str> = Vec::new();
    let mut sections: Vec<TakeSection> = Vec::new();
    let mut current: Option<usize> = None;
    let mut item_seen = false;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(open) = trimmed.strip_prefix('<') {
            let kind = open.split_whitespace().next().unwrap_or("");
            if stack.len() >= 64 {
                return Err("item state nesting budget exceeded".into());
            }
            if stack.is_empty() {
                if item_seen || kind != "ITEM" {
                    return Err("item state must contain exactly one ITEM block".into());
                }
                item_seen = true;
                current = Some(index + 1);
            }
            if kind == "ITEM" && !stack.is_empty() {
                return Err("nested ITEM state is unsupported".into());
            }
            stack.push(kind);
            continue;
        }
        if trimmed == ">" {
            if stack.pop().is_none() {
                return Err("unbalanced item state".into());
            }
            if stack.is_empty() {
                if let Some(start) = current.take() {
                    sections.push(TakeSection { start, end: index });
                }
            }
            continue;
        }
        // 裸 `TAKE` 标记（深度 1）：它**不属于**任何一条 take 的字段区，只是分节符。
        if stack.len() == 1 && trimmed == "TAKE" {
            if let Some(start) = current.take() {
                sections.push(TakeSection { start, end: index });
            }
            current = Some(index + 1);
        }
    }
    if !stack.is_empty() || !item_seen {
        return Err("unbalanced or missing item state".into());
    }
    if sections.is_empty() {
        return Err("item state has no take".into());
    }
    Ok(sections)
}

/// 找出 GUID 等于 `take_guid` 的那一条 take 的字段区。
pub(super) fn section_with_guid(lines: &[&str], take_guid: &str) -> Result<TakeSection, String> {
    if take_guid.is_empty() {
        return Err("host take identity missing".into());
    }
    let mut found: Option<TakeSection> = None;
    for section in sections(lines)? {
        let Some(index) = field(lines, &section, "GUID") else {
            continue;
        };
        if lines[index].split_whitespace().nth(1) != Some(take_guid) {
            continue;
        }
        if found.is_some() {
            // 两条 take 报同一个 GUID 是 chunk 损坏的证据，不能挑一个写。
            return Err("item state repeats the take identity".into());
        }
        found = Some(section);
    }
    found.ok_or_else(|| "the host item does not hold the bound take".to_string())
}

/// 字段区里深度 0 的字段行（`SOFFS` / `GUID` …）。
pub(super) fn field(lines: &[&str], section: &TakeSection, name: &str) -> Option<usize> {
    let mut depth = 0usize;
    for index in section.start..section.end.min(lines.len()) {
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('<') {
            depth += 1;
            continue;
        }
        if trimmed == ">" {
            depth = depth.saturating_sub(1);
            continue;
        }
        if depth == 0 && trimmed.split_whitespace().next() == Some(name) {
            return Some(index);
        }
    }
    None
}

/// 字段区里深度 0 的块（`<PITCHENV>` / `<SOURCE>` / `<TAKEFX>` …）。
pub(super) fn block(lines: &[&str], section: &TakeSection, kind: &str) -> Option<Block> {
    let mut depth = 0usize;
    let mut open: Option<usize> = None;
    for index in section.start..section.end.min(lines.len()) {
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('<') {
            if depth == 0 && open.is_none() && rest.split_whitespace().next() == Some(kind) {
                open = Some(index);
            }
            depth += 1;
            continue;
        }
        if trimmed == ">" {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                if let Some(start) = open.take() {
                    return Some(Block { start, end: index });
                }
            }
        }
    }
    None
}

/// 这份 chunk 的字节预算（与 `item_chunk::MAX_ITEM_CHUNK_BYTES` 同一上限）。
pub(super) fn check_budget(chunk: &str) -> Result<(), String> {
    if chunk.is_empty() || chunk.len() > MAX_ITEM_CHUNK_BYTES || chunk.contains('\0') {
        return Err("invalid or oversized REAPER item state".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAKE_A: &str = "{AAAAAAAA-1111-2222-3333-444444444444}";
    const TAKE_B: &str = "{BBBBBBBB-1111-2222-3333-444444444444}";

    fn single() -> Vec<&'static str> {
        vec![
            "<ITEM",
            "POSITION 0",
            "LENGTH 1",
            "GUID {AAAAAAAA-1111-2222-3333-444444444444}",
            "<SOURCE WAVE",
            "FILE \"a.wav\"",
            ">",
            ">",
        ]
    }

    fn multi() -> Vec<&'static str> {
        vec![
            "<ITEM",
            "POSITION 0",
            "LENGTH 1",
            "GUID {AAAAAAAA-1111-2222-3333-444444444444}",
            "<SOURCE WAVE",
            "FILE \"a.wav\"",
            ">",
            "<PITCHENV",
            "PT 0 1 0",
            ">",
            "TAKE",
            "GUID {BBBBBBBB-1111-2222-3333-444444444444}",
            "SOFFS 2",
            "<SOURCE WAVE",
            "FILE \"b.wav\"",
            ">",
            ">",
        ]
    }

    /// 单 take：整份 ITEM 就是一个字段区，`TAKE` 标记不参与。
    #[test]
    fn a_single_take_item_is_one_section() {
        let lines = single();
        let sections = sections(&lines).unwrap();
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].start, 1);
        assert_eq!(field(&lines, &sections[0], "GUID"), Some(3));
        assert_eq!(
            block(&lines, &sections[0], "SOURCE").map(|b| b.start),
            Some(4)
        );
    }

    /// 多 take：每个 `TAKE` 标记切一刀，字段与块各自归位。
    #[test]
    fn a_multi_take_item_splits_on_the_take_marker() {
        let lines = multi();
        let sections = sections(&lines).unwrap();
        assert_eq!(sections.len(), 2);
        let first = section_with_guid(&lines, TAKE_A).unwrap();
        let second = section_with_guid(&lines, TAKE_B).unwrap();
        assert_eq!(first, sections[0]);
        assert_eq!(second, sections[1]);
        // 第二条 take 的 SOFFS 与 SOURCE 只在它自己的字段区里。
        assert_eq!(field(&lines, &second, "SOFFS"), Some(12));
        assert_eq!(block(&lines, &second, "SOURCE").map(|b| b.start), Some(13));
        assert!(field(&lines, &first, "SOFFS").is_none());
        // 第一条 take 的包络不会被认成第二条的。
        assert!(block(&lines, &first, "PITCHENV").is_some());
        assert!(block(&lines, &second, "PITCHENV").is_none());
    }

    /// 找不到 / 重复的 GUID 都必须报错，绝不退回"第一个"。
    #[test]
    fn an_unknown_or_duplicated_identity_is_refused() {
        let lines = multi();
        assert!(section_with_guid(&lines, "{CCCCCCCC-1111-2222-3333-444444444444}").is_err());
        assert!(section_with_guid(&lines, "").is_err());
        let mut duplicated = multi();
        duplicated[11] = "GUID {AAAAAAAA-1111-2222-3333-444444444444}";
        assert!(section_with_guid(&duplicated, TAKE_A).is_err());
    }

    /// 结构坏了就报错（截断 / 未闭合 / 没有 ITEM）。
    #[test]
    fn malformed_state_is_refused() {
        assert!(sections(&["<ITEM", "POSITION 0"]).is_err());
        assert!(sections(&["POSITION 0", ">"]).is_err());
        assert!(sections(&["<ITEM", "<ITEM", ">", ">"]).is_err());
        assert!(check_budget("").is_err());
    }
}
