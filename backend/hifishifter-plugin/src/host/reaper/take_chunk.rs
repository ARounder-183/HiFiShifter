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
//! # take 标记有四种写法
//!
//! 宿主写的是 `TAKE` / `TAKE SEL` / `TAKE NULL` / `TAKE NULL SEL`（`reaper.exe` 里
//! 这四个字面量就是宿主读取器逐字比较的那张表）。只认裸 `TAKE` 会让**活跃**的那条 take
//! 的标记被当成普通字段：它的字段区于是并进前一条 take，`GUID` 认到的是上一条 —— 表现
//! 是"这条 take 不在 item 里"，而用户明明看得见它。
//!
//! # 一行也能自带闭合符
//!
//! RPPXML 允许叶子块写在同一行（`<KIND …>`）。这种行既不是"开"也不是"关"：把它当开会让
//! 栈在文件末尾多出一层，整份 chunk 被判成"不完整" —— 而调用方据此认定宿主读坏了数据，
//! 一个本来完好的 item 就此写不进去。
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

/// 一行的形状。
enum Line<'a> {
    /// `<KIND …`：开一个块，等它自己的 `>`。
    Open(&'a str),
    /// `<KIND …>`：自带闭合符，栈深度不变。
    Leaf,
    /// `>`：关一个块。
    Close,
    /// 其余：块内字段（或空行）。
    Field,
}

/// 判定一行的形状。
///
/// 【为什么只此一份】三处扫描（分节 / 找字段 / 找块）共用它，于是"自闭合算不算开"这类
/// 问题不可能在一处修了、另一处忘了修。
fn classify(line: &str) -> Line<'_> {
    let trimmed = line.trim();
    if let Some(open) = trimmed.strip_prefix('<') {
        if trimmed.ends_with('>') {
            return Line::Leaf;
        }
        return Line::Open(open.split_whitespace().next().unwrap_or(""));
    }
    if trimmed == ">" {
        Line::Close
    } else {
        Line::Field
    }
}

/// 深度 1 上的 take 标记（`TAKE` / `TAKE SEL` / `TAKE NULL` / `TAKE NULL SEL`）。
///
/// `TAKECOLOR` / `TAKEFX_NCH` 这类字段虽然以 `TAKE` 开头，但不是整词 `TAKE`，不会误判。
fn is_take_marker(line: &str) -> bool {
    let mut words = line.split_whitespace();
    words.next() == Some("TAKE") && words.all(|word| matches!(word, "SEL" | "NULL"))
}

/// 校验 chunk 的基本形状并切出全部 take 的字段区。
pub(super) fn sections(lines: &[&str]) -> Result<Vec<TakeSection>, String> {
    let scan = scan(lines)?;
    if scan.sections.is_empty() {
        return Err("item state has no take".into());
    }
    Ok(scan.sections)
}

/// 这份文本是不是**一整份** item chunk（而不是被宿主按缓冲区大小截断的前半份）。
///
/// 【为什么要单独回答它】`GetItemStateChunk` 能写多少由调用方给的缓冲区决定；读到的文本
/// 可能只是一份前缀。用"能不能解析成一整份 ITEM"当判据，比再猜一个字节数可靠 —— 截断
/// 必然留下未闭合的块，而完好的 chunk 一定闭合。
///
/// 【为什么**不**要求"里面有 take"】这条判据服务的是"读没读完"，不是"能不能写包络"。
/// 剪贴板复制同样走它，而一个空 item（没有可用的 take 身份）照样有完整的状态；
/// 把它判成"没读完"只会让复制对着 8 MiB 的预算空转一圈然后报一个错的原因。
pub(super) fn is_complete_item_state(text: &str) -> bool {
    if check_budget(text).is_err() {
        return false;
    }
    let lines: Vec<&str> = text.lines().collect();
    scan(&lines).is_ok()
}

struct Scan {
    sections: Vec<TakeSection>,
}

/// 唯一的扫描器：分节 + 校验形状。
fn scan(lines: &[&str]) -> Result<Scan, String> {
    let mut stack: Vec<&str> = Vec::new();
    let mut sections: Vec<TakeSection> = Vec::new();
    let mut current: Option<usize> = None;
    let mut item_seen = false;
    let mut item_closed = false;
    for (index, line) in lines.iter().enumerate() {
        match classify(line) {
            Line::Field => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if item_closed {
                    // 【ITEM 之后还有内容】那不是一份完整的 item chunk。
                    return Err("item state continues past the ITEM block".into());
                }
                if stack.len() == 1 && is_take_marker(trimmed) {
                    if let Some(start) = current.take() {
                        push_section(&mut sections, start, index);
                    }
                    current = Some(index + 1);
                }
            }
            Line::Open(kind) => {
                if item_closed {
                    return Err("item state continues past the ITEM block".into());
                }
                if stack.len() >= 64 {
                    return Err("item state nesting budget exceeded".into());
                }
                if stack.is_empty() {
                    if item_seen || kind != "ITEM" {
                        return Err("item state must contain exactly one ITEM block".into());
                    }
                    item_seen = true;
                    current = Some(index + 1);
                } else if kind == "ITEM" {
                    return Err("nested ITEM state is unsupported".into());
                }
                stack.push(kind);
            }
            // 自闭合行不改变栈深度；但它同样只能出现在 ITEM 之内。
            Line::Leaf => {
                if item_closed {
                    return Err("item state continues past the ITEM block".into());
                }
            }
            Line::Close => {
                if stack.pop().is_none() {
                    return Err("unbalanced item state".into());
                }
                if stack.is_empty() {
                    if let Some(start) = current.take() {
                        push_section(&mut sections, start, index);
                    }
                    item_closed = true;
                }
            }
        }
    }
    if !stack.is_empty() || !item_seen {
        return Err("unbalanced or missing item state".into());
    }
    // 【只留"真的是一条 take"的字段区】多 take item 里，`<ITEM` 与第一个标记之间是 **item 级**
    // 字段（`POSITION` / `LENGTH` / `IGUID` …），而 `TAKE NULL` 是个空槽。判据是内容
    // —— 有没有 take 自己的 `GUID` —— 而不是位置：这样"第一条 take 带不带标记"两种宿主写法
    // 都成立（真实工程里两种都存在），也不会让 item 级那一段冒充"第 0 条 take"。
    sections.retain(|section| field(lines, section, "GUID").is_some());
    Ok(Scan { sections })
}

/// 只有非空的字段区才算一条 take（多 take 的第一个标记会切出一段空区间）。
fn push_section(sections: &mut Vec<TakeSection>, start: usize, end: usize) {
    if start < end {
        sections.push(TakeSection { start, end });
    }
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
        match classify(lines[index]) {
            Line::Open(_) => depth += 1,
            Line::Leaf => {}
            Line::Close => depth = depth.saturating_sub(1),
            Line::Field => {
                if depth == 0 && lines[index].split_whitespace().next() == Some(name) {
                    return Some(index);
                }
            }
        }
    }
    None
}

/// 字段区里深度 0 的块（`<PITCHENV>` / `<SOURCE>` / `<TAKEFX>` …）。
pub(super) fn block(lines: &[&str], section: &TakeSection, kind: &str) -> Option<Block> {
    let mut depth = 0usize;
    let mut open: Option<usize> = None;
    for index in section.start..section.end.min(lines.len()) {
        match classify(lines[index]) {
            Line::Open(name) => {
                if depth == 0 && open.is_none() && name == kind {
                    open = Some(index);
                }
                depth += 1;
            }
            Line::Leaf => {}
            Line::Close => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(start) = open.take() {
                        return Some(Block { start, end: index });
                    }
                }
            }
            Line::Field => {}
        }
    }
    None
}

/// 一段（可能不完整的）item state 的形状摘要，供"宿主给的 chunk 用不了"时写日志。
///
/// 【为什么是形状而不是全文】chunk 里可能带着插件状态与文件路径（体积从几十 KB 到几 MB）。
/// 排查需要的只有"它断在哪一层"：字节数、非空行数、扫描结束时还剩几层没闭合、以及
/// 首尾两行 —— 三样合起来足以区分"被截断"（depth_end > 0）、"多了一个闭合"（depth_min < 0）
/// 与"根本不是 item chunk"（first 不是 `<ITEM`）。
pub(super) fn describe_shape(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let mut depth = 0_i32;
    let mut lowest = 0_i32;
    for line in &lines {
        match classify(line) {
            Line::Open(_) => depth += 1,
            Line::Leaf => {}
            Line::Close => {
                depth -= 1;
                lowest = lowest.min(depth);
            }
            Line::Field => {}
        }
    }
    let clip = |line: &str| -> String {
        if line.chars().count() <= 60 {
            line.to_owned()
        } else {
            line.chars().take(60).collect::<String>() + "…"
        }
    };
    format!(
        "{} bytes, {} non-empty lines, depth_end={}, depth_min={}, first={:?}, last={:?}",
        text.len(),
        lines.len(),
        depth,
        lowest,
        lines
            .first()
            .map_or(String::new(), |line| clip(line.trim())),
        lines.last().map_or(String::new(), |line| clip(line.trim())),
    )
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

    const TAKE_C: &str = "{CCCCCCCC-1111-2222-3333-444444444444}";

    /// 三条 take、标记写法各不相同的一份 chunk（宿主就这四种写法）。
    fn marked(markers: [&str; 3], guids: [&str; 3]) -> String {
        let mut text = String::from(
            "<ITEM
POSITION 0
LENGTH 1
",
        );
        for (marker, guid) in markers.iter().zip(guids) {
            if !marker.is_empty() {
                text.push_str(marker);
                text.push('\n');
            }
            text.push_str("NAME take\n");
            text.push_str(&format!("GUID {guid}\n"));
            text.push_str("<SOURCE WAVE\nFILE \"a.wav\"\n>\n");
        }
        text.push_str(">\n");
        text
    }

    /// **四种 take 标记都得分节。**
    ///
    /// 【为什么这条是"这条 take 不在 item 里"的回归】只认裸 `TAKE` 时，活跃那条 take 的标记
    /// （`TAKE SEL`）被当成普通字段，它的字段区并进前一条 take —— 于是被绑定的 take 认到的
    /// 是上一条的 `GUID`，写回直接报"宿主 item 里没有这条 take"，而用户明明看得见它。
    #[test]
    fn every_take_marker_variant_splits_the_sections() {
        let text = marked(["TAKE", "TAKE SEL", "TAKE NULL"], [TAKE_A, TAKE_B, TAKE_C]);
        let lines: Vec<&str> = text.lines().collect();
        let sections = sections(&lines).unwrap();
        assert_eq!(sections.len(), 3, "三种标记各切出一条 take");
        for guid in [TAKE_A, TAKE_B, TAKE_C] {
            let section = section_with_guid(&lines, guid).unwrap();
            let guid_line = field(&lines, &section, "GUID").unwrap();
            assert_eq!(lines[guid_line].split_whitespace().nth(1), Some(guid));
            // 每条 take 的 SOURCE 只在自己的字段区里。
            assert!(block(&lines, &section, "SOURCE").is_some(), "{guid}");
        }
        // 字段区互不重叠。
        assert!(sections.windows(2).all(|pair| pair[0].end <= pair[1].start));
    }

    /// `TAKECOLOR` / `TAKEFX_NCH` / `TAKE_FX_HAVE_NEW_PDC_AUTOMATION_BEHAVIOR` 这类字段
    /// 虽然以 `TAKE` 开头，但不是整词 `TAKE` —— 它们不得切出一条新 take。
    #[test]
    fn take_prefixed_fields_are_not_markers() {
        let text = "<ITEM\nPOSITION 0\nTAKE_FX_HAVE_NEW_PDC_AUTOMATION_BEHAVIOR 1\nTAKECOLOR 255 1\nTAKEFX_NCH 2\nGUID {AAAAAAAA-1111-2222-3333-444444444444}\n<SOURCE WAVE\nFILE \"a.wav\"\n>\n>\n";
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(sections(&lines).unwrap().len(), 1);
        assert!(section_with_guid(&lines, TAKE_A).is_ok());
    }

    /// **一行自带闭合符的叶子块既不开也不关。**
    ///
    /// 【为什么值得单测】把它当"开"会让栈在文件末尾多出一层 ⇒ 整份 chunk 被判成"不完整"
    /// ⇒ 一个本来完好的 item 从此写不进去（音高转换/倒放全线失败）。RPPXML 的确允许这种写法。
    #[test]
    fn a_self_closing_line_keeps_the_item_complete() {
        let lines = vec![
            "<ITEM",
            "POSITION 0",
            "GUID {AAAAAAAA-1111-2222-3333-444444444444}",
            "<SOURCE WAVE",
            "FILE \"a.wav\"",
            ">",
            "<TAKEFX",
            "<VST \"VST3: HiFiShifter\" HiFiShifter.vst3 0 \"\" 12345{ABCD} com.hifishifter.ara.archive>",
            ">",
            ">",
        ];
        let sections = sections(&lines).unwrap();
        assert_eq!(sections.len(), 1);
        assert!(is_complete_item_state(&lines.join("\n")));
        assert!(block(&lines, &sections[0], "TAKEFX").is_some());
        // 自闭合行不影响同层字段的定位。
        assert_eq!(field(&lines, &sections[0], "GUID"), Some(2));
    }

    /// **半份 chunk 必须被判成"不完整"**：读侧据此扩容重试，而不是把宿主的好数据当坏数据。
    #[test]
    fn a_truncated_item_state_is_not_complete() {
        let truncated = "<ITEM\nPOSITION 0\nGUID {AAAAAAAA-1111-2222-3333-444444444444}\n<SOURCE WAVE\nFILE \"a.wav\"\n>\n";
        assert!(!is_complete_item_state(truncated));
        assert!(sections(&truncated.lines().collect::<Vec<_>>()).is_err());
        // 完整的那份是完整的。
        assert!(is_complete_item_state(&single().join("\n")));
        // ITEM 之后还有内容 ⇒ 不是一份 chunk。
        let trailing = format!("{}\n>", single().join("\n"));
        assert!(!is_complete_item_state(&trailing));
        // 没有 take 身份的 item 仍然是一份**完整**的 chunk：剪贴板复制不关心 take，
        // 判据服务的是"读没读完"，不是"能不能写包络"。
        assert!(is_complete_item_state("<ITEM\nPOSITION 0\nIGUID {A}\n>\n"));
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
