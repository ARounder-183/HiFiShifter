//! 把 take 的**播放方向**写回 REAPER —— 它没有"反向 setter"，倒放是结构。
//!
//! REAPER 表达倒放的方式是 `<SOURCE SECTION MODE n>` 这一层**包装**：一个正向
//! `<SOURCE WAVE>` 被放进段里，`MODE` 置反向位后这段反向播放，`PLAYRATE` 保持
//! 正数。编码细节见 [`SECTION_MODE_REVERSED`]（真机取值来自 REAPER 7.85 的
//! `Item properties: Toggle take reverse`，不是推断）。
//!
//! ## 为什么单独一个模块
//!
//! `host_edit` 是"逐字段 setter"的世界：它只写 `D_POSITION` / `D_STARTOFFS` 这类
//! 数值，写不了结构。`item_chunk` 会改写结构，但它的职责是"换身份"（复制粘贴）。
//! 把"换方向"单独放开，是因为它需要**逐字节保留**除目标 take 之外的一切 —— 未知块、
//! 用户插件的不透明状态、别的 take。任何"重新序列化整份 chunk"的做法都会在不报错的
//! 情况下丢掉它们。
//!
//! ## 作用范围：单 take item
//!
//! 插件与宿主 item 的唯一绑定是 `active take` 那一条通道（`HostClipTarget` 冻结的就是它）。
//! 多 take item 上"改哪一个 take 的方向"没有宿主无关的答案，因此这里**明确拒绝**，
//! 而不是猜一个。拒绝是可读的错误，猜错是工程损坏。

use super::item_chunk::MAX_ITEM_CHUNK_BYTES;

/// REAPER 里"这一段反向播放"的 `MODE` 取值。
///
/// 【真机取证（REAPER 7.85）】对一个普通 audio item 执行宿主自带的
/// `Item properties: Toggle take reverse`（命令号 41051），它写出的是：
///
/// ```text
/// SOFFS 26.29858333333333          <- D − 正向窗口终点
/// <SOURCE SECTION
///   LENGTH 3                       <- 正向窗口跨度
///   MODE 3
///   STARTPOS 0
///   OVERLAP 0.01
///   <SOURCE WAVE ... >
/// >
/// ```
///
/// 而 `PCM_Source_GetSectionInfo` 对四种 `MODE` 的读数（同一份 chunk、段
/// `STARTPOS 5 / LENGTH 3`）是：
///
/// | MODE | offs | len | rev |
/// |------|------|-----|-----|
/// | 0 | 5 | 3 | false |
/// | 1 | 0 | 媒体全长 | false |
/// | 2 | 5 | 3 | true |
/// | 3 | 0 | 媒体全长 | true |
///
/// 即 `MODE` 是**位**：`1` = "段边界无效、按整份媒体算"，`2` = 反向。
/// 此前这里写 `1`，宿主读回来的是一个**正放**的整媒体段 —— "写→读→比"永远
/// 不符，用户看到的是"倒放点了没反应"（现场日志：`the host did not apply the
/// requested take direction`）。
///
/// 因此倒放统一写成宿主自己的编码：`MODE 3`（整份媒体 + 反向）+ 镜像 `SOFFS`。
/// 读取侧（`take_geometry`）本来就是"轴 = 整份媒体、SOFFS = 镜像量"的解算，
/// 两个方向就此对齐；真机往返（写→读）已按 `MODE 3` 复核过。
pub(super) const SECTION_MODE_REVERSED: i32 = 3;

/// 反向段与源的默认重叠量（秒）。REAPER 自己写 SECTION 时也带这一行；
/// 缺省值由 REAPER 的 item 属性决定，显式写出来才能让往返不依赖它。
pub(super) const SECTION_OVERLAP_SEC: f64 = 0.01;

/// 目标方向在宿主文本里的完整描述。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DirectionEdit {
    /// `None` = 取消倒放（拆掉 SECTION 包装，回到裸 `<SOURCE>`）。
    pub section: Option<SectionSpec>,
    /// take 的 `SOFFS`（正向域：正放是窗口起点；倒放且无段时是镜像量）。
    pub s_offs: f64,
}

impl DirectionEdit {
    /// 正向：拆掉 SECTION，SOFFS 回到正向窗口起点。
    pub(super) fn forward(s_offs: f64) -> Self {
        Self {
            section: None,
            s_offs,
        }
    }

    /// 反向：把源包进一个反向段。
    ///
    /// `length_sec` 是**正向窗口跨度**（写进 SECTION 的 `LENGTH`，宿主自己的
    /// 写法就是跨度而不是起点），`s_offs` 是镜像起点 `D − 正向窗口终点`。
    /// `STARTPOS` 恒写 0：`MODE 3` 下宿主不拿它解算，写 0 与宿主自己的输出一致。
    pub(super) fn reversed(length_sec: f64, s_offs: f64) -> Self {
        Self {
            section: Some(SectionSpec {
                start_sec: 0.0,
                length_sec,
                mode: SECTION_MODE_REVERSED,
                overlap_sec: SECTION_OVERLAP_SEC,
            }),
            s_offs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SectionSpec {
    /// `STARTPOS`（秒）。倒放编码里恒为 0（见 [`DirectionEdit::reversed`]）。
    pub start_sec: f64,
    /// `LENGTH`（秒）：正向窗口跨度。
    pub length_sec: f64,
    /// `MODE` 位（见 [`SECTION_MODE_REVERSED`]）。
    pub mode: i32,
    /// `OVERLAP`（秒）：反相段与源的默认重叠量，宿主自己也写 0.01。
    pub overlap_sec: f64,
}

impl SectionSpec {
    fn is_valid(&self) -> bool {
        self.start_sec.is_finite()
            && self.length_sec.is_finite()
            && self.length_sec > 0.0
            && self.mode >= 0
            && self.overlap_sec.is_finite()
    }
}

/// 改写一份 item state chunk 里**唯一那条 take** 的方向。
///
/// 只动三处：目标 take 的 `<SOURCE>` 子树、它的 `SOFFS`、以及（倒放时）新加的
/// SECTION 头。其余行**逐字节原样输出** —— 与 `item_chunk::rewrite_item` 同一条纪律：
/// 不认识的块不解析、不重排、不格式化。
///
/// `take_guid` 只用于**校验**：chunk 里必须真的有一条 GUID 等于它的 take，
/// 否则说明我们拿到的不是以为的那份 chunk（宿主在两次调用之间换了 item），
/// 此时宁可不写。
pub(super) fn rewrite_take_direction(
    chunk: &str,
    take_guid: &str,
    edit: &DirectionEdit,
) -> Result<String, String> {
    if chunk.is_empty() || chunk.len() > MAX_ITEM_CHUNK_BYTES || chunk.contains('\0') {
        return Err("invalid or oversized REAPER item state".into());
    }
    if !edit.s_offs.is_finite() {
        return Err("nonfinite take start offset".into());
    }
    if let Some(section) = edit.section {
        if !section.is_valid() {
            return Err("invalid reverse section".into());
        }
    }
    let lines: Vec<&str> = chunk.lines().collect();
    let blocks = outer_source_blocks(&lines)?;
    if blocks.len() != 1 {
        return Err(format!(
            "reversing is only supported for single-take items (found {} takes)",
            blocks.len()
        ));
    }
    let block = &blocks[0];
    if !has_take_guid(&lines, take_guid) {
        return Err("host take identity changed before the reverse write".into());
    }
    let soffs = find_root_field(&lines, "SOFFS")
        .ok_or("host take state has no SOFFS line; cannot write the direction")?;

    let base_indent = indent_of(lines[block.start]);
    // SOFFS 保留**它自己**的缩进：段块的缩进属于源块，两者在真实 chunk 里不一定相同
    // （段可能被额外缩进过）。用源块的缩进改写它会制造一次无意义的格式漂移。
    let soffs_indent = indent_of(lines[soffs]);
    let mut output: Vec<String> = Vec::with_capacity(lines.len() + 8);
    let mut index = 0;
    while index < lines.len() {
        if index == soffs {
            output.push(format!(
                "{soffs_indent}SOFFS {}",
                format_number(edit.s_offs)
            ));
            index += 1;
            continue;
        }
        if index == block.start {
            match edit.section {
                Some(section) => {
                    // 段头：REAPER 的 SECTION 是"源的一段"，内层仍是原来的 SOURCE。
                    let inner = inner_source_lines(&lines, block);
                    output.push(format!("{base_indent}<SOURCE SECTION"));
                    let field_indent = format!("{base_indent}  ");
                    output.push(format!(
                        "{field_indent}LENGTH {}",
                        format_number(section.length_sec)
                    ));
                    output.push(format!("{field_indent}MODE {}", section.mode));
                    output.push(format!(
                        "{field_indent}STARTPOS {}",
                        format_number(section.start_sec)
                    ));
                    output.push(format!(
                        "{field_indent}OVERLAP {}",
                        format_number(section.overlap_sec)
                    ));
                    // 【为什么内层不加缩进】整份改写的纪律是"除目标外逐字节原样"。
                    // 加缩进会让"倒放→取消倒放"的往返在文本上漂两格，而 REAPER 对
                    // 缩进不敏感 —— 于是白拿一次 diff 噪声，换不到任何可读性。
                    for line in inner {
                        output.push(line.to_string());
                    }
                    output.push(format!("{base_indent}>"));
                }
                None => {
                    // 取消倒放：把 SECTION 的内层原样提出来（若本来就不是 SECTION，
                    // 这段就是原文本身 —— 幂等）。
                    for line in inner_source_lines(&lines, block) {
                        output.push(line.to_string());
                    }
                }
            }
            index = block.end + 1;
            continue;
        }
        output.push(lines[index].to_string());
        index += 1;
    }
    let text = output.join("\n") + "\n";
    if text.len() > MAX_ITEM_CHUNK_BYTES {
        return Err("rewritten item state budget exceeded".into());
    }
    Ok(text)
}

/// 一条最外层（深度 1，直接挂在 ITEM 上）的 `<SOURCE ...>` 块。
struct SourceBlock {
    /// `<SOURCE ...` 那一行的下标。
    start: usize,
    /// 与之配对的 `>` 行的下标。
    end: usize,
    /// 块类型（WAVE / SECTION / MIDI / VIDEO …）。
    kind: String,
}

/// 找出所有"直接挂在 ITEM 上"的 SOURCE 块 —— 也就是 take 的源。
///
/// 【为什么按块数判 take 数】REAPER 的 chunk 对单 take item 直接把 take 的字段写在
/// ITEM 下面，多 take 才有 `TAKE` 标记行。用"外层 SOURCE 块的数量"判 take 数，
/// 同时覆盖两种形态，也不会把 `TAKEFX` 里的东西算进来。
fn outer_source_blocks(lines: &[&str]) -> Result<Vec<SourceBlock>, String> {
    let mut blocks = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut current: Option<SourceBlock> = None;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(open) = trimmed.strip_prefix('<') {
            let kind = open
                .split_whitespace()
                .next()
                .ok_or("invalid item state block")?
                .to_string();
            if stack.len() >= 64 {
                return Err("item state nesting budget exceeded".into());
            }
            if stack.len() == 1 && kind == "SOURCE" && current.is_none() {
                current = Some(SourceBlock {
                    start: index,
                    end: index,
                    kind: open.split_whitespace().nth(1).unwrap_or("").to_string(),
                });
            }
            stack.push(kind);
        } else if trimmed == ">" {
            if stack.pop().is_none() {
                return Err("unbalanced item state closing block".into());
            }
            if stack.len() == 1 {
                if let Some(block) = current.as_mut() {
                    block.end = index;
                    blocks.push(SourceBlock {
                        start: block.start,
                        end: block.end,
                        kind: block.kind.clone(),
                    });
                    current = None;
                }
            }
        }
    }
    if !stack.is_empty() {
        return Err("unbalanced item state".into());
    }
    Ok(blocks)
}

/// 段块的内层行：去掉 SECTION 自己的 LENGTH / MODE / STARTPOS / OVERLAP，
/// 保留真正描述源的那些（`<SOURCE WAVE>` / FILE / GUID …）。
fn inner_source_lines<'a>(lines: &[&'a str], block: &SourceBlock) -> Vec<&'a str> {
    if block.kind != "SECTION" {
        return lines[block.start..=block.end].to_vec();
    }
    let mut out = Vec::new();
    let mut depth = 0usize;
    for line in &lines[block.start + 1..block.end] {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('<') {
            depth += 1;
        } else if trimmed == ">" {
            depth = depth.saturating_sub(1);
        } else if depth == 0 {
            let key = trimmed.split_whitespace().next().unwrap_or("");
            if matches!(key, "LENGTH" | "MODE" | "STARTPOS" | "OVERLAP") {
                continue;
            }
        }
        out.push(*line);
    }
    out
}

/// 深度 1 上的字段行（`SOFFS` / `GUID` / `POSITION` …）。
fn find_root_field(lines: &[&str], field: &str) -> Option<usize> {
    let mut stack: Vec<&str> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(open) = trimmed.strip_prefix('<') {
            stack.push(open.split_whitespace().next().unwrap_or(""));
            continue;
        }
        if trimmed == ">" {
            stack.pop();
            continue;
        }
        if stack.len() == 1 && trimmed.split_whitespace().next() == Some(field) {
            return Some(index);
        }
    }
    None
}

/// take 身份校验：深度 1 上有一条 `GUID &lt;take_guid&gt;`。
fn has_take_guid(lines: &[&str], take_guid: &str) -> bool {
    if take_guid.is_empty() {
        return false;
    }
    let Some(index) = find_root_field(lines, "GUID") else {
        return false;
    };
    lines[index].split_whitespace().nth(1) == Some(take_guid)
}

fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// 数字格式化：`%.17g` 等价形态，保证往返逐位一致。
fn format_number(value: f64) -> String {
    format!("{value:.17}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAKE: &str = "{AAAAAAAA-1111-2222-3333-444444444444}";

    fn chunk() -> String {
        format!(
            "<ITEM\nPOSITION 2.0\nLENGTH 4.0\nSEL 1\nIGUID {{BBBBBBBB-1111-2222-3333-444444444444}}\n\
             SOFFS 1.5\nPLAYRATE 1 1 0\nGUID {TAKE}\n\
             <SOURCE WAVE\nFILE \"C:/audio/take.wav\"\nGUID {{CCCCCCCC-1111-2222-3333-444444444444}}\n>\n>\n"
        )
    }

    /// 倒放：只换上 SECTION 头与 SOFFS，别的行逐字节不变。
    #[test]
    fn reversing_wraps_the_source_and_rewrites_only_soffs() {
        let rewrite = rewrite_take_direction(&chunk(), TAKE, &DirectionEdit::reversed(2.5, 0.0))
            .expect("reverse write");
        assert!(rewrite.contains("<SOURCE SECTION\n"));
        assert!(rewrite.contains("LENGTH 2.50000000000000000\n"));
        assert!(rewrite.contains("MODE 3\n"));
        assert!(rewrite.contains("STARTPOS 0.00000000000000000\n"));
        assert!(rewrite.contains("SOFFS 0.00000000000000000\n"));
        // 源本身仍然是那一条，文件路径与 GUID 原样保留。
        assert!(rewrite.contains("<SOURCE WAVE"));
        assert!(rewrite.contains("C:/audio/take.wav"));
        assert!(rewrite.contains("POSITION 2.0"));
        assert!(rewrite.contains("PLAYRATE 1 1 0"));
    }

    /// 取消倒放：拆掉 SECTION，SOFFS 回到正向起点；重复调用幂等。
    #[test]
    fn un_reversing_unwraps_the_section_idempotently() {
        let reversed =
            rewrite_take_direction(&chunk(), TAKE, &DirectionEdit::reversed(2.5, 0.0)).unwrap();
        let forward =
            rewrite_take_direction(&reversed, TAKE, &DirectionEdit::forward(1.5)).unwrap();
        assert!(!forward.contains("SOURCE SECTION"));
        assert!(forward.contains("SOFFS 1.50000000000000000"));
        assert!(forward.contains("<SOURCE WAVE"));
        let again = rewrite_take_direction(&forward, TAKE, &DirectionEdit::forward(1.5)).unwrap();
        assert_eq!(again, forward);
    }

    /// 身份不符、多 take、非有限值、坏结构都必须拒绝 —— 宁可"没改"。
    #[test]
    fn refuses_identity_mismatch_multi_take_and_malformed_state() {
        let two_takes = format!(
            "{}TAKE\nGUID {{DDDDDDDD-1111-2222-3333-444444444444}}\n<SOURCE WAVE\nFILE \"b.wav\"\n>\n>\n",
            chunk().trim_end_matches('>').trim_end()
        );
        for (label, chunk, guid, edit) in [
            (
                "identity",
                chunk(),
                "{EEEEEEEE-1111-2222-3333-444444444444}",
                DirectionEdit::reversed(1.0, 0.0),
            ),
            (
                "multi-take",
                two_takes,
                TAKE,
                DirectionEdit::reversed(1.0, 0.0),
            ),
            ("nan", chunk(), TAKE, DirectionEdit::forward(f64::NAN)),
            (
                "bad-section",
                chunk(),
                TAKE,
                DirectionEdit::reversed(0.0, 0.0),
            ),
            (
                "truncated",
                "<ITEM\nPOSITION 0\n".to_string(),
                TAKE,
                DirectionEdit::forward(0.0),
            ),
        ] {
            assert!(
                rewrite_take_direction(&chunk, guid, &edit).is_err(),
                "{label} 必须被拒绝"
            );
        }
    }
}
