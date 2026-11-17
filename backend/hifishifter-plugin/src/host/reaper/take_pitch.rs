//! take 的音高包络（REAPER 的 PITCHENV）在 item state chunk 里的读写。
//!
//! ## 为什么走 chunk 而不是 Envelope* API
//!
//! 1. **创建**：\`GetTakeEnvelopeByName\` 在 take 还没有音高包络时返回空指针，而
//!    "新建一条音高包络"没有独立的创建入口 —— 只能靠写 state chunk。独立 App 的
//!    \`.rpp\` 导出早就这么做了（\`import::reaper_export::build_take_pitch_envelope\`），
//!    格式因此已经被往返测试钉住；
//! 2. **一致性**：倒放写回已经建立了"只改目标块、其余逐字节保留"的纪律，
//!    两条结构写回共用同一套解析；
//! 3. **少一束风险**：\`InsertEnvelopePoint\` / \`DeleteEnvelopePointRange\` /
//!    \`Envelope_SortPoints\` 的签名要逐版本核对，而 chunk 是纯文本。
//!
//! ## 作用范围：单 take item
//!
//! 与 \`source_direction\` 同一条边界：插件只绑定 active take，多 take item 上
//! "改哪一个 take 的包络"没有宿主无关的答案，因此明确拒绝而不是猜。

use super::item_chunk::MAX_ITEM_CHUNK_BYTES;
use hifishifter_kernel::pitch_bridge::EnvelopePoint;

/// 从 chunk 里读到的音高状态。
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TakePitch {
    /// 包络点（u 域，半音）。空 = 没有包络或包络没有点。
    pub points: Vec<EnvelopePoint>,
    /// chunk 里确实存在 \`<PITCHENV>\` 块。
    pub present: bool,
}

/// 读 take 的音高包络。
///
/// 【为什么读不到也算成功】"没有音高包络"是**常态**（绝大多数 take 都没有），
/// 调用方据此把宿主侧的值当作 0，而不是报错。
pub(super) fn read_take_pitch(chunk: &str, take_guid: &str) -> Result<TakePitch, String> {
    if chunk.is_empty() || chunk.len() > MAX_ITEM_CHUNK_BYTES || chunk.contains('\0') {
        return Err("invalid or oversized REAPER item state".into());
    }
    let lines: Vec<&str> = chunk.lines().collect();
    verify_take_identity(&lines, take_guid)?;
    let Some(block) = find_take_block(&lines, "PITCHENV") else {
        return Ok(TakePitch::default());
    };
    let mut points = Vec::new();
    for line in &lines[block.start + 1..block.end] {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.first() != Some(&"PT") || fields.len() < 3 {
            continue;
        }
        let time = fields[1]
            .parse::<f64>()
            .map_err(|_| "invalid PITCHENV point time")?;
        let value = fields[2]
            .parse::<f64>()
            .map_err(|_| "invalid PITCHENV point value")?;
        if !time.is_finite() || !value.is_finite() {
            continue;
        }
        points.push(EnvelopePoint::new(time, value));
    }
    points.sort_by(|a, b| a.time.total_cmp(&b.time));
    Ok(TakePitch {
        points,
        present: true,
    })
}

/// 把 take 的音高包络整块替换成 \`points\`；空列表 = 删除该块。
///
/// 【为什么整块替换而不是逐点插删】包络的读与写必须来自同一份数据（这条路径是
/// 一次性快照，不是实时绑定）。逐点增删要处理"旧点残留"，而残留点会让换算后的
/// 曲线比预期多几段 —— 用户听到的是"转换后音高不对"，却看不出多了一个点。
pub(super) fn rewrite_take_pitch(
    chunk: &str,
    take_guid: &str,
    points: &[EnvelopePoint],
) -> Result<String, String> {
    if chunk.is_empty() || chunk.len() > MAX_ITEM_CHUNK_BYTES || chunk.contains('\0') {
        return Err("invalid or oversized REAPER item state".into());
    }
    if points
        .iter()
        .any(|point| !point.time.is_finite() || !point.value.is_finite())
    {
        return Err("nonfinite pitch envelope point".into());
    }
    let lines: Vec<&str> = chunk.lines().collect();
    verify_take_identity(&lines, take_guid)?;
    let existing = find_take_block(&lines, "PITCHENV");
    let source = outer_source_block(&lines)?;
    let indent = indent_of(lines[source.start]).to_string();

    let mut output: Vec<String> = Vec::with_capacity(lines.len() + points.len() + 8);
    let mut index = 0;
    while index < lines.len() {
        if let Some(block) = &existing {
            if index == block.start {
                push_envelope_block(&mut output, &indent, points);
                index = block.end + 1;
                continue;
            }
        }
        output.push(lines[index].to_string());
        // 没有旧块时，插在 take 的 SOURCE 之后 —— 那里是 take 字段区的末尾，
        // 且仍在外层 ITEM 之内（单 take item 的 take 字段就挂在 ITEM 下）。
        if existing.is_none() && index == source.end {
            push_envelope_block(&mut output, &indent, points);
        }
        index += 1;
    }
    let text = output.join("\n") + "\n";
    if text.len() > MAX_ITEM_CHUNK_BYTES {
        return Err("rewritten item state budget exceeded".into());
    }
    Ok(text)
}

/// 写一块 \`<PITCHENV>\`；\`points\` 为空时什么都不写（等价于删除）。
fn push_envelope_block(out: &mut Vec<String>, indent: &str, points: &[EnvelopePoint]) {
    if points.is_empty() {
        return;
    }
    // 显示范围按最大偏移取整（最少 ±3 半音），与独立 App 的导出同一口径 ——
    // 范围太小会把曲线画在框外，用户以为"没写进去"。
    let max_abs = points
        .iter()
        .map(|point| point.value.abs())
        .fold(0.0_f64, f64::max);
    let display_range = (max_abs.ceil() as i64).clamp(3, 96);
    out.push(format!("{indent}<PITCHENV"));
    out.push(format!("{indent}  ACT 1 -1"));
    out.push(format!("{indent}  VIS 1 1 1"));
    out.push(format!("{indent}  LANEHEIGHT 0 0"));
    out.push(format!("{indent}  ARM 0"));
    out.push(format!("{indent}  DEFSHAPE 0 {display_range} -1"));
    for point in points {
        out.push(format!(
            "{indent}  PT {:.17} {:.17} 0",
            point.time, point.value
        ));
    }
    out.push(format!("{indent}>"));
}

struct Block {
    start: usize,
    end: usize,
}

/// take 字段区里名为 \`kind\` 的块（\`<PITCHENV>\`）。
fn find_take_block(lines: &[&str], kind: &str) -> Option<Block> {
    let mut stack: Vec<&str> = Vec::new();
    let mut open: Option<usize> = None;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('<') {
            let name = rest.split_whitespace().next().unwrap_or("");
            if stack.len() == 1 && name == kind {
                open = Some(index);
            }
            stack.push(name);
        } else if trimmed == ">" {
            stack.pop();
            if stack.len() == 1 {
                if let Some(start) = open.take() {
                    return Some(Block { start, end: index });
                }
            }
        }
    }
    None
}

/// take 的源块（与 \`source_direction\` 同一判据：直接挂在 ITEM 上的 SOURCE）。
fn outer_source_block(lines: &[&str]) -> Result<Block, String> {
    let mut stack: Vec<&str> = Vec::new();
    let mut open: Option<usize> = None;
    let mut found: Option<Block> = None;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('<') {
            let name = rest.split_whitespace().next().unwrap_or("");
            if stack.len() == 1 && name == "SOURCE" {
                // 第二条外层 SOURCE 就是第二个 take：本条路径只处理单 take item。
                if found.is_some() || open.is_some() {
                    return Err("the host item has more than one take".into());
                }
                open = Some(index);
            }
            stack.push(name);
        } else if trimmed == ">" {
            stack.pop();
            if stack.len() == 1 {
                if let Some(start) = open.take() {
                    found = Some(Block { start, end: index });
                }
            }
        }
    }
    if !stack.is_empty() {
        return Err("unbalanced item state".into());
    }
    found.ok_or_else(|| "the host item has no take source".to_string())
}

/// 深度 1 上的 \`GUID <take_guid>\`。
fn verify_take_identity(lines: &[&str], take_guid: &str) -> Result<(), String> {
    if take_guid.is_empty() {
        return Err("host take identity missing".into());
    }
    let mut stack: Vec<&str> = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('<') {
            stack.push(rest.split_whitespace().next().unwrap_or(""));
            continue;
        }
        if trimmed == ">" {
            stack.pop();
            continue;
        }
        if stack.len() == 1
            && trimmed.split_whitespace().next() == Some("GUID")
            && trimmed.split_whitespace().nth(1) == Some(take_guid)
        {
            return Ok(());
        }
    }
    Err("host take identity changed before the pitch write".into())
}

fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAKE: &str = "{AAAAAAAA-1111-2222-3333-444444444444}";

    fn chunk() -> String {
        format!(
            "<ITEM\nPOSITION 0\nLENGTH 1\nIGUID {{BBBBBBBB-1111-2222-3333-444444444444}}\nSOFFS 0\nGUID {TAKE}\n<SOURCE WAVE\nFILE \"a.wav\"\n>\n>\n"
        )
    }

    /// 没有包络是常态：读成空、且 \`present == false\`。
    #[test]
    fn a_take_without_an_envelope_reads_as_absent() {
        let pitch = read_take_pitch(&chunk(), TAKE).unwrap();
        assert!(!pitch.present);
        assert!(pitch.points.is_empty());
    }

    /// 写入 → 读回：点与值逐位一致，且块前的原文一字不改。
    #[test]
    fn writing_and_reading_the_envelope_round_trips() {
        let points = [
            EnvelopePoint::new(0.0, 0.0),
            EnvelopePoint::new(0.5, 2.0),
            EnvelopePoint::new(1.0, 2.0),
        ];
        let text = rewrite_take_pitch(&chunk(), TAKE, &points).unwrap();
        assert!(text.contains("<PITCHENV"));
        assert!(text.contains("DEFSHAPE 0 3 -1"), "{text}");
        assert!(text.contains("PT 0.50000000000000000 2.00000000000000000 0"));
        let read = read_take_pitch(&text, TAKE).unwrap();
        assert!(read.present);
        assert_eq!(read.points, points);
        // 原有字段逐字保留。
        assert!(text.contains("SOFFS 0"));
        assert!(text.contains("FILE \"a.wav\""));
    }

    /// 空列表 = 删除整块，回到没有包络的形态。
    #[test]
    fn writing_an_empty_envelope_removes_the_block() {
        let points = [EnvelopePoint::new(0.0, 1.0)];
        let with = rewrite_take_pitch(&chunk(), TAKE, &points).unwrap();
        let without = rewrite_take_pitch(&with, TAKE, &[]).unwrap();
        assert!(!without.contains("PITCHENV"));
        assert!(!read_take_pitch(&without, TAKE).unwrap().present);
    }

    /// 身份不符 / 多 take / 非有限值都必须拒绝。
    #[test]
    fn refuses_identity_mismatch_multi_take_and_nonfinite_points() {
        // 两个 take：两条**深度 1** 的 SOURCE 块（第二条挂在 TAKE 标记之后）。
        let two_takes = format!(
            "<ITEM\nPOSITION 0\nLENGTH 1\nGUID {TAKE}\n<SOURCE WAVE\nFILE \"a.wav\"\n>\nTAKE\nGUID {{DDDDDDDD-1111-2222-3333-444444444444}}\n<SOURCE WAVE\nFILE \"b.wav\"\n>\n>\n"
        );
        let points = [EnvelopePoint::new(0.0, 1.0)];
        assert!(
            rewrite_take_pitch(&chunk(), "{EEEEEEEE-1111-2222-3333-444444444444}", &points)
                .is_err()
        );
        assert!(rewrite_take_pitch(&two_takes, TAKE, &points).is_err());
        assert!(rewrite_take_pitch(&chunk(), TAKE, &[EnvelopePoint::new(f64::NAN, 1.0)]).is_err());
    }
}
