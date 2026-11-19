//! take 的音高包络（REAPER 的 PITCHENV）在 item state chunk 里的读写。
//!
//! ## 为什么走 chunk 而不是 Envelope* API
//!
//! 1. **创建**：`GetTakeEnvelopeByName` 在 take 还没有音高包络时返回空指针，而
//!    "新建一条音高包络"没有独立的创建入口 —— 只能靠写 state chunk。独立 App 的
//!    `.rpp` 导出早就这么做了（`import::reaper_export::build_take_pitch_envelope`），
//!    格式因此已经被往返测试钉住；
//! 2. **一致性**：倒放写回已经建立了"只改目标块、其余逐字节保留"的纪律，
//!    两条结构写回共用同一套解析；
//! 3. **少一束风险**：`InsertEnvelopePoint` / `DeleteEnvelopePointRange` /
//!    `Envelope_SortPoints` 的签名要逐版本核对，而 chunk 是纯文本。
//!
//! ## 作用范围：**按 take GUID 定位的那一条**
//!
//! 两条结构写回共用 [`super::take_chunk`] 的解析：单 take 与多 take 是同一套（多 take
//! 只是在每条 take 前多一行 `TAKE` 标记）。此前这里只认"take 字段直接挂在 ITEM 下"的
//! 单 take 形态，多 take 一律拒绝 —— 而 take FX 最常挂的正是多 take item 的某一条。
//! 现在的判据是"GUID 相等的那一条"，找不到唯一匹配仍然拒绝（不猜）。
//!
//! ## 坐标域：播放时间，不含源起点
//!
//! `EnvelopePoint::time` 就是 chunk 里 `PT` 的第一个字段，属于 take 的**播放**时间轴
//! （u = 0 在入点，按 `PLAYRATE` 缩放）。`SOFFS`、SECTION 的 `STARTPOS`、倒放位都不
//! 进入它 —— 真机渲染实测见 `docs/2026-11-17-take-envelope-axis-findings.md`。因此本
//! 模块既不读也不写 `SOFFS`：整块替换时其余字段逐字节保留，**不要**顺手补上源起点。

use super::item_chunk::MAX_ITEM_CHUNK_BYTES;
use super::take_chunk;
use hifishifter_kernel::pitch_bridge::EnvelopePoint;

/// 从 chunk 里读到的音高状态。
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TakePitch {
    /// 包络点（u 域，半音）。空 = 没有包络或包络没有点。
    pub points: Vec<EnvelopePoint>,
    /// chunk 里确实存在 `<PITCHENV>` 块。
    pub present: bool,
}

/// 读 take 的音高包络。
///
/// 【为什么读不到也算成功】"没有音高包络"是**常态**（绝大多数 take 都没有），
/// 调用方据此把宿主侧的值当作 0，而不是报错。
pub(super) fn read_take_pitch(chunk: &str, take_guid: &str) -> Result<TakePitch, String> {
    take_chunk::check_budget(chunk)?;
    let lines: Vec<&str> = chunk.lines().collect();
    let section = take_chunk::section_with_guid(&lines, take_guid)?;
    let Some(block) = take_chunk::block(&lines, &section, "PITCHENV") else {
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

/// 把 take 的音高包络整块替换成 `points`；空列表 = 删除该块。
///
/// 【为什么整块替换而不是逐点插删】包络的读与写必须来自同一份数据（这条路径是
/// 一次性快照，不是实时绑定）。逐点增删要处理"旧点残留"，而残留点会让换算后的
/// 曲线比预期多几段 —— 用户听到的是"转换后音高不对"，却看不出多了一个点。
pub(super) fn rewrite_take_pitch(
    chunk: &str,
    take_guid: &str,
    points: &[EnvelopePoint],
) -> Result<String, String> {
    take_chunk::check_budget(chunk)?;
    if points
        .iter()
        .any(|point| !point.time.is_finite() || !point.value.is_finite())
    {
        return Err("nonfinite pitch envelope point".into());
    }
    let lines: Vec<&str> = chunk.lines().collect();
    let section = take_chunk::section_with_guid(&lines, take_guid)?;
    let existing = take_chunk::block(&lines, &section, "PITCHENV");
    let source = take_chunk::block(&lines, &section, "SOURCE")
        .ok_or_else(|| "the host take has no source".to_string())?;
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
        // 没有旧块时，插在**本条 take** 的 SOURCE 之后 —— 那里是这条 take 字段区的末尾，
        // 而且仍在它自己的字段区之内（多 take item 下不会串到隔壁那条）。
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

/// 从一份**包络自己的** chunk（`GetEnvelopeStateChunk`）里读点。
///
/// 【为什么与 `read_take_pitch` 分开】那份要先在 item chunk 里按 GUID 找到"这一条 take 的
/// 字段区"；包络对象给的本来就是一块 `<PITCHENV>`，没有 take 上下文可找。两份都只做
/// "把 `PT` 行读成点"，不变量相同，输入不同。
pub(super) fn read_block(text: &str) -> Result<TakePitch, String> {
    take_chunk::check_budget(text)?;
    if !text.contains("<PITCHENV") {
        return Ok(TakePitch::default());
    }
    let mut points = Vec::new();
    for line in text.lines() {
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

/// 交给 `SetEnvelopeStateChunk` 的整块文本（无缩进）。
///
/// 【为什么空点集也返回一个**只有头**的块】真机实测：把只有 `ACT/VIS/…` 的 `<PITCHENV>`
/// 交给宿主，宿主会把整条包络删掉 —— 那正是"清空一条曲线"的语义。item chunk 那条路相反
/// （空块会被宿主丢弃，所以那边靠"整块不写"表达删除）。
pub(super) fn block_text(points: &[EnvelopePoint]) -> String {
    block_lines("", points).join("\n") + "\n"
}

/// 写一块 `<PITCHENV>`；`points` 为空时什么都不写（等价于删除，用于 item chunk）。
fn push_envelope_block(out: &mut Vec<String>, indent: &str, points: &[EnvelopePoint]) {
    if points.is_empty() {
        return;
    }
    out.extend(block_lines(indent, points));
}

/// 一块 `<PITCHENV>` 的全部行。
fn block_lines(indent: &str, points: &[EnvelopePoint]) -> Vec<String> {
    let mut out = Vec::with_capacity(points.len() + 8);
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
    out
}

fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAKE: &str = "{AAAAAAAA-1111-2222-3333-444444444444}";
    const SIBLING: &str = "{DDDDDDDD-1111-2222-3333-444444444444}";

    fn chunk() -> String {
        format!(
            "<ITEM\nPOSITION 0\nLENGTH 1\nIGUID {{BBBBBBBB-1111-2222-3333-444444444444}}\nSOFFS 0\nGUID {TAKE}\n<SOURCE WAVE\nFILE \"a.wav\"\n>\n>\n"
        )
    }

    /// 多 take item：本 take 在前、兄弟 take 在后（各自带自己的 SOURCE 与包络）。
    fn multi_take_chunk() -> String {
        format!(
            "<ITEM\nPOSITION 0\nLENGTH 1\nIGUID {{BBBBBBBB-1111-2222-3333-444444444444}}\n\
             SOFFS 0\nGUID {TAKE}\n<SOURCE WAVE\nFILE \"a.wav\"\n>\n\
             TAKE\nGUID {SIBLING}\nSOFFS 3\n<PITCHENV\n  PT 0 5 0\n>\n\
             <SOURCE WAVE\nFILE \"b.wav\"\n>\n>\n"
        )
    }

    /// 没有包络是常态：读成空、且 present 为假。
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

    /// 【u 域不含源起点/倒放镜像】源起点 30 s、带倒放位的 take：写进去的 PT
    /// 必须原样是 0 / 1。补一个 SOFFS 或镜像一次，整条曲线就整体平移 ——
    /// 真机实测见 `docs/2026-11-17-take-envelope-axis-findings.md`。
    #[test]
    fn the_envelope_is_written_in_playback_time_without_the_source_offset() {
        let shifted = format!(
            "<ITEM\nPOSITION 0\nLENGTH 1\nSOFFS 30\nGUID {TAKE}\n<SOURCE SECTION\nLENGTH 1\nMODE 3\nSTARTPOS 0\n<SOURCE WAVE\nFILE \"a.wav\"\n>\n>\n>\n"
        );
        let points = [EnvelopePoint::new(0.0, 0.0), EnvelopePoint::new(1.0, 2.0)];
        let text = rewrite_take_pitch(&shifted, TAKE, &points).unwrap();
        assert!(
            text.contains("PT 0.00000000000000000 0.00000000000000000 0"),
            "{text}"
        );
        assert!(
            text.contains("PT 1.00000000000000000 2.00000000000000000 0"),
            "{text}"
        );
        assert!(!text.contains("PT 30"), "源起点不得进入包络坐标：{text}");
        assert!(text.contains("SOFFS 30"), "源起点必须逐字保留：{text}");
        assert_eq!(read_take_pitch(&text, TAKE).unwrap().points, points);
    }

    /// **多 take item：只改 GUID 匹配的那一条，兄弟 take 的包络一字不动。**
    ///
    /// 【为什么这条是本轮的核心回归】take FX 最常挂的正是多 take item 的某一条。
    /// 此前整条路径直接返回 "the host item has more than one take"，用户看到的是
    /// "转换成功、音高没变"（或干脆报错）。
    #[test]
    fn a_multi_take_item_rewrites_only_the_take_with_the_matching_guid() {
        let points = [EnvelopePoint::new(0.0, 0.0), EnvelopePoint::new(1.0, 12.0)];
        let text = rewrite_take_pitch(&multi_take_chunk(), TAKE, &points).unwrap();
        // 本 take 的包络写进去了。
        assert_eq!(read_take_pitch(&text, TAKE).unwrap().points, points);
        // 兄弟 take 的包络原样保留（PT 0 5 0 仍在，且仍属于它自己）。
        let sibling = read_take_pitch(&text, SIBLING).unwrap();
        assert!(sibling.present);
        assert_eq!(sibling.points, [EnvelopePoint::new(0.0, 5.0)]);
        // 源文件与 SOFFS 都没有被换位。
        assert!(text.contains("FILE \"a.wav\""));
        assert!(text.contains("FILE \"b.wav\""));
        assert!(text.contains("SOFFS 3"));
        // 本 take 的包络插在它自己的 SOURCE 之后，而不是兄弟 take 的 SOURCE 之后。
        let inserted = text.find("PT 1.00000000000000000").unwrap();
        let sibling_source = text.find("FILE \"b.wav\"").unwrap();
        assert!(inserted < sibling_source, "{text}");
    }

    /// 身份不符 / 非有限值都必须拒绝。
    #[test]
    fn refuses_identity_mismatch_and_nonfinite_points() {
        let points = [EnvelopePoint::new(0.0, 1.0)];
        assert!(
            rewrite_take_pitch(&chunk(), "{EEEEEEEE-1111-2222-3333-444444444444}", &points)
                .is_err()
        );
        assert!(rewrite_take_pitch(&chunk(), TAKE, &[EnvelopePoint::new(f64::NAN, 1.0)]).is_err());
    }
}
