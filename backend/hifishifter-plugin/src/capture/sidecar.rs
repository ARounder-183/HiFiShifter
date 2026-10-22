//! Capture 音频的落盘：sidecar 文件。
//!
//! # 为什么不能嵌进组件 state
//!
//! `IComponent::getState` 的载荷是宿主存档的一部分，会被写进每一次保存、撤销快照与
//! 模板。而组件 state 的预算是 64 MiB（`MAX_COMPONENT_STATE`），4 分钟立体声 96 kHz 已是
//! 184 MB —— 是上限的三倍。**即使放得下也不该放**：把上百 MB 音频塞进宿主存档，
//! 等于让用户的工程文件与撤销历史一起爆炸。
//!
//! 所以：**PCM 落 sidecar 文件，组件 state 只带清单**（take 号、段几何、文件路径）。
//! 清单是几 KB，撞不到任何上限。
//!
//! # 文件格式
//!
//! ```text
//! magic    8 字节  "HFSXFER1"
//! version  u32     = 2（v1 仍可读）
//! segments u32
//! 每段:
//!   origin   i64    工程采样起点
//!   rate     u32    采样率
//!   channels u32
//!   frames   u64
//!   loop     i64    循环起点（`i64::MIN` = 非循环段；仅 v2）
//!   samples  frames × channels × f32（小端，平面顺序）
//! digest   32 字节 BLAKE3（覆盖以上全部内容）
//! ```
//!
//! 【为什么要有尾部摘要】截断的写入（磁盘满、进程被杀）会留下一个"看起来合法、
//! 读出来是半个段"的文件。没有摘要，那种文件会被当成完整音频载入 —— 用户听到的是
//! 一段被砍掉的录音，而没有任何提示。摘要让"文件不完整"变成一次明确的读取失败。
//!
//! 【为什么不是 WAV】sidecar 只在插件与自己的渲染管线之间传递，不需要互操作性；
//! 而 WAV 的头部字段（fmt 块、块对齐、容器位深）带来的是一整类"写对了但读错了"
//! 的问题。这里的格式只有五个字段，且有测试钉住。

use super::recorder::{ClosedReason, Segment, SegmentLoss};
use std::io::Write;
use std::path::{Path, PathBuf};

/// 文件魔数（含版本，便于将来一眼分辨）。
const MAGIC: &[u8; 8] = b"HFSXFER1";
/// 载荷版本。
///
/// v2 在每段几何后多写一个 `loop_start`（`i64::MIN` = 非循环段）；v1 仍可读。
/// v3 再多写一段 coverage（段内相对帧的覆盖区间）；v1/v2 仍可读，读出来是"整段都算录到"。
/// v4 又写了一段"位置可信度"（`trusted`）。**v5 起不再写它，读 v4 时丢弃。**
///
/// 【为什么 v4 的字段必须"读但忽略"而不是删掉解析】v4 已经落在用户的磁盘上（实测
/// `take-1/2.hfx` 里 `trusted = [31568, …]`）。那个字段的判据是错的：它把"位置由插件
/// 按块长接着数"（起播窗口）当成了"位置不可信"，于是一段**真实录音**被声明成洞 ——
/// 播放侧回退宿主原声（前 0.658 s 不受参数线影响），并在边界处做一次源切换（那一下"咔"）。
/// 只删代码不会修好已经落盘的那些文件：读入侧必须**主动丢弃**这张表，让它们回到
/// "拷到样本就是我们的素材"这条唯一自洽的语义。
const VERSION: u32 = 5;
/// 尾部摘要长度。
const DIGEST_LEN: usize = 32;
/// 单次读取允许的最大帧数（防一个损坏的长度字段撑爆内存）。
///
/// 【为什么需要一个上限】`frames` 是文件里的一个 u64。损坏或被篡改的文件可以让它
/// 变成 2^63，而"按声明的长度分配"就是一次必然的 OOM。上限取 2 小时 @ 384 kHz。
const MAX_FRAMES: u64 = 384_000 * 3600 * 2;

/// sidecar 目录：本机数据目录下的 `capture/`。
///
/// 【为什么不是配置目录】配置目录会随用户配置漫游/同步，而这是可能上百 MB 的缓存；
/// 把它放进同步目录是给用户的网络与配额添麻烦。
pub(crate) fn default_dir() -> Option<PathBuf> {
    hifishifter_kernel::config_location::default_local_data_dir().map(|dir| dir.join("capture"))
}

/// 一个 take 的 sidecar 路径。
pub(crate) fn take_path(dir: &Path, take_id: u64) -> PathBuf {
    dir.join(format!("take-{take_id}.hfx"))
}

/// 把段写成 sidecar。返回实际写入的路径。
pub(crate) fn write(dir: &Path, take_id: u64, segments: &[Segment]) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = take_path(dir, take_id);
    let mut payload = Vec::new();
    payload.extend_from_slice(MAGIC);
    payload.extend_from_slice(&VERSION.to_le_bytes());
    payload.extend_from_slice(&(segments.len() as u32).to_le_bytes());
    for segment in segments {
        let channels = segment.planes.len() as u32;
        payload.extend_from_slice(&segment.origin.to_le_bytes());
        payload.extend_from_slice(&segment.sample_rate.to_le_bytes());
        payload.extend_from_slice(&channels.to_le_bytes());
        payload.extend_from_slice(&(segment.filled as u64).to_le_bytes());
        payload.extend_from_slice(&segment.loop_start.unwrap_or(i64::MIN).to_le_bytes());
        // v3：覆盖区间。槽位用尽时写 0 个区间 —— 读回来就是"整段都算录到"。
        let ranges = if segment.coverage_overrun {
            &segment.coverage[..0]
        } else {
            &segment.coverage[..segment.coverage_len.min(segment.coverage.len())]
        };
        payload.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
        for range in ranges {
            payload.extend_from_slice(&(range.start as u32).to_le_bytes());
            payload.extend_from_slice(&(range.end as u32).to_le_bytes());
        }
        // 【v5 不再写 v4 的"位置可信度"段】见 `VERSION` 的说明：那条判据把真实录音
        // 声明成了洞。读入侧仍会跳过 v4 的那一段（老文件要能读），但新文件里它不存在。
        for plane in &segment.planes {
            for sample in plane {
                payload.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }
    let digest = blake3::hash(&payload);
    payload.extend_from_slice(digest.as_bytes());
    // 【为什么先写临时文件再改名】直接写目标路径时，进程被杀会留下半个文件 ——
    // 而它在名字上已经是"完整"的。改名是原子的，于是磁盘上要么是旧版本、要么是
    // 完整的新版本，不存在中间态。
    let temp = path.with_extension("hfx.part");
    {
        let mut file =
            std::fs::File::create(&temp).map_err(|e| format!("create {}: {e}", temp.display()))?;
        file.write_all(&payload)
            .map_err(|e| format!("write {}: {e}", temp.display()))?;
        file.sync_all()
            .map_err(|e| format!("sync {}: {e}", temp.display()))?;
    }
    std::fs::rename(&temp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        format!("rename {}: {e}", path.display())
    })?;
    Ok(path)
}

/// 读回一个 sidecar。
pub(crate) fn read(path: &Path) -> Result<Vec<Segment>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < MAGIC.len() + 4 + 4 + DIGEST_LEN {
        return Err(format!("{}: truncated capture sidecar", path.display()));
    }
    // 摘要校验必须在**任何解析之前**：先信内容再验摘要，等于让损坏的字段参与解析。
    let split = bytes.len() - DIGEST_LEN;
    let (payload, digest) = bytes.split_at(split);
    if blake3::hash(payload).as_bytes() != digest {
        return Err(format!(
            "{}: capture sidecar digest mismatch (truncated or corrupted)",
            path.display()
        ));
    }
    let mut cursor = Cursor::new(payload);
    if cursor.take(8)? != MAGIC {
        return Err(format!("{}: not a capture sidecar", path.display()));
    }
    let version = cursor.u32()?;
    if !(1..=VERSION).contains(&version) {
        return Err(format!(
            "{}: unsupported capture sidecar version {version}",
            path.display()
        ));
    }
    let count = cursor.u32()? as usize;
    let mut segments = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let origin = cursor.i64()?;
        let sample_rate = cursor.u32()?;
        let channels = cursor.u32()?;
        let frames = cursor.u64()?;
        let loop_start = if version >= 2 {
            let raw = cursor.i64()?;
            (raw != i64::MIN).then_some(raw)
        } else {
            None
        };
        // v3：覆盖区间。按声明长度读，越界就是文件损坏。
        // 槽位用尽 ⇒ 整段都算录到（旧语义）：宁可少过滤，也不能分配。
        let mut coverage = std::array::from_fn(|_| 0_usize..0_usize);
        let mut coverage_len = 0_usize;
        if version >= 3 {
            let count = cursor.u32()?;
            let count = usize::try_from(count).unwrap_or(usize::MAX);
            if count > frames as usize {
                return Err(format!(
                    "{}: implausible coverage range count ({count} > {frames})",
                    path.display()
                ));
            }
            for _ in 0..count {
                let start = cursor.u32()? as usize;
                let end = cursor.u32()? as usize;
                if start >= end || end > frames as usize {
                    return Err(format!(
                        "{}: invalid coverage range {start}..{end} (frames={frames})",
                        path.display()
                    ));
                }
                if coverage_len >= super::recorder::COVERAGE_SLOTS {
                    // 超出版本允许的槽位：整段都算录到。
                    coverage_len = 0;
                    break;
                }
                coverage[coverage_len] = start..end;
                coverage_len += 1;
            }
        }
        // 【v4 的"位置可信度"段：**读但丢弃**】
        //
        // v4 写过一个 `u8 known` + 一段区间表。那段数据的判据是错的（见 `VERSION` 的
        // 说明）：它把起播窗口里"位置由插件按块长接着数"的帧标成"不是我们的"，于是
        // 一段真实录音被声明成洞。已经落盘的文件必须在这里被**主动忽略** —— 只删写入侧
        // 不会修好用户机器上那些文件。
        //
        // 【为什么仍然完整解析而不跳过字节】跳过长度未知的段需要先知道它多长，而
        // 长度本身就在段里。照原格式读过去（并校验边界），是唯一能保持摘要与后续字段
        // 对齐的做法。
        if version == 4 {
            let _known = cursor.u8()? != 0;
            let count = cursor.u32()?;
            let count = usize::try_from(count).unwrap_or(usize::MAX);
            if count > frames as usize {
                return Err(format!(
                    "{}: implausible trusted range count ({count} > {frames})",
                    path.display()
                ));
            }
            for _ in 0..count {
                let start = cursor.u32()? as usize;
                let end = cursor.u32()? as usize;
                if start >= end || end > frames as usize {
                    return Err(format!(
                        "{}: invalid trusted range {start}..{end} (frames={frames})",
                        path.display()
                    ));
                }
            }
        }
        if !(1..=2).contains(&channels) || frames == 0 || frames > MAX_FRAMES {
            return Err(format!(
                "{}: implausible segment shape (channels={channels}, frames={frames})",
                path.display()
            ));
        }
        let mut planes = Vec::with_capacity(channels as usize);
        for _ in 0..channels {
            let mut plane = Vec::new();
            plane
                .try_reserve_exact(frames as usize)
                .map_err(|_| format!("{}: PCM allocation failed", path.display()))?;
            for _ in 0..frames {
                plane.push(cursor.f32()?);
            }
            planes.push(plane);
        }
        segments.push(Segment {
            origin,
            sample_rate,
            planes,
            filled: frames as usize,
            // 【为什么 loss 恒为 First】loss 描述的是"本段与前一段的关系"，它是
            // **本次捕获**的诊断事实，不是音频的属性。载入时没有前一段可比，
            // 谎报 `None`（"连续"）会让界面显示一条不存在的连续性保证。
            loss: SegmentLoss::First,
            loop_start,
            peak: [0.0; 2],
            closed: ClosedReason::Stopped,
            coverage,
            coverage_len,
            coverage_overrun: false,
            migrated_from_v4: version == 4,
            _reservation: None,
        });
    }
    Ok(segments)
}

/// 删除一个 sidecar。
///
/// 【为什么移入回收站而不是直接删】录了几分钟的东西被误删是不可接受的；
/// 与文件浏览器的删除语义保持一致（同一个 `trash` 依赖）。
pub(crate) fn remove(path: &Path) {
    if path.exists() {
        if let Err(error) = trash::delete(path) {
            log::warn!("[capture] could not trash {}: {error}", path.display());
        }
    }
}

/// 极简的小端读取游标；越界一律是明确错误而不是 panic。
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or("capture sidecar ends mid-field")?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        let bytes = self.take(1)?;
        Ok(bytes[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i64(&mut self) -> Result<i64, String> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    fn f32(&mut self) -> Result<f32, String> {
        let bytes = self.take(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(origin: i64, values: &[f32]) -> Segment {
        Segment {
            origin,
            sample_rate: 48_000,
            planes: vec![values.to_vec(), values.to_vec()],
            filled: values.len(),
            loss: SegmentLoss::First,
            loop_start: None,
            peak: [0.0; 2],
            closed: ClosedReason::Stopped,
            coverage: std::array::from_fn(|_| 0..0),
            coverage_len: 0,
            coverage_overrun: false,
            migrated_from_v4: false,
            _reservation: None,
        }
    }

    /// 每个用例一个独立目录。
    ///
    /// 【为什么不用 uuid】`uuid` 不是本 crate 的直接依赖（它由 `ara2-bridge` 间接带入），
    /// 为一个测试目录去加依赖不值得。进程号 + 单调计数在单个测试进程内已经唯一。
    fn temp_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hfs-sidecar-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 往返：几何、采样率与样本值必须逐位一致。
    #[test]
    fn a_round_trip_preserves_geometry_and_samples() {
        let dir = temp_dir();
        let segments = vec![segment(0, &[0.25, -0.5, 0.75]), segment(9600, &[1.0, 0.0])];
        let path = write(&dir, 7, &segments).unwrap();
        let restored = read(&path).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].origin, 0);
        assert_eq!(restored[0].sample_rate, 48_000);
        assert_eq!(restored[0].planes[0], vec![0.25, -0.5, 0.75]);
        assert_eq!(restored[0].planes[1], vec![0.25, -0.5, 0.75]);
        assert_eq!(restored[1].origin, 9600);
        assert_eq!(restored[1].planes[0], vec![1.0, 0.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **截断必须被发现**：半个文件不能被当成完整音频载入。
    ///
    /// 【为什么这是最重要的一条】截断是磁盘满与进程被杀最常见的后果，而它留下的
    /// 文件在结构上"看起来合法" —— 没有摘要校验就会被读成一段被砍掉的录音，
    /// 用户听得到、但听不全，且没有任何提示。
    #[test]
    fn a_truncated_file_is_rejected_instead_of_read_as_half_audio() {
        let dir = temp_dir();
        let path = write(&dir, 1, &[segment(0, &[0.1, 0.2, 0.3, 0.4])]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 12]).unwrap();
        let error = read(&path).unwrap_err();
        assert!(
            error.contains("digest") || error.contains("truncated"),
            "必须明确报出文件不完整，而不是返回半段音频：{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 内容被改写（哪怕一个字节）也必须被发现。
    #[test]
    fn a_corrupted_payload_is_rejected() {
        let dir = temp_dir();
        let path = write(&dir, 2, &[segment(0, &[0.1, 0.2])]).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last_sample = bytes.len() - DIGEST_LEN - 1;
        bytes[last_sample] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read(&path).unwrap_err().contains("digest"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 不是 sidecar 的文件被明确拒绝，而不是按巧合解析出内容。
    #[test]
    fn a_foreign_file_is_rejected() {
        let dir = temp_dir();
        let path = dir.join("foreign.hfx");
        std::fs::write(&path, vec![0u8; 128]).unwrap();
        assert!(read(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 写入是原子的：失败不会留下半个目标文件。
    #[test]
    fn a_successful_write_leaves_no_partial_file_behind() {
        let dir = temp_dir();
        let path = write(&dir, 3, &[segment(0, &[0.5])]).unwrap();
        assert!(path.exists());
        assert!(
            !path.with_extension("hfx.part").exists(),
            "临时文件必须已被改名消费掉"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// take 号决定文件名：同一个 take 重写会覆盖同一路径（不产生孤儿）。
    #[test]
    fn the_path_is_derived_from_the_take_id() {
        let dir = temp_dir();
        assert_eq!(take_path(&dir, 12), dir.join("take-12.hfx"));
        let first = write(&dir, 12, &[segment(0, &[0.1])]).unwrap();
        let second = write(&dir, 12, &[segment(0, &[0.9])]).unwrap();
        assert_eq!(first, second);
        assert_eq!(read(&second).unwrap()[0].planes[0], vec![0.9]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// v2 往返必须保留循环 pass 的起点；None 也必须原样回来。
    #[test]
    fn a_round_trip_preserves_the_loop_start_metadata() {
        let dir = temp_dir();
        let mut looped = segment(0, &[0.1, 0.2]);
        looped.loop_start = Some(9_600);
        let plain = segment(4, &[0.3]);
        let path = write(&dir, 4, &[looped, plain]).unwrap();
        let restored = read(&path).unwrap();
        assert_eq!(restored[0].loop_start, Some(9_600));
        assert_eq!(restored[1].loop_start, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// v1 文件（没有 loop_start 字段）仍必须可读，且按“非循环段”解释。
    #[test]
    fn a_version_one_sidecar_still_loads() {
        let dir = temp_dir();
        let path = dir.join("legacy.hfx");
        let mut payload = Vec::new();
        payload.extend_from_slice(MAGIC);
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&0_i64.to_le_bytes());
        payload.extend_from_slice(&48_000_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&2_u64.to_le_bytes());
        payload.extend_from_slice(&0.25_f32.to_le_bytes());
        payload.extend_from_slice(&0.5_f32.to_le_bytes());
        let digest = blake3::hash(&payload);
        payload.extend_from_slice(digest.as_bytes());
        std::fs::write(&path, payload).unwrap();

        let restored = read(&path).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].loop_start, None);
        assert_eq!(restored[0].planes[0], vec![0.25, 0.5]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **v4 文件里那张"位置可信度"表必须被丢弃。**
    ///
    /// 【为什么这是一条回归测试而不是兼容性测试】v4 的 `trusted` 判据把起播窗口里
    /// "位置由插件按块长接着数"的帧标成"不是我们的"。实测用户机器上的 `take-1/2.hfx`
    /// 里就是 `trusted = [31568, …]`（0.658 s）。只删写入侧不会修好它们：读入侧必须
    /// 主动忽略，否则那些 take 会**永远**在开头留一个 0.658 s 的假洞 —— 播放时回退
    /// 宿主原声（参数线不生效）并在边界处咔一下。
    #[test]
    fn a_version_four_sidecar_drops_its_trusted_table() {
        let dir = temp_dir();
        let path = dir.join("v4.hfx");
        let mut payload = Vec::new();
        payload.extend_from_slice(MAGIC);
        payload.extend_from_slice(&4_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&0_i64.to_le_bytes());
        payload.extend_from_slice(&48_000_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&8_u64.to_le_bytes());
        payload.extend_from_slice(&i64::MIN.to_le_bytes());
        // coverage：整段都拷到了。
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&8_u32.to_le_bytes());
        // v4 的 trusted：`known = 1`，一段 `4..8`（前一半被判成"不可信"）。
        payload.push(1);
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&4_u32.to_le_bytes());
        payload.extend_from_slice(&8_u32.to_le_bytes());
        for index in 0..8 {
            payload.extend_from_slice(&(index as f32 * 0.1).to_le_bytes());
        }
        let digest = blake3::hash(&payload);
        payload.extend_from_slice(digest.as_bytes());
        std::fs::write(&path, payload).unwrap();

        let restored = read(&path).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(
            restored[0].coverage_len, 1,
            "覆盖必须原样读回：v4 的 coverage 是对的"
        );
        assert_eq!(
            restored[0].coverage[0],
            0..8,
            "覆盖必须整段，不受 v4 的 trusted 表影响"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
