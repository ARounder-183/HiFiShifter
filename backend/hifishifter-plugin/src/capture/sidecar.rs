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
//! magic     8 字节  "HFSXFERC"
//! version   u32     = 1
//! segments  u32
//! 每段:
//!   origin   i64    工程采样起点
//!   rate     u32    采样率
//!   channels u32
//!   frames   u64
//!   loop     i64    循环起点（`i64::MIN` = 非循环段）
//!   coverage u32 个 (start u32, end u32) —— 段内**真正拷到宿主样本**的帧区间
//!   samples  frames × channels × f32（小端，平面顺序）
//! digest    32 字节 BLAKE3（覆盖以上全部内容）
//! ```
//!
//! 【为什么要有尾部摘要】截断的写入（磁盘满、进程被杀）会留下一个"看起来合法、
//! 读出来是半个段"的文件。没有摘要，那种文件会被当成完整音频载入 —— 用户听到的是
//! 一段被砍掉的录音，而没有任何提示。摘要让"文件不完整"变成一次明确的读取失败。
//!
//! 【为什么不是 WAV】sidecar 只在插件与自己的渲染管线之间传递，不需要互操作性；
//! 而 WAV 的头部字段（fmt 块、块对齐、容器位深）带来的是一整类"写对了但读错了"
//! 的问题。这里的格式只有几个字段，且有测试钉住。
//!
//! # 为什么版本号是 1，且没有任何向后兼容分支
//!
//! 这个格式此前演进到过 v5，每一版都在读入侧留了分支：`loop_start`（v2+）、
//! `coverage`（v3+），以及 v4 那段必须"读但丢弃"的 `trusted` 表。那些分支保护的是
//! **已经落在某台机器磁盘上的数据** —— 而 VST3 侧（含这条捕获通路）从未向任何用户
//! 分发过，那些文件只存在于开发者的测试机上。
//!
//! 为一份没有第二份拷贝的数据保留五条读路径，代价是持续的：每段多三个版本判断、
//! 一个只为报告"这是旧文件"而存在的 `Segment::migrated_from_v4` 字段（十处引用）、
//! 两条钉住"不再会产生"的数据的测试，以及一条会把"未来版本"误报成
//! `unsupported version` 的判据。所以格式在这里**重置**：版本回到 1，读入侧只认
//! 这一种形状，魔数也去掉那个早已与 `VERSION` 不一致的版本位。
//!
//! 旧文件不会被静默误读：魔数与版本不匹配时返回一句可执行的错误（见 `read`），
//! 而 `CaptureStore::restore_manifest` 本来就会跳过读不出来的 take。

use super::recorder::{ClosedReason, Segment, SegmentLoss};
use std::io::Write;
use std::path::{Path, PathBuf};

/// 文件魔数。只回答"这是不是 HiFiShifter 的捕获 sidecar"，**不含版本**。
///
/// 【为什么去掉版本位】此前的 `HFSXFER1` 里那个 `1` 与载荷里的 `VERSION`（当时是 5）
/// 早已不一致 —— 魔数里的版本位是个死的副本，真正被解析的只有载荷里那个 u32。
/// 两份真相必然漂移，所以只留一份。
///
/// 【为什么仍然占满 8 字节】`MAGIC.len()` 参与最小长度判据，8 是齐整字长，
/// 且 `&[u8; 8]` 与旧类型一致 —— 改动面最小。
const MAGIC: &[u8; 8] = b"HFSXFERC";
/// 载荷版本（重置后的第一版）。
const VERSION: u32 = 1;
/// 尾部摘要长度。
const DIGEST_LEN: usize = 32;
/// 单次读取允许的最大帧数（防一个损坏的长度字段撑爆内存）。
///
/// 【为什么需要一个上限】`frames` 是文件里的一个 u64。损坏或被篡改的文件可以让它
/// 变成 2^63，而"按声明的长度分配"就是一次必然的 OOM。上限取 2 小时 @ 384 kHz。
const MAX_FRAMES: u64 = 384_000 * 3600 * 2;
/// 临时文件后缀（写完后原子改名为目标文件）。
const TEMP_EXT: &str = "hfscap.part";
/// 最终文件的扩展名。
const FILE_EXT: &str = "hfscap";

/// sidecar 目录：本机数据目录下的 `capture/`。
///
/// 【为什么不是配置目录】配置目录会随用户配置漫游/同步，而这是可能上百 MB 的缓存；
/// 把它放进同步目录是给用户的网络与配额添麻烦。
pub(crate) fn default_dir() -> Option<PathBuf> {
    hifishifter_kernel::config_location::default_local_data_dir().map(|dir| dir.join("capture"))
}

/// 一个 take 的 sidecar 路径。
///
/// 【为什么文件名里有一个 nonce，而不只是 `take-{id}`】`take_id` 是**插件实例级**的
/// 计数器：两个不同的宿主工程各自录第一个 take，都叫 `take-1` —— 而它们共用同一个
/// 缓存目录。只按号命名时，后录的会**覆盖**先录的：打开第一个工程会读到第二个工程
/// 的音频，而且没有任何报错。
///
/// 【为什么不是 `{工程id}-take-{id}`】捕获通路**根本没有工程身份**（宿主不给几何，
/// 也不给工程路径 —— 那正是这条通路存在的理由）。给每个 take 一个一次性 nonce
/// 不需要任何跨层管道，而且比"每个工程一个前缀"更强：它连"同一工程的两个实例"
/// 都分得开。
///
/// 清单里存的是**绝对路径**，所以恢复不依赖这条命名规则。
fn take_path(dir: &Path, take_id: u64, nonce: &str) -> PathBuf {
    dir.join(format!("capture-{take_id}-{nonce}.{FILE_EXT}"))
}

/// 生成一个 take 的一次性文件名后缀（16 位十六进制）。
///
/// 【为什么不引入 uuid/rand】本 crate 没有那两个依赖，而这里要的只是"同一进程内
/// 单调、跨进程几乎不撞"。进程号 + 单调计数 + 墙钟纳秒足够，且不需要新依赖 ——
/// 与 `hifishifter-host-link` 的 `link_id` 同一套取舍。
fn take_nonce(take_id: u64) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hfs-capture-take-v1");
    hasher.update(&std::process::id().to_le_bytes());
    hasher.update(&nanos.to_le_bytes());
    hasher.update(&seq.to_le_bytes());
    hasher.update(&take_id.to_le_bytes());
    hasher.finalize().to_hex()[..16].to_string()
}

/// 把段写成 sidecar。返回实际写入的路径。
pub(crate) fn write(dir: &Path, take_id: u64, segments: &[Segment]) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = take_path(dir, take_id, &take_nonce(take_id));
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
        // 覆盖区间。槽位用尽时写 0 个区间 —— 读回来就是"整段都算录到"。
        // 见 `Segment::coverage` 的说明：那四类零都是"我们没拿到样本"，不是"录到静音"。
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
    let temp = path.with_extension(TEMP_EXT);
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
///
/// 【错误文案为什么是可执行的】重置格式之后，开发者自己机器上还留着旧版文件。
/// 一句"`unsupported version 5`"只会让人去翻代码；这里直接说明"这是重置前构建留下的"
/// 并给出下一步（删掉或重录）。
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
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(format!(
            "{}: not a HiFiShifter capture sidecar (stale file from a pre-reset build?); \
             delete it or re-capture",
            path.display()
        ));
    }
    let version = cursor.u32()?;
    if version != VERSION {
        return Err(format!(
            "{}: capture sidecar version {version}, this build writes {VERSION} \
             (stale file from a pre-reset build?); delete it or re-capture",
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
        let raw_loop = cursor.i64()?;
        let loop_start = (raw_loop != i64::MIN).then_some(raw_loop);
        // 覆盖区间：按声明长度读，越界就是文件损坏。
        // 槽位用尽 ⇒ 整段都算录到（宁可少过滤，也不能分配）。
        let mut coverage = std::array::from_fn(|_| 0_usize..0_usize);
        let mut coverage_len = 0_usize;
        let range_count = cursor.u32()?;
        let range_count = usize::try_from(range_count).unwrap_or(usize::MAX);
        if range_count > frames as usize {
            return Err(format!(
                "{}: implausible coverage range count ({range_count} > {frames})",
                path.display()
            ));
        }
        for _ in 0..range_count {
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
        let path = dir.join("foreign.hfscap");
        std::fs::write(&path, vec![0u8; 128]).unwrap();
        assert!(read(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **重置前构建留下的旧文件必须被明确拒绝，且错误要可执行。**
    ///
    /// 【为什么这条取代了原来的两条版本兼容测试】格式已重置（见模块头）：
    /// v1..v5 的读路径与 `Segment::migrated_from_v4` 一起删掉了。留下的唯一义务是
    /// 别把旧文件**误读**成新文件 —— 魔数变了，所以这里钉住"报错且说清怎么办"。
    #[test]
    fn a_pre_reset_sidecar_is_rejected_with_an_actionable_error() {
        let dir = temp_dir();
        let path = dir.join("legacy.hfscap");
        // 旧魔数 + 旧版本号 + 一段几何，摘要按新规则附在尾部（这样摘要校验会通过，
        // 真正被检验的是魔数/版本判据本身）。
        let mut payload = Vec::new();
        payload.extend_from_slice(b"HFSXFER1");
        payload.extend_from_slice(&5_u32.to_le_bytes());
        payload.extend_from_slice(&0_u32.to_le_bytes());
        let digest = blake3::hash(&payload);
        payload.extend_from_slice(digest.as_bytes());
        std::fs::write(&path, payload).unwrap();

        let error = read(&path).unwrap_err();
        assert!(
            error.contains("not a HiFiShifter capture sidecar")
                && error.contains("delete it or re-capture"),
            "旧文件必须被明确拒绝并给出下一步：{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 写入是原子的：失败不会留下半个目标文件。
    #[test]
    fn a_successful_write_leaves_no_partial_file_behind() {
        let dir = temp_dir();
        let path = write(&dir, 3, &[segment(0, &[0.5])]).unwrap();
        assert!(path.exists());
        assert!(
            !path.with_extension(TEMP_EXT).exists(),
            "临时文件必须已被改名消费掉"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **同一个 take 号写两次不得互相覆盖，也不得跨工程串味。**
    ///
    /// 【为什么这条是回归】`take_id` 是插件实例级计数器：两个不同的宿主工程各自
    /// 录第一个 take，都叫 `take-1`，而它们共用同一个缓存目录。只按号命名时后写的
    /// 会覆盖先写的 —— 打开第一个工程会读到第二个工程的音频，且没有任何报错。
    #[test]
    fn two_takes_with_the_same_id_do_not_overwrite_each_other() {
        let dir = temp_dir();
        let first = write(&dir, 12, &[segment(0, &[0.1])]).unwrap();
        let second = write(&dir, 12, &[segment(0, &[0.9])]).unwrap();
        assert_ne!(first, second, "同号 take 必须落在不同文件上");
        assert_eq!(read(&first).unwrap()[0].planes[0], vec![0.1]);
        assert_eq!(read(&second).unwrap()[0].planes[0], vec![0.9]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 循环 pass 的起点必须原样往返；`None` 也必须原样回来。
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

    /// 覆盖区间必须原样往返 —— 它是"我们没有这一段"的唯一证据。
    #[test]
    fn a_round_trip_preserves_the_coverage_ranges() {
        let dir = temp_dir();
        let mut holed = segment(0, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        holed.coverage[0] = 0..2;
        holed.coverage[1] = 4..6;
        holed.coverage_len = 2;
        let path = write(&dir, 5, &[holed]).unwrap();
        let restored = read(&path).unwrap();
        assert_eq!(restored[0].coverage_len, 2);
        assert_eq!(restored[0].coverage[0], 0..2);
        assert_eq!(restored[0].coverage[1], 4..6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 覆盖槽位溢出时退化为"整段都算录到"，而不是写出一份读不回来的文件。
    #[test]
    fn an_overrun_coverage_writes_zero_ranges() {
        let dir = temp_dir();
        let mut overrun = segment(0, &[0.1, 0.2]);
        overrun.coverage[0] = 0..1;
        overrun.coverage_len = 1;
        overrun.coverage_overrun = true;
        let path = write(&dir, 6, &[overrun]).unwrap();
        let restored = read(&path).unwrap();
        assert_eq!(restored[0].coverage_len, 0, "溢出 ⇒ 整段都算录到");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
