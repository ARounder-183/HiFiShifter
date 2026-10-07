//! ARA宿主mono合成PCM的内容磁盘缓存；不参与文件App缓存，不在实时线程调用。
//! 完整内容/模型键与完整payload摘要校验，固定配额、原子替换；缓存失败只退回原推理。

use crate::renderer::ClipProcessContext;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"HFSARA01";
const HEADER: usize = 96;
const ENTRY_LIMIT: usize = 64 * 1024 * 1024;
const DISK_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
static HITS: AtomicU64 = AtomicU64::new(0);
static STORES: AtomicU64 = AtomicU64::new(0);
static MAINTENANCE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 独立命名空间；测试可显式指向worktree，不改变App的磁盘设置/缓存世代。
fn root() -> PathBuf {
    if let Some(path) = std::env::var_os("HIFISHIFTER_ARA_CACHE_DIR").filter(|p| !p.is_empty()) {
        return PathBuf::from(path);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Caches"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("HiFiShifter")
        .join("ara-content-v1")
}

/// 不包含clip名/声道序号，实际mono内容自行区分平面；共通混音参数不属于合成输入。
pub(crate) fn key(
    ctx: &ClipProcessContext<'_>,
    vocoder: &str,
    separator: Option<&str>,
) -> blake3::Hash {
    let mut hash = blake3::Hasher::new();
    hash.update(b"ara-host-processor-pcm-v1");
    hash.update(&crate::synth_clip_cache::RENDER_PIPELINE_VERSION.to_le_bytes());
    hash.update(&(vocoder.len() as u64).to_le_bytes());
    hash.update(vocoder.as_bytes());
    if let Some(separator) = separator {
        hash.update(&(separator.len() as u64).to_le_bytes());
        hash.update(separator.as_bytes());
    } else {
        hash.update(&0_u64.to_le_bytes());
    }
    hash.update(&ctx.sample_rate.to_le_bytes());
    hash.update(&(ctx.out_frames as u64).to_le_bytes());
    for value in [
        ctx.clip_start_sec,
        ctx.seg_start_sec,
        ctx.seg_end_sec,
        ctx.frame_period_ms,
        ctx.playback_rate,
    ] {
        hash.update(&value.to_bits().to_le_bytes());
    }
    for values in [ctx.mono_pcm, ctx.pitch_edit, ctx.clip_midi] {
        hash.update(&(values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(&value.to_bits().to_le_bytes());
        }
    }
    let mut curves = ctx
        .extra_curves
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "volume" | "pan" | "dyn" | "hifigan_volume"))
        .collect::<Vec<_>>();
    curves.sort_by_key(|(name, _)| *name);
    hash.update(&(curves.len() as u64).to_le_bytes());
    for (name, values) in curves {
        hash.update(&(name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(&(values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(&value.to_bits().to_le_bytes());
        }
    }
    let mut params = ctx.extra_params.iter().collect::<Vec<_>>();
    params.sort_by_key(|(name, _)| *name);
    hash.update(&(params.len() as u64).to_le_bytes());
    for (name, value) in params {
        hash.update(&(name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(&value.to_bits().to_le_bytes());
    }
    hash.finalize()
}

fn path_for(root: &Path, key: blake3::Hash) -> PathBuf {
    root.join(format!("{}.hfsara", key.to_hex()))
}
fn invalid() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid ARA PCM cache")
}

/// 校验未受信任的文件头、精确文件大小与完整payload；分块解码避免第二份整段byte缓冲。
fn read(path: &Path, key: blake3::Hash, rate: u32, frames: usize) -> std::io::Result<Vec<f32>> {
    let bytes = frames
        .checked_mul(4)
        .filter(|n| *n <= ENTRY_LIMIT)
        .ok_or_else(invalid)?;
    let mut file = File::open(path)?;
    if file.metadata()?.len() != HEADER as u64 + bytes as u64 {
        return Err(invalid());
    }
    let mut header = [0_u8; HEADER];
    file.read_exact(&mut header)?;
    if &header[..8] != MAGIC
        || u32::from_le_bytes(header[8..12].try_into().unwrap())
            != crate::synth_clip_cache::RENDER_PIPELINE_VERSION
        || u32::from_le_bytes(header[12..16].try_into().unwrap()) != rate
        || u64::from_le_bytes(header[16..24].try_into().unwrap()) != frames as u64
        || &header[24..56] != key.as_bytes()
        || header[88..].iter().any(|v| *v != 0)
    {
        return Err(invalid());
    }
    let mut out = Vec::with_capacity(frames);
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0_u8; 32 * 1024];
    while out.len() < frames {
        let bytes = ((frames - out.len()) * 4).min(buffer.len());
        file.read_exact(&mut buffer[..bytes])?;
        hash.update(&buffer[..bytes]);
        for bits in buffer[..bytes].chunks_exact(4) {
            let value = f32::from_le_bytes(bits.try_into().unwrap());
            if !value.is_finite() {
                return Err(invalid());
            }
            out.push(value);
        }
    }
    if hash.finalize().as_bytes() != &header[56..88] || file.read(&mut buffer[..1])? != 0 {
        return Err(invalid());
    }
    Ok(out)
}

/// 只删除本命名空间内准确内容键的损坏文件；缓存缺失/权限失败均不是产品错误。
pub(crate) fn load(key: blake3::Hash, rate: u32, frames: usize) -> Option<Vec<f32>> {
    let path = path_for(&root(), key);
    match read(&path, key, rate, frames) {
        Ok(out) => {
            HITS.fetch_add(1, Ordering::Relaxed);
            Some(out)
        }
        Err(error) => {
            if error.kind() == std::io::ErrorKind::InvalidData
                || error.kind() == std::io::ErrorKind::UnexpectedEof
            {
                let _ = fs::remove_file(&path);
                log::warn!("[ara-cache] corrupt entry rejected: {error}");
            }
            None
        }
    }
}

fn cache_entry_name(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "hfsara")
        && path
            .file_stem()
            .and_then(|p| p.to_str())
            .is_some_and(|stem| stem.len() == 64 && stem.bytes().all(|b| b.is_ascii_hexdigit()))
}
fn temporary_entry_name(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "tmp")
        && path
            .file_stem()
            .and_then(|p| p.to_str())
            .is_some_and(|stem| {
                stem.len() == 101
                    && stem.as_bytes()[64] == b'.'
                    && stem.as_bytes()[..64].iter().all(|b| b.is_ascii_hexdigit())
                    && uuid::Uuid::parse_str(&stem[65..]).is_ok()
            })
}
/// 单层精确cache文件集合按字节驱逐；不递归删除目录、不碰其它名称/链接/用户资产。
fn prune(root: &Path, limit: u64) -> std::io::Result<()> {
    let mut entries = Vec::new();
    let mut total = 0_u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        if temporary_entry_name(&entry.path()) {
            if entry
                .metadata()?
                .modified()?
                .elapsed()
                .is_ok_and(|age| age.as_secs() > 3600)
            {
                let _ = fs::remove_file(entry.path());
            }
            continue;
        }
        if !cache_entry_name(&entry.path()) {
            continue;
        }
        let metadata = entry.metadata()?;
        total = total.saturating_add(metadata.len());
        entries.push((metadata.modified().ok(), metadata.len(), entry.path()));
    }
    entries.sort_by_key(|(time, _, _)| *time);
    for (_, bytes, path) in entries {
        if total <= limit {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(bytes);
        }
    }
    if total > limit {
        return Err(std::io::Error::other("ARA cache quota cannot be reclaimed"));
    }
    Ok(())
}

/// 有界原子写在worker完成，不排队克隆整段；取消/过期任务由调用方在此入口前拒绝。
fn write(root: &Path, key: blake3::Hash, rate: u32, pcm: &[f32]) -> std::io::Result<()> {
    if pcm
        .len()
        .checked_mul(4)
        .is_none_or(|bytes| bytes > ENTRY_LIMIT)
        || pcm.iter().any(|v| !v.is_finite())
    {
        return Err(invalid());
    }
    fs::create_dir_all(root)?;
    let mut digest = blake3::Hasher::new();
    for value in pcm {
        digest.update(&value.to_bits().to_le_bytes());
    }
    let mut header = [0_u8; HEADER];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&crate::synth_clip_cache::RENDER_PIPELINE_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&rate.to_le_bytes());
    header[16..24].copy_from_slice(&(pcm.len() as u64).to_le_bytes());
    header[24..56].copy_from_slice(key.as_bytes());
    header[56..88].copy_from_slice(digest.finalize().as_bytes());
    let path = path_for(root, key);
    let temporary = root.join(format!("{}.{}.tmp", key.to_hex(), uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&header)?;
        let mut buffer = [0_u8; 32 * 1024];
        for values in pcm.chunks(buffer.len() / 4) {
            for (bytes, value) in buffer.chunks_exact_mut(4).zip(values) {
                bytes.copy_from_slice(&value.to_bits().to_le_bytes());
            }
            file.write_all(&buffer[..values.len() * 4])?;
        }
        file.flush()?;
        drop(file);
        fs::rename(&temporary, &path)?;
        prune(root, DISK_LIMIT)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn store(key: blake3::Hash, rate: u32, pcm: &[f32]) {
    let _maintenance = MAINTENANCE.lock().unwrap_or_else(|e| e.into_inner());
    match write(&root(), key, rate, pcm) {
        Ok(()) => {
            STORES.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => log::warn!("[ara-cache] write skipped: {error}"),
    }
}
/// 非实时诊断统计；不包含模型session烟测，也不代表native宿主验收。
pub fn stats() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), STORES.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn content_key_ignores_owner_and_mix_only_curves_but_tracks_all_synthesis_dependencies() {
        use std::collections::HashMap;
        let audio = [0.1, 0.2, 0.3];
        let pitch = [60., 61.];
        let midi = [57., 57.];
        let curves = HashMap::new();
        let params = HashMap::new();
        let mut ctx = ClipProcessContext {
            mono_pcm: &audio,
            channel_index: 0,
            sample_rate: 44100,
            source_fingerprint: Some(7),
            clip_start_sec: 0.,
            seg_start_sec: 0.,
            seg_end_sec: 0.5,
            frame_period_ms: 5.,
            pitch_edit: &pitch,
            clip_midi: &midi,
            playback_rate: 1.,
            out_frames: 3,
            clip_id: "a",
            extra_curves: &curves,
            extra_params: &params,
        };
        let baseline = key(&ctx, "model-a", Some("separator-a"));
        ctx.clip_id = "another-owner";
        ctx.channel_index = 1;
        ctx.source_fingerprint = Some(8);
        assert_eq!(
            baseline,
            key(&ctx, "model-a", Some("separator-a")),
            "调用者身份/粗指纹不替代完整实际PCM"
        );
        let mix_only = HashMap::from([
            ("volume".into(), vec![0.25]),
            ("pan".into(), vec![0.5]),
            ("dyn".into(), vec![0.2]),
        ]);
        ctx.extra_curves = &mix_only;
        assert_eq!(baseline, key(&ctx, "model-a", Some("separator-a")));
        assert_ne!(baseline, key(&ctx, "model-b", Some("separator-a")));
        assert_ne!(baseline, key(&ctx, "model-a", Some("separator-b")));
        ctx.mono_pcm = &[0.1, 0.25, 0.3];
        assert_ne!(baseline, key(&ctx, "model-a", Some("separator-a")));
        ctx.mono_pcm = &audio;
        ctx.pitch_edit = &[64., 65.];
        assert_ne!(baseline, key(&ctx, "model-a", Some("separator-a")));
        ctx.pitch_edit = &pitch;
        let tension = HashMap::from([("hifigan_tension".into(), vec![75.])]);
        ctx.extra_curves = &tension;
        assert_ne!(baseline, key(&ctx, "model-a", Some("separator-a")));
        ctx.extra_curves = &curves;
        let separation = HashMap::from([("breath_enabled".into(), 1.)]);
        ctx.extra_params = &separation;
        assert_ne!(baseline, key(&ctx, "model-a", Some("separator-a")));
    }
    #[test]
    fn disk_pcm_roundtrip_rejects_corruption_identity_rate_and_size() {
        let dir = std::env::temp_dir().join(format!("hfs-ara-disk-{}", uuid::Uuid::new_v4()));
        let key = blake3::hash(b"key");
        let pcm = [0., 0.25, -0.5, 0.75];
        write(&dir, key, 44100, &pcm).unwrap();
        let path = path_for(&dir, key);
        assert_eq!(read(&path, key, 44100, 4).unwrap(), pcm);
        assert!(read(&path, blake3::hash(b"other"), 44100, 4).is_err());
        assert!(read(&path, key, 48000, 4).is_err());
        assert!(read(&path, key, 44100, 3).is_err());
        let mut bytes = fs::read(&path).unwrap();
        bytes[HEADER + 5] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(read(&path, key, 44100, 4).is_err());
        fs::write(&path, [0; HEADER]).unwrap();
        assert!(read(&path, key, 44100, usize::MAX).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn quota_removes_only_owned_key_files_not_unrelated_assets() {
        let dir = std::env::temp_dir().join(format!("hfs-ara-quota-{}", uuid::Uuid::new_v4()));
        for index in 0..3 {
            write(&dir, blake3::hash(&[index]), 44100, &[0.25; 4]).unwrap();
        }
        fs::write(dir.join("user-audio.hfsara"), b"user asset").unwrap();
        prune(&dir, (HEADER + 16) as u64).unwrap();
        assert_eq!(
            fs::read(dir.join("user-audio.hfsara")).unwrap(),
            b"user asset"
        );
        assert_eq!(
            fs::read_dir(&dir)
                .unwrap()
                .filter(|entry| cache_entry_name(&entry.as_ref().unwrap().path()))
                .count(),
            1
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
