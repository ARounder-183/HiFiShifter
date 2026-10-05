//! 宿主PCM的私有分析副本：只写调用方提供的私有目录，绝不把宿主ID当文件读取。
use crate::state::TimelineState;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64,Ordering};
pub struct PcmView<'a> {
    pub persistent_id:&'a str,
    pub sample_rate:u32,
    pub planes:&'a [Vec<f32>],
}

/// 在所有源/几何校验成功后生成只用于波形与分析的WAV，返回路径→宿主ID反向表。
/// 音频渲染仍从ARA授权PCM注入；不回读这些文件作为宿主供音。
pub fn materialize(timeline:TimelineState,sources:&[PcmView<'_>],dir:&Path)
    ->Result<(TimelineState,HashMap<String,String>),String> {
    materialize_with_byte_limit(timeline,sources,dir,64*1024*1024)
}

/// 同进程授权源已由调用者计入硬预算；文件流式写出不复制PCM，不按时长拒绝正常长源。
/// 外部IPC入口仍走materialize的64MiB限制，不能借此扩大旧传输协议。
pub fn materialize_with_byte_limit(mut timeline:TimelineState,sources:&[PcmView<'_>],dir:&Path,byte_limit:usize)
    ->Result<(TimelineState,HashMap<String,String>),String> {
    if byte_limit==0||byte_limit>512*1024*1024 {return Err("invalid host PCM byte limit".into());}
    if !timeline.bpm.is_finite() || timeline.bpm<=0. || !timeline.project_sec.is_finite() || timeline.project_sec<0. {
        return Err("invalid host timeline".into());
    }
    let mut known=HashMap::new();
    let mut bytes=0_usize;
    for pcm in sources {
        let frames=pcm.planes.first().map(Vec::len).unwrap_or(0);
        bytes=bytes.checked_add(frames.checked_mul(pcm.planes.len()).and_then(|n|n.checked_mul(4)).ok_or("host PCM size overflow")?).ok_or("host PCM size overflow")?;
        if bytes>byte_limit || pcm.persistent_id.is_empty() || !matches!(pcm.sample_rate,44100|48000)
            || !(1..=2).contains(&pcm.planes.len()) || frames==0
            || pcm.planes.iter().any(|p|p.len()!=frames || p.iter().any(|n|!n.is_finite()))
            || known.insert(pcm.persistent_id,pcm).is_some() { return Err("invalid or oversized host PCM".into()); }
    }
    for clip in &mut timeline.clips {
        clip.normalize_takes();
        if clip.reversed || clip.takes.iter().any(|t|t.reversed) {return Err("ARA reverse playback is unsupported".into());}
        if !clip.start_sec.is_finite() || !clip.length_sec.is_finite() || clip.length_sec<=0.
            || clip.reversed || clip.takes.iter().any(|t|t.reversed || !t.source_start_sec.is_finite()
                || !t.source_end_sec.is_finite() || !t.playback_rate.is_finite() || t.playback_rate<=0.) {
            return Err("invalid or unsupported host clip geometry".into());
        }
        for id in std::iter::once(&clip.source_path).chain(clip.takes.iter().map(|t|&t.source_path)) {
            if !known.contains_key(id.as_deref().ok_or("missing host PCM reference")?) { return Err("missing host PCM".into()); }
        }
    }
    std::fs::create_dir_all(dir).map_err(|e|e.to_string())?;
    let mut paths=HashMap::new(); let mut reverse=HashMap::new();
    for pcm in sources {
        let mut fingerprint=blake3::Hasher::new();
        fingerprint.update(pcm.persistent_id.as_bytes()); fingerprint.update(&pcm.sample_rate.to_le_bytes());
        fingerprint.update(&(pcm.planes.len() as u64).to_le_bytes());
        for plane in pcm.planes { for sample in plane { fingerprint.update(&sample.to_le_bytes()); } }
        let path=dir.join(format!("source-{}.wav",fingerprint.finalize().to_hex()));
        if !matches_authorized_pcm(&path,pcm) {
            static NEXT:AtomicU64=AtomicU64::new(1);
            let temporary=path.with_extension(format!("{}-{}.tmp",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            let written=(|| {
                let mut wav=hound::WavWriter::create(&temporary,hound::WavSpec { channels:pcm.planes.len() as u16,
                    sample_rate:pcm.sample_rate,bits_per_sample:32,sample_format:hound::SampleFormat::Float }).map_err(|e|e.to_string())?;
                for frame in 0..pcm.planes[0].len() { for plane in pcm.planes { wav.write_sample(plane[frame]).map_err(|e|e.to_string())?; } }
                wav.finalize().map_err(|e|e.to_string())?;
                std::fs::rename(&temporary,&path).map_err(|e|e.to_string())
            })();
            if written.is_err() {let _=std::fs::remove_file(&temporary);}written?;
        }
        let path=path.to_string_lossy().into_owned(); paths.insert(pcm.persistent_id,path.clone());
        reverse.insert(path,pcm.persistent_id.to_owned());
    }
    for clip in &mut timeline.clips {
        clip.source_path=clip.source_path.as_deref().and_then(|id|paths.get(id).cloned());
        clip.source_path_relative=None;
        for take in &mut clip.takes {
            let pcm=known[take.source_path.as_deref().ok_or("missing host PCM")?];
            take.source_path=Some(paths[pcm.persistent_id].clone()); take.source_path_relative=None;
            take.source_sample_rate=Some(pcm.sample_rate); take.source_channels=Some(pcm.planes.len() as u16);
            take.duration_frames=Some(pcm.planes[0].len() as u64); take.duration_sec=Some(pcm.planes[0].len() as f64/pcm.sample_rate as f64);
            take.source_file_fingerprint=None; take.source_file_mtime=None; take.source_file_size=None;
            take.waveform_preview=None; take.pitch_range=None;
        }
        clip.normalize_takes();
    }
    Ok((timeline,reverse))
}

/// 完整逐样本核对授权PCM后才复用私有分析文件；位置/名称变化保持mtime与原线cache。
/// 文件损坏重建采用临时文件原子替换，不允许分析worker读到半个WAV。
fn matches_authorized_pcm(path:&Path,pcm:&PcmView<'_>)->bool {
    let Ok(mut wav)=hound::WavReader::open(path) else {return false;};let spec=wav.spec();
    if spec.sample_rate!=pcm.sample_rate||spec.channels as usize!=pcm.planes.len()
        ||spec.bits_per_sample!=32||spec.sample_format!=hound::SampleFormat::Float
        ||wav.duration() as usize!=pcm.planes[0].len() {return false;}
    let mut samples=wav.samples::<f32>();
    for frame in 0..pcm.planes[0].len() {for plane in pcm.planes {
        if !matches!(samples.next(),Some(Ok(sample)) if sample.to_bits()==plane[frame].to_bits()) {return false;}
    }}samples.next().is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 位置/名称不影响文件身份和mtime；同ID新PCM独立命名，私有文件损坏从授权源重建。
    #[test]
    fn materialized_analysis_file_reuses_content_and_repairs_corruption() {
        let dir=std::env::temp_dir().join(format!("hfs-host-pcm-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"T","order":0}],"bpm":120,"project_sec":1,
            "clips":[{"id":"clip","name":"C","track_id":"track","start_sec":0,"length_sec":4.0/44100.,
                "takes":[{"id":"take","source_path":"same-host-id","source_end_sec":4.0/44100.}]}]
        })).unwrap();let planes=vec![vec![0.1,0.2,0.3,0.4]];
        let view=PcmView {persistent_id:"same-host-id",sample_rate:44100,planes:&planes};
        let (mut first,_)=materialize(timeline.clone(),&[view],&dir).unwrap();let path=first.clips[0].source_path.clone().unwrap();
        let modified=std::fs::metadata(&path).unwrap().modified().unwrap();first.clips[0].start_sec=10.;
        first.clips[0].name="Moved and renamed".into();first.clips[0].source_path=Some("same-host-id".into());
        first.clips[0].takes[0].source_path=Some("same-host-id".into());
        let (moved,_)=materialize(first,&[PcmView {persistent_id:"same-host-id",sample_rate:44100,planes:&planes}],&dir).unwrap();
        assert_eq!(moved.clips[0].source_path.as_deref(),Some(path.as_str()));assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(),modified);
        std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(4).unwrap();
        let (repaired,_)=materialize(timeline.clone(),&[PcmView {persistent_id:"same-host-id",sample_rate:44100,planes:&planes}],&dir).unwrap();
        assert_eq!(repaired.clips[0].source_path.as_deref(),Some(path.as_str()));assert!(matches_authorized_pcm(Path::new(&path),
            &PcmView {persistent_id:"same-host-id",sample_rate:44100,planes:&planes}));
        let changed=vec![vec![0.5,0.6,0.7,0.8]];let (next,_)=materialize(timeline,&[PcmView {persistent_id:"same-host-id",sample_rate:44100,planes:&changed}],&dir).unwrap();
        assert_ne!(next.clips[0].source_path.as_deref(),Some(path.as_str()));
    }
}
