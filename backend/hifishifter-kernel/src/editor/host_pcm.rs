//! 宿主PCM的私有分析副本：只写调用方提供的私有目录，绝不把宿主ID当文件读取。
use crate::state::TimelineState;
use std::collections::HashMap;
use std::path::Path;
pub struct PcmView<'a> {
    pub persistent_id:&'a str,
    pub sample_rate:u32,
    pub planes:&'a [Vec<f32>],
}

/// 在所有源/几何校验成功后生成只用于波形与分析的WAV，返回路径→宿主ID反向表。
/// 音频渲染仍从ARA授权PCM注入；不回读这些文件作为宿主供音。
pub fn materialize(mut timeline:TimelineState,sources:&[PcmView<'_>],dir:&Path)
    ->Result<(TimelineState,HashMap<String,String>),String> {
    if !timeline.bpm.is_finite() || timeline.bpm<=0. || !timeline.project_sec.is_finite() || timeline.project_sec<0. {
        return Err("invalid host timeline".into());
    }
    let mut known=HashMap::new();
    let mut bytes=0_usize;
    for pcm in sources {
        let frames=pcm.planes.first().map(Vec::len).unwrap_or(0);
        bytes=bytes.checked_add(frames.checked_mul(pcm.planes.len()).and_then(|n|n.checked_mul(4)).ok_or("host PCM size overflow")?).ok_or("host PCM size overflow")?;
        if bytes>64*1024*1024 || pcm.persistent_id.is_empty() || !matches!(pcm.sample_rate,44100|48000)
            || !(1..=2).contains(&pcm.planes.len()) || frames==0 || frames>pcm.sample_rate as usize*30
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
        {
            let mut wav=hound::WavWriter::create(&path,hound::WavSpec { channels:pcm.planes.len() as u16,
                sample_rate:pcm.sample_rate,bits_per_sample:32,sample_format:hound::SampleFormat::Float }).map_err(|e|e.to_string())?;
            for frame in 0..pcm.planes[0].len() { for plane in pcm.planes { wav.write_sample(plane[frame]).map_err(|e|e.to_string())?; } }
            wav.finalize().map_err(|e|e.to_string())?;
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
