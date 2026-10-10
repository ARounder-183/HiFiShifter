//! 在 ARA 授权回调 scope 内读取源 PCM，reader 不进入快照或音频线程。

use ara2_bridge::core::AraError;
use ara2_bridge::plugin::HostContentScope;

#[derive(Debug)]
pub(crate) struct SourcePcm {
    pub sample_rate: u32,
    pub planes: Vec<Vec<f32>>,
    pub version: u64,
    pub _reservation: Option<super::budget::Reservation>,
}

/// 完整读取后才返回；任何失败必须释放 reader，不发布部分音频。
pub(crate) fn read_source_pcm(
    host: &HostContentScope<'_, '_>,
    frames: usize,
    channels: usize,
    sample_rate: u32,
    version: u64,
) -> Result<SourcePcm, AraError> {
    // 【为什么不再限定 44.1/48k】宿主工程与素材都可能是 88.2/96/176.4/192k（母带、后期
    // 很常见）。此前直接 `Unsupported` ⇒ 插件**一条 region 都拿不到** ⇒ 整个插件静默
    // 不可用。读取路径是**离线**的（不在音频回调里），而下游两处都已经会重采样：
    // `mix_plain_regions` 按 `source.sample_rate` 线性取点，内核路径走
    // `linear_resample_interleaved`。所以这里只需要挡明显不合理的值。
    if !(8000..=384_000).contains(&sample_rate)
        || !(1..=2).contains(&channels)
        || frames == 0
        || frames > i64::MAX as usize
    {
        return Err(AraError::Unsupported(
            "ARA requires nonempty mono/stereo sources within 8k..=384kHz",
        ));
    }
    let source = host
        .current_audio_source()
        .ok_or(AraError::InvalidState("missing source scope"))?;
    let bytes = frames
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(4))
        .ok_or(AraError::InvalidState("PCM memory size overflow"))?;
    let reservation = super::budget::global_budget()
        .reserve(bytes)
        .ok_or(AraError::InvalidState("PCM memory budget exceeded"))?;
    let mut reader = host.audio_reader::<f32>(source, channels)?;
    let mut planes = Vec::with_capacity(channels);
    for _ in 0..channels {
        let mut plane = Vec::new();
        plane
            .try_reserve_exact(frames)
            .map_err(|_| AraError::InvalidState("PCM allocation failed"))?;
        plane.resize(frames, 0_f32);
        planes.push(plane);
    }
    for start in (0..frames).step_by(4096) {
        let end = (start + 4096).min(frames);
        let mut buffers = planes
            .iter_mut()
            .map(|plane| &mut plane[start..end])
            .collect::<Vec<_>>();
        reader.read(start as i64, &mut buffers)?;
    }
    if planes.iter().flatten().any(|sample| !sample.is_finite()) {
        return Err(AraError::Peer("host returned nonfinite PCM"));
    }
    Ok(SourcePcm {
        sample_rate,
        planes,
        version,
        _reservation: Some(reservation),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::{HostAudioSourceRef, HostClients};
    use std::sync::atomic::Ordering;
    /// 真ARA reader在4096帧窗口内读取45秒授权源，末块与预算释放都不依赖旧30秒常量。
    #[test]
    fn long_authorized_source_reads_the_tail_and_releases_its_budget() {
        let frames = 44100 * 45 + 17;
        let samples = (0..frames)
            .map(|n| (n % 17) as f32 / 32.)
            .collect::<Vec<_>>();
        let mut fixture = crate::test_host::HostFixture::new(vec![samples]);
        let host = fixture.instance();
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0_u8;
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let baseline = super::super::budget::global_budget().used();
        let pcm = clients
            .with_audio_source_management(source, |scope| {
                read_source_pcm(&scope, frames, 1, 44100, 3)
            })
            .unwrap();
        assert_eq!(pcm.planes[0], fixture.planes[0]);
        assert_eq!(pcm.planes[0].len(), frames);
        assert!(super::super::budget::global_budget().used() >= baseline + frames * 4);
        drop(pcm);
        assert_eq!(super::super::budget::global_budget().used(), baseline);
        assert_eq!(fixture.created.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.destroyed.load(Ordering::Relaxed), 1);
        assert!(clients
            .with_audio_source_management(source, |scope| read_source_pcm(
                &scope,
                usize::MAX / 4,
                2,
                44100,
                4
            ))
            .is_err());
        assert_eq!(
            fixture.created.load(Ordering::Relaxed),
            1,
            "超预算应在打开host reader之前拒绝"
        );
    }

    /// 通过真实 bridge reader 读完整数据，并在 scope 返回前释放它。
    #[test]
    fn scoped_reader_copies_pcm_and_releases_the_host_reader() {
        let mut fixture = crate::test_host::HostFixture::new(vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
        ]);
        let host = fixture.instance();
        // SAFETY: fixture 与接口存储保留到 clients 和 reader 已释放。
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0_u8;
        // SAFETY: source host ref 仅在本 scope 作不透明身份。
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let pcm = clients
            .with_audio_source_management(source, |scope| read_source_pcm(&scope, 4, 2, 44100, 7))
            .unwrap();
        assert_eq!(
            pcm.planes,
            [vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]]
        );
        assert_eq!(pcm.version, 7);
        assert_eq!(fixture.created.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.destroyed.load(Ordering::Relaxed), 1);
    }

    /// 宿主读失败不能泄漏 reader，也不能返回静默补零的伪成功数据。
    #[test]
    fn host_read_failure_drops_the_reader_without_publishing_partial_pcm() {
        let mut fixture = crate::test_host::HostFixture::new(vec![vec![0.1, 0.2]]);
        let host = fixture.instance();
        // SAFETY: fixture 保留到 clients 释放。
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0_u8;
        // SAFETY: 不透明身份在读取期间存活。
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        assert!(clients
            .with_audio_source_management(source, |scope| read_source_pcm(&scope, 4, 1, 44100, 0))
            .is_err());
        assert_eq!(fixture.created.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.destroyed.load(Ordering::Relaxed), 1);
    }
}
