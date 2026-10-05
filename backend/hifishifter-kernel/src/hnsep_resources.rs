//! 整段HNSEP推理前资源保护；不切块、不改样本或退化为未处理原声。
//! 保守模型工作区估计来自实测算子形状，不等同插件PCM额度或承诺绝对不会OOM。

const MODEL_RATE:u64=44100;
const HOP:u64=512;
const SEGMENT:u64=32;
const SAFETY_BYTES:u64=512*1024*1024;

/// 原模型dec1_2的Concat同时保留65+32输入与97输出；按完整对齐谱帧估计峰值。
/// 加入复数STFT/掩码等工作域及固定模型开销。新模型仍需重新实测，不能把它当通用上界。
pub(crate) fn estimated_working_bytes(frames:usize,sample_rate:u32)->Result<u64,String> {
    if sample_rate==0 {return Err("HNSEP资源预检：采样率无效".into());}
    let samples=u64::try_from(frames).ok().and_then(|n|n.checked_mul(MODEL_RATE))
        .map(|n|n.div_ceil(sample_rate as u64)).ok_or("HNSEP资源预检：样本数溢出")?;
    let aligned=samples.checked_add(HOP).and_then(|n|n.div_ceil(HOP*SEGMENT).checked_mul(SEGMENT))
        .ok_or("HNSEP资源预检：谱帧数溢出")?;
    let concat_per_frame=2_u64*97*1024*4;
    let spectral_per_frame=1025_u64*48;
    aligned.checked_mul(concat_per_frame+spectral_per_frame).and_then(|n|n.checked_add(256*1024*1024))
        .ok_or_else(||"HNSEP资源预检：工作区大小溢出".into())
}

/// 成功缓存命中后无需本检查；真正STFT/网络分配前检查系统可用物理及提交空间。
pub(crate) fn preflight(frames:usize,sample_rate:u32)->Result<(),String> {
    let required=estimated_working_bytes(frames,sample_rate)?;
    if let Some(available)=available_memory()? {check_available(required,available)?;}
    else {log::warn!("HNSEP资源预检：此平台可用内存尚未核对，估计整段工作区{required}字节；不能声称资源门已通过");}
    Ok(())
}

/// 短缺内存不重试分配巨型Tensor；调用方显示失败并保留旧就绪音频/编辑权威。
fn check_available(required:u64,available:u64)->Result<(),String> {
    let guarded=required.checked_add(SAFETY_BYTES).ok_or("HNSEP资源预检：额度溢出")?;
    if guarded>available {return Err(format!("HNSEP整段处理内存不足：预计工作区约{:.2} GiB（另保留0.50 GiB），当前可用约{:.2} GiB；未执行推理，未切块或忽略气声/张力。",required as f64/1073741824.,available as f64/1073741824.));}
    Ok(())
}

/// Windows使用官方typed系统接口，Linux只读MemAvailable；其它平台不猜测可用内存。
fn available_memory()->Result<Option<u64>,String> {
    #[cfg(target_os="windows")]
    {
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx,MEMORYSTATUSEX};
        let mut state=MEMORYSTATUSEX {dwLength:std::mem::size_of::<MEMORYSTATUSEX>() as u32,..Default::default()};
        // SAFETY: 结构与大小取自官方windows绑定，在同步系统调用期间保持可写。
        unsafe {GlobalMemoryStatusEx(&mut state)}.map_err(|error|format!("HNSEP资源预检：系统内存查询失败：{error}"))?;
        Ok(Some(state.ullAvailPhys.min(state.ullAvailPageFile)))
    }
    #[cfg(target_os="linux")]
    {let text=std::fs::read_to_string("/proc/meminfo").map_err(|error|format!("HNSEP资源预检：读取MemAvailable失败：{error}"))?;
        parse_linux_available(&text).map(Some)}
    #[cfg(not(any(target_os="windows",target_os="linux")))]
    {Ok(None)}
}

#[cfg(any(target_os="linux",test))]
fn parse_linux_available(text:&str)->Result<u64,String> {
    let line=text.lines().find(|line|line.starts_with("MemAvailable:")).ok_or("HNSEP资源预检：没有MemAvailable")?;
    let mut fields=line.split_whitespace();fields.next();let value=fields.next().and_then(|v|v.parse::<u64>().ok()).ok_or("HNSEP资源预检：MemAvailable无效")?;
    if fields.next()!=Some("kB") {return Err("HNSEP资源预检：MemAvailable单位不支持".into());}
    value.checked_mul(1024).ok_or_else(||"HNSEP资源预检：MemAvailable溢出".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 估计严格使用整段谱帧，不按HiFiGAN神经块低估HNSEP；缺内存明确拒绝。
    #[test]
    fn full_length_resource_estimate_rejects_small_ram_without_duration_cap() {
        let short=estimated_working_bytes(44100,44100).unwrap();let long=estimated_working_bytes(44100*180,44100).unwrap();
        assert!(short<512*1024*1024);assert!(long>12*1024*1024*1024);
        assert_eq!(long,estimated_working_bytes(48000*180,48000).unwrap());
        assert!(check_available(long,8*1024*1024*1024).unwrap_err().contains("未执行推理"));
        assert!(check_available(long,24*1024*1024*1024).is_ok());
        assert!(estimated_working_bytes(usize::MAX,1).is_err());assert!(estimated_working_bytes(1,0).is_err());
    }
    #[test]
    fn linux_available_parser_uses_available_not_total_and_checks_units() {
        assert_eq!(parse_linux_available("MemTotal: 1000 kB\nMemAvailable: 24 kB\n").unwrap(),24576);
        for invalid in ["MemTotal: 24 kB","MemAvailable: 24 MB","MemAvailable: none kB"] {assert!(parse_linux_available(invalid).is_err());}
    }
}
