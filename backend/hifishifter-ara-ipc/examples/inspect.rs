//! 本地一次性通信诊断，只请求宿主快照并输出摘要，不输出token或音频。
fn main() {
    let instances = hifishifter_ara_ipc::discover().expect("instance discovery");
    for instance in instances {
        match hifishifter_ara_ipc::exchange(&instance, &hifishifter_ara_ipc::Request::Snapshot) {
            Ok(response) => println!(
                "{} ok={} error={:?} revision={} clips={} sources={}",
                instance.name,
                response.ok,
                response.error,
                response.revision,
                response
                    .timeline
                    .as_ref()
                    .and_then(|t| t["clips"].as_array())
                    .map_or(0, Vec::len),
                response.sources.len()
            ),
            Err(error) => println!("{} transport error: {error}", instance.name),
        }
    }
}
