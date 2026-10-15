//! 最小桥接参考实现：模拟一个"宿主侧扩展"，让独立 App 能连上并拉时间轴。
//!
//! 【为什么仓库里要有它】对外契约（`docs/host-timeline-channel.md`）是给**另一个
//! 仓库**的作者看的；在 HiFiShifter 侧留一个能跑的最小实现，接收端就不必等 Vegas
//! 扩展写完才能验收。它说的是同一门语言：写实例记录 → 起命名管道 → 回
//! `hello` / `snapshot` / `commit`。
//!
//! 【它不是 ARA】这条通路完全不经过 ARA SDK；`source_kind` 声明为 `bridge`，
//! 客户端因此按"逐条降级"的准入策略对待它的载荷。
//!
//! 用法（仓库根）：
//! `cargo run -p hifishifter-host-channel --example bridge_mock`
//!
//! 然后在独立 App 的「宿主」面板里刷新实例，会看到 `Mock Vegas Bridge`，
//! 来源标为 Bridge；连上后应出现 1 条轨道 + 2 条片段，且能听到正弦波。
//! Ctrl+C 退出 —— 被强杀的进程会留下一条记录文件，它在 15 秒无心跳后自动从
//! 列表里消失（见契约 §3.3）。

#[cfg(windows)]
fn main() {
    let server = hifishifter_host_channel::Server::start_as(
        "Mock Vegas Bridge".into(),
        hifishifter_host_channel::SourceKind::Bridge,
        handle,
    )
    .expect("start the bridge channel");
    println!("bridge instance : {}", server.record().instance_id);
    println!("pipe            : {}", server.record().pipe_name);
    println!("instance dir    : {}", hifishifter_host_channel::instance_dir().unwrap().display());
    println!("connect from the standalone app's host panel; Ctrl+C to stop");
    // 心跳由 `Server` 自己的工作线程维持；主线程只需要活着。
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("the host timeline channel is Windows-only");
}

#[cfg(windows)]
fn handle(
    request: hifishifter_host_channel::Request,
) -> hifishifter_host_channel::Reply {
    use hifishifter_host_channel::{
        Capability, HelloResponse, Reply, Request, Response, SourceKind, TransportState,
        TIMELINE_SCHEMA_VERSION,
    };
    match request {
        // 能力协商**必须**是第一个请求。声明了就做到：这三个能力本实现都真的实现了。
        Request::Hello { .. } => Reply::Hello(HelloResponse {
            source_kind: SourceKind::Bridge,
            timeline_schema: TIMELINE_SCHEMA_VERSION,
            granted: vec![
                Capability::HostPcm,
                Capability::Commit,
                Capability::Transport,
            ],
            host_label: "Vegas Pro 22 (mock)".into(),
            implementation: format!("bridge_mock {}", env!("CARGO_PKG_VERSION")),
        }),
        Request::Snapshot => Reply::Response(Response {
            ok: true,
            revision: 1,
            model_revision: 1,
            timeline: Some(timeline()),
            sources: sources(),
            transport: Some(TransportState {
                position_sec: 0.0,
                playing: false,
                tempo: Some(120.0),
                time_signature: Some((4, 4)),
            }),
            ..Default::default()
        }),
        // 乐观并发：`base_revision` 对不上就是冲突，客户端会提示"宿主已改变，请刷新"。
        Request::Commit { base_revision, .. } => Reply::Response(Response {
            ok: base_revision == 1,
            error: (base_revision != 1).then(|| "Conflict: host model changed".to_string()),
            revision: 2,
            model_revision: 1,
            ..Default::default()
        }),
    }
}

/// 一个合法的最小 `TimelineState`。
///
/// 【为什么只写 `takes`】`Clip` 上的媒体字段（`source_path` / `source_start_sec` …）
/// 是活跃 Take 的**内存投影**，`skip_serializing` —— 写了也不会被读，权威数据在
/// `takes` 里（见契约 §6.3）。
#[cfg(windows)]
fn timeline() -> serde_json::Value {
    serde_json::json!({
        "tracks": [{"id": "bridge-track", "name": "Bridge Vocal", "order": 0, "volume": 1.0}],
        "clips": [
            {
                "id": "bridge-clip-a",
                "track_id": "bridge-track",
                "name": "Verse",
                "start_sec": 0.0,
                "length_sec": 1.0,
                "takes": [{
                    "id": "bridge-take-a",
                    "source_path": "mock://media/vocal",
                    "source_start_sec": 0.0,
                    "source_end_sec": 1.0,
                    "playback_rate": 1.0
                }]
            },
            {
                "id": "bridge-clip-b",
                "track_id": "bridge-track",
                "name": "Chorus",
                "start_sec": 1.5,
                "length_sec": 1.0,
                "takes": [{
                    "id": "bridge-take-b",
                    "source_path": "mock://media/harmony",
                    "source_start_sec": 0.0,
                    "source_end_sec": 1.0,
                    "playback_rate": 1.0
                }]
            }
        ],
        "bpm": 120.0,
        "project_sec": 3.0
    })
}

#[cfg(windows)]
fn sources() -> Vec<hifishifter_host_channel::HostPcm> {
    vec![pcm("mock://media/vocal", 220.0), pcm("mock://media/harmony", 330.0)]
}

/// 1 秒立体声正弦（内嵌形态；小素材不必走文件引用）。
#[cfg(windows)]
fn pcm(persistent_id: &str, frequency: f64) -> hifishifter_host_channel::HostPcm {
    const RATE: u32 = 48_000;
    let samples: Vec<f32> = (0..RATE)
        .map(|frame| {
            let seconds = f64::from(frame) / f64::from(RATE);
            (0.2 * (std::f64::consts::TAU * frequency * seconds).sin()) as f32
        })
        .collect();
    hifishifter_host_channel::HostPcm {
        persistent_id: persistent_id.to_string(),
        sample_rate: RATE,
        // 契约里 `fingerprint` 是给用户看的来源摘要；本实现直接用素材 id。
        fingerprint: persistent_id.to_string(),
        planes: vec![samples.clone(), samples],
        file: None,
    }
}
