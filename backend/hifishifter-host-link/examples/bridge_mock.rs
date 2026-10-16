//! **参考生产者**：一个不需要 DAW 的假宿主，用来跑通整条交接。
//!
//! 它写一份工作区（两条 clip、内嵌正弦 PCM），然后守着 `edits.hfsedit`：
//! 文件一出现就"接受"它并把结果写进 `status.hfslink`。
//!
//! # 它为什么存在
//!
//! 1. **第三方宿主扩展的参考实现。** 契约是文件，所以一个能跑的样板比一段文字更清楚 ——
//!    原来那份管道契约要求实现者写一个 Windows 命名管道服务端（`windows-sys` FFI、
//!    SDDL、心跳、令牌），这里只需要写文件。
//! 2. **本仓没有非 ARA 宿主。** `docs/plans/2026-10-15-vst-transfer-mode.md` §18.6 记过
//!    这条缺口（"真机验收未做：本仓没有非 ARA 宿主可跑"）。有了它，消费侧
//!    （打开 / 编辑 / 提交 / 看状态）可以完全不依赖 DAW 地被端到端验证。
//!
//! ```text
//! cargo run -p hifishifter-host-link --example bridge_mock -- <工作区目录>
//! ```

use hifishifter_host_link::{
    link_id, now_ms, AudioState, Capability, EditManifest, LinkManifest, LinkStatus, Producer,
    Revisions, SourceEntry, SourceKind, Workspace, FORMAT, FORMAT_VERSION,
};
use std::path::PathBuf;
use std::time::Duration;

fn main() -> Result<(), String> {
    let root: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            hifishifter_host_link::workspace::default_root()
                .unwrap_or_else(std::env::temp_dir)
                .join("Mock Vegas Bridge")
        });
    let workspace = Workspace::at(root);
    let source_kind = SourceKind::Bridge;
    let identity = link_id(source_kind, "Mock Vegas Bridge", "demo-project");

    let (sources, clips) = build_content(&workspace)?;
    let manifest = LinkManifest {
        format: FORMAT.into(),
        format_version: FORMAT_VERSION,
        link_id: identity.clone(),
        source_kind,
        host_label: "Mock Vegas Bridge".into(),
        created_unix_ms: now_ms(),
        producer: Producer {
            implementation: format!("bridge_mock {}", env!("CARGO_PKG_VERSION")),
            pid: Some(std::process::id()),
        },
        timeline: serde_json::json!({
            "bpm": 120.0,
            "project_sec": 4.0,
            "tracks": [{"id": "mock-track-1", "name": "Mock Track", "order": 0, "volume": 1.0}],
            "clips": clips,
        }),
        // 基线由生产者算 —— 桥接实现里这就是"我会拿什么去比"。
        // 【为什么用时间轴自己算一份】桥接没有内核的投影函数，所以直接送一份
        // 结构相同的投影；真正的第三方实现按自己的字段学问填即可。
        host_geometry: serde_json::json!({
            "clips": clips.iter().map(|clip| serde_json::json!({
                "id": clip["id"],
                "track_id": clip["track_id"],
                "start_sec": clip["start_sec"],
                "length_sec": clip["length_sec"],
                "source_start_sec": clip["source_start_sec"],
                "source_end_sec": clip["source_end_sec"],
                "playback_rate": clip["playback_rate"],
            })).collect::<Vec<_>>(),
            "tracks": [{"id": "mock-track-1", "name": "Mock Track", "order": 0}],
        }),
        sources,
        // 桥接声明它会接受提交与几何写回 —— 这正是它与 ARA 的分野（见 SourceKind::admission）。
        granted: vec![Capability::Commit, Capability::GeometryWrite, Capability::Audio],
        revisions: Revisions { edits: 0, model: 1 },
        admission: source_kind.admission(),
        losses: Vec::new(),
        diagnostics: None,
    };
    workspace.write_manifest(&manifest)?;
    workspace.write_status(&LinkStatus::initial(&identity, manifest.revisions))?;
    println!("workspace ready: {}", workspace.root().display());

    let mut seen = workspace.edit_stamp_ms();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let stamp = workspace.edit_stamp_ms();
        if stamp == seen {
            continue;
        }
        seen = stamp;
        // 半个文件（正好写在当口）不是错误：下一轮会读到完整的。
        let Ok(edit) = workspace.read_edit() else {
            continue;
        };
        match respond(&workspace, &identity, &edit) {
            Ok(()) => println!(
                "accepted an edit: {} tracks, base revisions {:?}",
                edit.tracks.as_array().map_or(0, Vec::len),
                edit.base_revisions
            ),
            Err(error) => println!("rejected an edit: {error}"),
        }
    }
}

/// 回答一次编辑，并把回答写进 `status.hfslink`。
///
/// 【为什么身份对不上要单独报】`link_id` 是「这是哪一次交接」。把一份属于别的工作区的
/// 编辑合并进来，等于把用户在一个工程里的曲线贴到另一个工程上 —— 必须是一次明确的拒绝。
///
/// 【为什么"成功"也要写状态】管道时代"提交成功"只存在于那个连接的回包里，
/// 用户关掉 App 就再也查不到自己到底提交成功了没有。
fn respond(workspace: &Workspace, identity: &str, edit: &EditManifest) -> Result<(), String> {
    let mismatch = edit.link_id != identity;
    let message = mismatch.then(|| format!("edit belongs to {}, not {identity}", edit.link_id));
    workspace.write_status(&LinkStatus {
        format: hifishifter_host_link::STATUS_FORMAT.into(),
        format_version: FORMAT_VERSION,
        link_id: identity.to_string(),
        ok: !mismatch,
        code: mismatch.then_some(hifishifter_host_link::error_code::LINK_MISMATCH),
        error: message.clone(),
        revisions: Revisions {
            edits: if mismatch {
                edit.base_revisions.edits
            } else {
                edit.base_revisions.edits + 1
            },
            model: edit.base_revisions.model,
        },
        updated_unix_ms: now_ms(),
    })?;
    match message {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// 两条 clip、两份素材，PCM 是两条不同频率的正弦。
fn build_content(workspace: &Workspace) -> Result<(Vec<SourceEntry>, Vec<serde_json::Value>), String> {
    let rate = 48_000;
    let mut sources = Vec::new();
    let mut clips = Vec::new();
    for (index, (key, hz, start)) in [
        ("mock://tone-a", 220.0_f32, 0.0_f64),
        ("mock://tone-b", 330.0_f32, 2.0),
    ]
    .into_iter()
    .enumerate()
    {
        let frames = rate as usize * 2;
        let left: Vec<f32> = (0..frames)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.25)
            .collect();
        let right = left.clone();
        let audio = workspace.write_audio(key, rate, &[left, right])?;
        sources.push(SourceEntry {
            key: key.into(),
            sample_rate: rate,
            channels: 2,
            frames: frames as u64,
            state: AudioState::Complete,
            file: Some(audio.file.clone()),
            blake3: Some(audio.blake3),
            bytes: Some(audio.bytes),
            note: None,
        });
        let id = format!("mock-clip-{}", index + 1);
        clips.push(serde_json::json!({
            "id": id,
            "track_id": "mock-track-1",
            "name": format!("Tone {hz} Hz"),
            "start_sec": start,
            "length_sec": 2.0,
            "source_path": key,
            "source_start_sec": 0.0,
            "source_end_sec": 2.0,
            "playback_rate": 1.0,
        }));
    }
    Ok((sources, clips))
}
