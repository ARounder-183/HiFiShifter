//! PipeWire-based application audio capture (Linux).
//!
//! PipeWire exposes every application output stream as a node in the audio
//! graph. We locate the node for the selected PID with `pw-dump`, then record
//! it directly with `pw-cat --record --target=<node>` at the user-requested
//! rate/channels. `pw-cat` resamples to the requested format itself, so the
//! bytes on stdout are exactly the interleaved f32 frames we write to WAV.
//!
//! This path requires a running PipeWire session with `pw-dump` and `pw-cat`
//! installed; otherwise a localized error is returned.

use super::capture::{AppAudioInfo, CaptureContext, CapturePlan};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

pub fn enumerate_applications() -> Vec<AppAudioInfo> {
    let Some(nodes) = dump_audio_nodes() else {
        return Vec::new();
    };
    let mut apps: HashMap<u32, (String, String, bool)> = HashMap::new();
    for node in &nodes {
        let Some(pid) = node_process_id(node) else {
            continue;
        };
        if pid == 0 {
            continue;
        }
        let name = node_display_name(node);
        let process_name = node_process_binary(node).unwrap_or_default();
        let active = node_is_active(node);
        let entry = apps
            .entry(pid)
            .or_insert_with(|| (name.clone(), process_name.clone(), false));
        if active {
            entry.2 = true;
        }
    }
    let mut result: Vec<AppAudioInfo> = apps
        .into_iter()
        .map(|(pid, (name, process_name, is_active))| AppAudioInfo {
            id: format!("pid:{pid}"),
            name,
            process_name,
            pid,
            is_active,
        })
        .collect();
    result.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    result
}

pub fn run_app_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let CapturePlan::Application {
        pid,
        sample_rate,
        channels,
        ..
    } = plan
    else {
        unreachable!()
    };

    if Command::new("pw-dump").arg("--version").output().is_err() {
        let _ = ready_tx.send(Err("recording_error_pipewire_missing".to_string()));
        return Err("recording_error_pipewire_missing".to_string());
    }

    let nodes = dump_audio_nodes().ok_or_else(|| "recording_error_pipewire_dump".to_string())?;
    let target = nodes
        .iter()
        .filter(|node| node_process_id(node) == Some(pid))
        .filter(|node| node_is_output_stream(node))
        .find_map(node_target_name)
        .ok_or_else(|| "recording_error_app_not_found".to_string())?;

    let mut child = Command::new("pw-cat")
        .args(&[
            "--record".to_string(),
            "--target".to_string(),
            target,
            "--rate".to_string(),
            sample_rate.to_string(),
            "--channels".to_string(),
            channels.to_string(),
            "--format".to_string(),
            "f32".to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| "recording_error_pipewire_cat".to_string())?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "recording_error_pipewire_cat".to_string())?;
    let _ = ready_tx.send(Ok(()));

    let chunk_frames = 1024usize;
    let chunk_bytes = chunk_frames * channels as usize * 4;
    let mut buffer = vec![0u8; chunk_bytes];
    let mut result = Ok(());
    'outer: loop {
        if ctx.stop.load(Ordering::Relaxed) {
            break;
        }
        let mut filled = 0usize;
        while filled < chunk_bytes {
            match stdout.read(&mut buffer[filled..]) {
                Ok(0) => break 'outer,
                Ok(read) => filled += read,
                Err(_) => {
                    result = Err("recording_error_pipewire_read".to_string());
                    break 'outer;
                }
            }
        }
        if filled > 0 {
            let samples: Vec<f32> = buffer[..filled]
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();
            ctx.push(&samples);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    result
}

// ---------------------------------------------------------------------------
// `pw-dump` parsing
// ---------------------------------------------------------------------------

fn dump_audio_nodes() -> Option<Vec<Value>> {
    let output = Command::new("pw-dump").arg("--monitor").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value: Value = serde_json::from_slice(&output.stdout).ok()?;
    let array = value.as_array()?;
    Some(
        array
            .iter()
            .filter(|item| {
                item.get("type").and_then(Value::as_str)
                    == Some("PipeWire:Interface:Node")
            })
            .cloned()
            .collect(),
    )
}

fn node_props(node: &Value) -> Option<&serde_json::Map<String, Value>> {
    node.get("info")?.get("props")?.as_object()
}

fn node_process_id(node: &Value) -> Option<u32> {
    let props = node_props(node)?;
    let value = props.get("application.process.id")?;
    value
        .as_u64()
        .map(|value| value as u32)
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn node_display_name(node: &Value) -> String {
    let props = node_props(node).unwrap_or_default();
    ["application.name", "node.description", "application.process.binary", "node.name"]
        .iter()
        .find_map(|key| props.get(*key).and_then(Value::as_str))
        .map(str::to_string)
        .unwrap_or_else(|| "Application".to_string())
}

fn node_process_binary(node: &Value) -> Option<String> {
    let props = node_props(node)?;
    props
        .get("application.process.binary")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn node_is_output_stream(node: &Value) -> bool {
    let Some(props) = node_props(node) else {
        return false;
    };
    let media_class = props
        .get("media.class")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let direction = props
        .get("direction")
        .and_then(Value::as_str)
        .unwrap_or_default();
    media_class == "Audio/Stream" && direction == "output"
}

fn node_is_active(node: &Value) -> bool {
    node.get("info")
        .and_then(|info| info.get("state"))
        .and_then(Value::as_str)
        .map(|state| state == "running" || state == "idle")
        .unwrap_or(false)
}

fn node_target_name(node: &Value) -> Option<String> {
    let props = node_props(node)?;
    props
        .get("node.name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| node.get("id").and_then(Value::as_u64).map(|id| id.to_string()))
}
