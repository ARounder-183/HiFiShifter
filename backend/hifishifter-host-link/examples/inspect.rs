//! 诊断工具：读一份工作区并打印它的全部读数。
//!
//! 交接是文件，所以"用户出问题了给我看什么"这个问题的答案从"复现一次连接"
//! 变成"把这个目录发给我"。这个工具就是把那个目录变成一段可读的报告 ——
//! 也是 `docs/host-link.md` 里那份「症状 → 原因」表的新后端。
//!
//! ```text
//! cargo run -p hifishifter-host-link --example inspect -- <工作区目录> [--audio]
//! ```

use hifishifter_host_link::{AudioState, Workspace};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .map(std::path::PathBuf::from)
        .ok_or("usage: inspect <workspace directory> [--audio]")?;
    let check_audio = args.any(|arg| arg == "--audio");
    let workspace = Workspace::at(&root);

    println!("workspace  {}", workspace.root().display());
    if !workspace.looks_like_workspace() {
        return Err(format!(
            "{} has no {} — this is not a workspace",
            root.display(),
            hifishifter_host_link::MANIFEST_FILE
        ));
    }
    let manifest = workspace.read_manifest()?;
    println!("identity   {}", manifest.link_id);
    println!(
        "source     {:?} / {}",
        manifest.source_kind, manifest.host_label
    );
    println!(
        "producer   {} (pid {})",
        manifest.producer.implementation,
        manifest
            .producer
            .pid
            .map_or_else(|| "-".into(), |pid| pid.to_string())
    );
    println!(
        "revisions  edits={} model={}",
        manifest.revisions.edits, manifest.revisions.model
    );
    println!("admission  {:?}", manifest.admission);
    println!(
        "timeline   {} clips on {} tracks",
        manifest.clip_count(),
        manifest.track_count()
    );
    let (complete, partial, external) = manifest.audio_summary();
    println!("audio      {complete} complete / {partial} partial / {external} extern");

    for source in &manifest.sources {
        let state = match source.state {
            AudioState::Complete => "complete",
            AudioState::Partial => "PARTIAL ",
            AudioState::Extern => "extern  ",
        };
        println!(
            "  {state} {} ({} Hz, {} ch, {} frames){}",
            source.key,
            source.sample_rate,
            source.channels,
            source.frames,
            source
                .note
                .as_deref()
                .map(|note| format!(" — {note}"))
                .unwrap_or_default()
        );
        if check_audio {
            match workspace.read_audio(source) {
                Ok(loaded) => println!(
                    "            verified: {} ch, {} frames @ {} Hz",
                    loaded.planes.len(),
                    loaded.planes.first().map_or(0, Vec::len),
                    loaded.sample_rate
                ),
                Err(error) => println!("            UNREADABLE: {error}"),
            }
        }
    }

    if !manifest.losses.is_empty() {
        println!(
            "losses     {} item(s) did not come across",
            manifest.losses.len()
        );
        for loss in manifest.losses.iter().take(16) {
            println!(
                "  {} — {}{}",
                loss.id,
                loss.reason,
                loss.detail
                    .as_deref()
                    .map(|detail| format!(" ({detail})"))
                    .unwrap_or_default()
            );
        }
    }

    match workspace.read_status() {
        Some(status) if status.is_current(&manifest.link_id) => println!(
            "last edit  {} at {} {}",
            if status.ok { "accepted" } else { "REJECTED" },
            status.updated_unix_ms,
            status.error.unwrap_or_default()
        ),
        Some(status) => println!(
            "last edit  belongs to another link ({}) — stale status file",
            status.link_id
        ),
        None => println!("last edit  none has been submitted"),
    }

    println!(
        "edit file  {}",
        match workspace.edit_stamp_ms() {
            Some(stamp) => format!("present (stamp {stamp})"),
            None => "absent".to_string(),
        }
    );
    if let Some(holder) = workspace.open_mark(12 * 60 * 60 * 1000) {
        println!("open mark  held by pid {}", holder.pid);
    }
    Ok(())
}
