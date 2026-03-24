pub(super) fn open_audio_dialog() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("Audio", &["wav", "flac", "mp3", "ogg", "m4a"])
        .pick_file();

    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}

pub(super) fn open_audio_dialog_multi() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("Audio", &["wav", "flac", "mp3", "ogg", "m4a"])
        .pick_files();

    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(paths) => {
            let path_list: Vec<String> = paths
                .into_iter()
                .map(|path| path.display().to_string())
                .collect();
            serde_json::json!({"ok": true, "canceled": false, "paths": path_list})
        }
    }
}

pub(super) fn pick_output_path() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("WAV", &["wav"])
        .set_file_name("output.wav")
        .save_file();

    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}

pub(super) fn pick_directory() -> serde_json::Value {
    let picked = rfd::FileDialog::new().pick_folder();

    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}

pub(super) fn open_midi_dialog() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("MIDI", &["mid", "midi"])
        .pick_file();

    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}
