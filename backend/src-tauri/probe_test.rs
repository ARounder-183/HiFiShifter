    #[test]
    fn probe_v4_flat_project_targets() {
        let dir = std::env::temp_dir().join("hifishifter_probe_v4");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("probe.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&wav, spec).unwrap();
        for i in 0..44_100 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        let src = wav.to_string_lossy().replace('\\', "/");

        let value = serde_json::json!({
            "version": 4,
            "name": "legacy",
            "timeline": {
                "tracks": [{
                    "id": "track_1", "name": "Track", "order": 0,
                    "muted": false, "solo": false, "volume": 1.0,
                    "compose_enabled": false,
                    "pitch_analysis_algo": "nsf_hifigan_onnx",
                    "color": "#4a8fd1"
                }],
                "clips": [{
                    "id": "clip_1", "track_id": "track_1", "name": "Legacy Clip",
                    "start_sec": 0.0, "length_sec": 1.0, "color": "blue",
                    "source_path": src,
                    "duration_sec": 1.0, "duration_frames": 44100,
                    "source_sample_rate": 44100,
                    "gain": 1.0, "muted": false,
                    "source_start_sec": 0.0, "source_end_sec": 1.0,
                    "playback_rate": 1.0, "reversed": false,
                    "channel_mode": 0,
                    "fade_in_sec": 0.0, "fade_out_sec": 0.0,
                    "fade_in_curve": "sine", "fade_out_curve": "sine"
                }],
                "bpm": 120.0, "playhead_sec": 0.0, "project_sec": 1.0,
                "next_track_order": 1
            }
        });
        let bytes = serde_json::to_vec(&value).unwrap();
        let loaded = match crate::project::load_project_file(&bytes) {
            Ok(v) => v,
            Err(e) => {
                println!("PARSE ERR: {e}");
                return;
            }
        };
        let (fin, _m) = crate::project::finalize_timeline_for_session(
            loaded.timeline,
            std::path::Path::new(&wav),
            4,
        );
        let clip = &fin.clips[0];
        println!(
            "V4FLAT: takes={} clip.src={:?} take0.src={:?} take0.region={:?}",
            clip.takes.len(),
            clip.source_path,
            clip.takes.first().and_then(|t| t.source_path.clone()),
            clip.takes
                .first()
                .map(|t| (t.source_start_sec, t.source_end_sec))
        );
        let policy = crate::config::channel_import_policy().for_explicit_scan();
        let filter: std::collections::HashSet<String> =
            [clip.id.clone()].into_iter().collect();
        println!(
            "V4FLAT TARGETS(filtered)={} TARGETS(all)={}",
            crate::commands::channel_scan::collect_targets(&fin, Some(&filter), &policy, true).len(),
            crate::commands::channel_scan::collect_targets(&fin, None, &policy, true).len()
        );
        let _ = std::fs::remove_file(&wav);
    }
