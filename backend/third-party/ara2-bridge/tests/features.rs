use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Mutex;

static SDK_ENV_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
struct Case<'a> {
    name: &'a str,
    default_features: bool,
    features: &'a [&'a str],
    modules: &'a [&'a str],
}

#[test]
fn facade_feature_matrix_compiles_in_isolated_consumers() {
    let _sdk_env_guard = SDK_ENV_LOCK.lock().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let cases = [
        Case {
            name: "default",
            default_features: true,
            features: &[],
            modules: &["plugin"],
        },
        Case {
            name: "no-default-features",
            default_features: false,
            features: &[],
            modules: &[],
        },
        Case {
            name: "plugin",
            default_features: false,
            features: &["plugin"],
            modules: &["plugin"],
        },
        Case {
            name: "host",
            default_features: false,
            features: &["host"],
            modules: &["host"],
        },
        Case {
            name: "clap",
            default_features: false,
            features: &["clap"],
            modules: &["companion"],
        },
        Case {
            name: "testkit",
            default_features: false,
            features: &["testkit"],
            modules: &["testkit"],
        },
        Case {
            name: "plugin-host",
            default_features: false,
            features: &["plugin", "host"],
            modules: &["plugin", "host"],
        },
    ];

    let mut failures = Vec::new();
    for case in cases {
        let output = run_case(root, case);
        if !output.status.success() {
            failures.push(format!(
                "{} failed:\n{}",
                case.name,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }

    let ara_configured = sdk_is_configured(root, "ARA_SDK_DIR", ".third-party/ARA_SDK");
    let vst3_configured = sdk_is_configured(root, "ARA_VST3_SDK_DIR", ".third-party/vst3sdk");
    let vst3_cases = [
        Case {
            name: "vst3-3-8",
            default_features: false,
            features: &["vst3"],
            modules: &["companion"],
        },
        Case {
            name: "full-portable-vst3-3-8",
            default_features: false,
            features: &["full-portable"],
            modules: &["plugin", "host", "companion"],
        },
    ];
    if vst3_configured && ara_configured {
        for case in vst3_cases {
            let output = run_case(root, case);
            if !output.status.success() {
                failures.push(format!(
                    "{} failed:\n{}",
                    case.name,
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }
    } else {
        for case in vst3_cases {
            if vst3_configured {
                assert_missing_sdk_error(
                    root,
                    case,
                    "ARA_SDK_DIR must point at the project-local SDK installed by scripts/install-ara-sdk.sh",
                );
            } else {
                assert_missing_vst3_error(root, case);
            }
        }
    }

    if cfg!(target_vendor = "apple") {
        let audio_unit_configured =
            sdk_is_configured(root, "ARA_AUDIO_UNIT_SDK_DIR", ".third-party/AudioUnitSDK");
        let audio_unit_case = Case {
            name: "audio-unit-v2",
            default_features: false,
            features: &["audio-unit-v2"],
            modules: &["companion"],
        };
        if audio_unit_configured && ara_configured {
            let output = run_case(root, audio_unit_case);
            if !output.status.success() {
                failures.push(format!(
                    "audio-unit-v2 failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        } else if audio_unit_configured {
            assert_missing_ara_error(root, audio_unit_case);
        } else {
            assert_missing_audio_unit_error(root, audio_unit_case);
        }

        let full_apple_case = Case {
            name: "full-apple",
            default_features: false,
            features: &["full-apple"],
            modules: &["plugin", "host", "companion"],
        };
        if vst3_configured && audio_unit_configured && ara_configured {
            let output = run_case(root, full_apple_case);
            if !output.status.success() {
                failures.push(format!(
                    "full-apple failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        } else if !vst3_configured {
            assert_missing_vst3_error(root, full_apple_case);
        } else if !ara_configured {
            assert_missing_ara_error(root, full_apple_case);
        } else {
            assert_missing_audio_unit_error(root, full_apple_case);
        }
    } else {
        assert_non_apple_audio_unit_error(root, "audio-unit-v2", &["audio-unit-v2"]);
    }

    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn native_build_scripts_resolve_ara_from_the_consuming_project() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    for relative in [
        "ara2-bridge-companion/build.rs",
        "ara2-bridge-testkit/build.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(
            source.contains("cargo:rerun-if-env-changed=ARA_SDK_DIR"),
            "{relative} does not track the consuming project's ARA SDK configuration"
        );
        assert!(
            !source.contains("reference/ARA_SDK")
                && !source.contains("root.join(\".third-party/ARA_SDK"),
            "{relative} assumes an SDK inside the ara2-bridge source tree"
        );
    }
}

#[test]
fn vst3_fallback_requires_the_locked_header_layout() {
    let _sdk_env_guard = SDK_ENV_LOCK.lock().unwrap();
    let root =
        std::env::temp_dir().join(format!("ara2-bridge-vst3-fallback-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    let original = std::env::var_os("ARA_VST3_SDK_DIR");
    std::env::set_var("ARA_VST3_SDK_DIR", &root);
    assert!(!sdk_is_configured(
        &root,
        "ARA_VST3_SDK_DIR",
        "missing-vst3-sdk"
    ));
    let mut command = Command::new("unused");
    configure_sdk(
        &mut command,
        "ARA_VST3_SDK_DIR",
        root.join("missing-vst3-sdk"),
    );
    assert_eq!(
        command
            .get_envs()
            .find(|(key, _)| *key == OsStr::new("ARA_VST3_SDK_DIR"))
            .and_then(|(_, value)| value),
        None
    );
    assert!(!sdk_fallback_is_configured("ARA_VST3_SDK_DIR", &root));
    for relative in [
        "pluginterfaces/base/funknown.h",
        "pluginterfaces/base/falignpush.h",
        "pluginterfaces/base/falignpop.h",
    ] {
        let marker = root.join(relative);
        fs::create_dir_all(marker.parent().unwrap()).unwrap();
        fs::write(marker, []).unwrap();
    }
    assert!(sdk_is_configured(
        &root,
        "ARA_VST3_SDK_DIR",
        "missing-vst3-sdk"
    ));
    assert!(sdk_fallback_is_configured("ARA_VST3_SDK_DIR", &root));

    if let Some(value) = original {
        std::env::set_var("ARA_VST3_SDK_DIR", value);
    } else {
        std::env::remove_var("ARA_VST3_SDK_DIR");
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn audio_unit_fallback_requires_the_locked_header_layout() {
    let root = std::env::temp_dir().join(format!(
        "ara2-bridge-audio-unit-fallback-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    assert!(!sdk_fallback_is_configured("ARA_AUDIO_UNIT_SDK_DIR", &root));
    let header = root.join("Source/AudioUnitSDK.h");
    fs::create_dir_all(header.parent().unwrap()).unwrap();
    fs::write(header, []).unwrap();
    assert!(sdk_fallback_is_configured("ARA_AUDIO_UNIT_SDK_DIR", &root));

    fs::remove_dir_all(root).unwrap();
}

fn assert_non_apple_audio_unit_error(root: &Path, name: &str, features: &[&str]) {
    let output = run_case(
        root,
        Case {
            name,
            default_features: false,
            features,
            modules: &["companion"],
        },
    );
    assert!(!output.status.success(), "{name} unexpectedly compiled");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("audio-unit-v2 is supported only on Apple targets"),
        "{name} produced the wrong diagnostic:\n{stderr}"
    );
}

fn assert_missing_vst3_error(root: &Path, case: Case<'_>) {
    assert_missing_sdk_error(
        root,
        case,
        "VST3 SDK v3.8.0_build_66 is required when this companion feature is enabled",
    );
}

fn assert_missing_audio_unit_error(root: &Path, case: Case<'_>) {
    assert_missing_sdk_error(
        root,
        case,
        "AudioUnitSDK-1.0.0 is required when this companion feature is enabled",
    );
}

fn assert_missing_ara_error(root: &Path, case: Case<'_>) {
    assert_missing_sdk_error(
        root,
        case,
        "ARA_SDK_DIR must point at the project-local SDK installed by scripts/install-ara-sdk.sh",
    );
}

fn assert_missing_sdk_error(root: &Path, case: Case<'_>, expected: &str) {
    let output = run_case(root, case);
    assert!(
        !output.status.success(),
        "{} unexpectedly compiled",
        case.name
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expected),
        "{} produced the wrong diagnostic:\n{stderr}",
        case.name
    );
}

fn run_case(root: &Path, case: Case<'_>) -> Output {
    let directory = std::env::temp_dir()
        .join(format!("ara2-bridge-feature-matrix-{}", std::process::id()))
        .join(case.name);
    assert!(
        !directory.starts_with(root),
        "isolated consumers must not inherit repository Cargo configuration"
    );
    let _ = fs::remove_dir_all(&directory);
    let source = directory.join("src");
    fs::create_dir_all(&source).unwrap();
    fs::write(directory.join("Cargo.toml"), manifest(root, case)).unwrap();
    fs::write(source.join("lib.rs"), consumer_source(case.modules)).unwrap();

    let mut command = Command::new(env!("CARGO"));
    command
        .arg("check")
        .arg("--quiet")
        .arg("--offline")
        .current_dir(&directory)
        .env(
            "CARGO_TARGET_DIR",
            root.join("target/feature-matrix-target"),
        );
    configure_sdk(
        &mut command,
        "ARA_SDK_DIR",
        root.join(".third-party/ARA_SDK"),
    );
    configure_sdk(&mut command, "ARA_CLAP_DIR", root.join(".third-party/clap"));
    configure_sdk(
        &mut command,
        "ARA_VST3_SDK_DIR",
        root.join(".third-party/vst3sdk"),
    );
    configure_sdk(
        &mut command,
        "ARA_AUDIO_UNIT_SDK_DIR",
        root.join(".third-party/AudioUnitSDK"),
    );
    let output = command.output().unwrap();
    fs::remove_dir_all(directory).unwrap();
    output
}

fn configure_sdk(command: &mut Command, variable: &str, fallback: PathBuf) {
    let configured = std::env::var_os(variable)
        .map(PathBuf::from)
        .filter(|path| sdk_fallback_is_configured(variable, path))
        .or_else(|| sdk_fallback_is_configured(variable, &fallback).then_some(fallback));
    if let Some(path) = configured {
        command.env(variable, path);
    } else {
        command.env_remove(variable);
    }
}

fn sdk_is_configured(root: &Path, variable: &str, fallback: &str) -> bool {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .is_some_and(|configured| sdk_fallback_is_configured(variable, &configured))
        || sdk_fallback_is_configured(variable, &root.join(fallback))
}

fn sdk_fallback_is_configured(variable: &str, fallback: &Path) -> bool {
    let required: &[&str] = match variable {
        "ARA_SDK_DIR" => &["ARA_API/ARAInterface.h"],
        "ARA_CLAP_DIR" => &["include/clap/clap.h"],
        "ARA_VST3_SDK_DIR" => &[
            "pluginterfaces/base/funknown.h",
            "pluginterfaces/base/falignpush.h",
            "pluginterfaces/base/falignpop.h",
        ],
        "ARA_AUDIO_UNIT_SDK_DIR" => &["Source/AudioUnitSDK.h"],
        _ => return fallback.is_dir(),
    };
    required
        .iter()
        .all(|relative| fallback.join(relative).is_file())
}

fn manifest(root: &Path, case: Case<'_>) -> String {
    let features = case
        .features
        .iter()
        .map(|feature| format!("\"{feature}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "[package]\nname = \"facade-case-{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[workspace]\n\n[dependencies.ara2-bridge]\npath = {:?}\ndefault-features = {}\nfeatures = [{}]\n",
        case.name,
        root.join("ara2-bridge"),
        case.default_features,
        features
    )
}

fn consumer_source(modules: &[&str]) -> String {
    let mut source = String::from(
        "#![allow(unused_imports)]\nuse ara2_bridge::core as _core;\nuse ara2_bridge::sys as _sys;\n",
    );
    for module in modules {
        source.push_str(&format!("use ara2_bridge::{module} as _{module};\n"));
    }
    source.push_str("pub fn consumer_compiles() {}\n");
    source
}
