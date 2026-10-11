use std::process::Command;

#[test]
fn caller_valid_malformed_storage_is_rejected_without_a_sanitizer() {
    let output = Command::new(env!("CARGO_BIN_EXE_invalid_pointer_case"))
        .arg("malformed")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn dangerous_cases_refuse_to_run_without_explicit_subprocess_opt_in() {
    for case in ["null-adjacent", "unreadable", "guard-page"] {
        let output = Command::new(env!("CARGO_BIN_EXE_invalid_pointer_case"))
            .arg(case)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(64));
        assert!(String::from_utf8_lossy(&output.stderr).contains("sanitizer harness opt-in"));
    }
}
