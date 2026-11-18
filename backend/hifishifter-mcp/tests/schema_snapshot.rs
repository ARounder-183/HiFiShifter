//! Snapshot gate ensuring committed JSON Schemas under `docs/schemas/` never
//! drift from the Rust types in `hifishifter-mcp`.
//!
//! Run with `HFS_UPDATE_SCHEMAS=1 cargo test -p hifishifter-mcp --test schema_snapshot`
//! after intentionally changing MCP tool inputs or voicebank contracts.

use hifishifter_mcp::{
    server::export_contract_schemas,
    voicebank::{PhraseMarks, VoicebankManifest},
};
use serde_json::json;
use std::path::PathBuf;

fn schemas_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("schemas")
}

fn assert_or_write_snapshot(filename: &str, expected: &serde_json::Value) {
    let dir = schemas_dir();
    let path = dir.join(filename);
    let rendered = format!("{}\n", serde_json::to_string_pretty(expected).unwrap());

    if std::env::var("HFS_UPDATE_SCHEMAS").as_deref() == Ok("1") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, &rendered).unwrap();
        return;
    }

    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Missing schema snapshot {}: {e}. Regenerate with HFS_UPDATE_SCHEMAS=1 cargo test -p hifishifter-mcp --test schema_snapshot",
            path.display()
        )
    });
    assert_eq!(
        on_disk.replace("\r\n", "\n"),
        rendered,
        "Schema drift detected in {}. Regenerate with HFS_UPDATE_SCHEMAS=1 cargo test -p hifishifter-mcp --test schema_snapshot",
        path.display()
    );
}

#[test]
fn mcp_tool_and_voicebank_json_schemas_match_snapshots() {
    let mcp_schemas = export_contract_schemas();
    assert_or_write_snapshot("mcp-tools.schema.json", &mcp_schemas);

    let voicebank_schemas = json!({
        "voicebank_manifest": schemars::schema_for!(VoicebankManifest),
        "phrase_marks": schemars::schema_for!(PhraseMarks),
    });
    assert_or_write_snapshot("voicebank-marks.schema.json", &voicebank_schemas);
}
