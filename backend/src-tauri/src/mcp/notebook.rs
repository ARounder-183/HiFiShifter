//! Non-destructive fenced Agent sections inside the Project Notebook (`hfs_notebook`).
//!
//! Agents collaborate with humans in `ProjectState.notes_markdown` by reading the full
//! notebook and upserting fenced blocks delimited by:
//! `<!-- hfs-agent:start:<section_id> -->` ... `<!-- hfs-agent:end:<section_id> -->`
//! Human notes outside those markers are preserved verbatim.

use super::draft::Failure;
use serde_json::{json, Value};

const MAX_MARKDOWN_BODY_BYTES: usize = 256 * 1024;

fn validate_section_id(section_id: &str) -> Result<(), Failure> {
    if section_id.is_empty()
        || section_id.len() > 64
        || !section_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(Failure::new(
            "INVALID_ARGUMENT",
            "section_id must match [A-Za-z0-9_-]{1,64}",
        ));
    }
    Ok(())
}

fn start_marker(section_id: &str) -> String {
    format!("<!-- hfs-agent:start:{section_id} -->")
}

fn end_marker(section_id: &str) -> String {
    format!("<!-- hfs-agent:end:{section_id} -->")
}

pub(super) fn list_agent_sections(markdown: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let prefix = "<!-- hfs-agent:start:";
    let mut search_from = 0usize;
    while let Some(rel) = markdown[search_from..].find(prefix) {
        let tag_start = search_from + rel;
        let id_start = tag_start + prefix.len();
        let Some(tag_close_rel) = markdown[id_start..].find("-->") else {
            break;
        };
        let section_id = markdown[id_start..id_start + tag_close_rel].trim();
        let content_start = id_start + tag_close_rel + 3;
        let end_tag = end_marker(section_id);
        if let Some(end_rel) = markdown[content_start..].find(&end_tag) {
            let content = markdown[content_start..content_start + end_rel].trim();
            out.push(json!({
                "section_id": section_id,
                "byte_len": content.len(),
                "content": content,
            }));
            search_from = content_start + end_rel + end_tag.len();
        } else {
            break;
        }
    }
    out
}

pub(super) fn upsert_fenced_section(
    existing: &str,
    section_id: &str,
    title: &str,
    markdown_body: &str,
) -> Result<String, Failure> {
    validate_section_id(section_id)?;
    let clean_title = title.trim();
    if clean_title.is_empty() || clean_title.len() > 200 || clean_title.contains("-->") {
        return Err(Failure::new(
            "INVALID_ARGUMENT",
            "title must be 1..=200 chars and must not contain HTML comment delimiters",
        ));
    }
    if markdown_body.len() > MAX_MARKDOWN_BODY_BYTES || markdown_body.contains("<!-- hfs-agent:") {
        return Err(Failure::new(
            "INVALID_ARGUMENT",
            "markdown_body exceeds 256 KiB or contains reserved hfs-agent markers",
        ));
    }
    let start = start_marker(section_id);
    let end = end_marker(section_id);
    let block = format!(
        "{start}\n### {clean_title}\n\n{}\n{end}",
        markdown_body.trim()
    );

    if let Some(s_idx) = existing.find(&start) {
        let after_start = s_idx + start.len();
        let Some(e_rel) = existing[after_start..].find(&end) else {
            return Err(Failure::new(
                "INVALID_ARGUMENT",
                format!("corrupted notebook fence: missing closing marker for '{section_id}'"),
            ));
        };
        let e_end = after_start + e_rel + end.len();
        let mut updated = String::with_capacity(existing.len() + block.len());
        updated.push_str(&existing[..s_idx]);
        updated.push_str(&block);
        updated.push_str(&existing[e_end..]);
        Ok(updated)
    } else if existing.trim().is_empty() {
        Ok(format!("{block}\n"))
    } else {
        let sep = if existing.ends_with("\n\n") {
            ""
        } else if existing.ends_with('\n') {
            "\n"
        } else {
            "\n\n"
        };
        Ok(format!("{existing}{sep}{block}\n"))
    }
}

pub(super) fn remove_fenced_section(
    existing: &str,
    section_id: &str,
) -> Result<(String, bool), Failure> {
    validate_section_id(section_id)?;
    let start = start_marker(section_id);
    let end = end_marker(section_id);
    let Some(s_idx) = existing.find(&start) else {
        return Ok((existing.to_string(), false));
    };
    let after_start = s_idx + start.len();
    let Some(e_rel) = existing[after_start..].find(&end) else {
        return Err(Failure::new(
            "INVALID_ARGUMENT",
            format!("corrupted notebook fence: missing closing marker for '{section_id}'"),
        ));
    };
    let mut e_end = after_start + e_rel + end.len();
    if existing[e_end..].starts_with('\n') {
        e_end += 1;
    }
    let mut updated = String::with_capacity(existing.len());
    updated.push_str(&existing[..s_idx]);
    updated.push_str(&existing[e_end..]);
    Ok((updated, true))
}
