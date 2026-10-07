//! 原生片段复制基础：有界RPPXML解析，只重建宿主对象身份，不修改源或插件不透明状态。
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void};

pub(super) const MAX_ITEM_CHUNK_BYTES: usize = 8 * 1024 * 1024;
pub(super) type GetChunk = unsafe extern "C" fn(*mut c_void, *mut c_char, i32, bool) -> bool;
pub(super) type SetChunk = unsafe extern "C" fn(*mut c_void, *const c_char, bool) -> bool;
pub(super) type GenGuid = unsafe extern "C" fn(*mut NativeGuid);
pub(super) type GuidString = unsafe extern "C" fn(*const NativeGuid, *mut c_char);

/// 对齐锁定官方SDK的GUID布局；不能用align=1的字节数组充当原生GUID对象。
#[repr(C)]
#[derive(Default)]
pub(super) struct NativeGuid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

pub(super) struct ItemStateApi {
    pub get: GetChunk,
    pub set: SetChunk,
    pub generate: GenGuid,
    pub stringify: GuidString,
}

pub(crate) struct RewrittenItem {
    pub text: String,
    pub item_guid: String,
    pub take_guids: Vec<String>,
    pub generated_guids: Vec<String>,
}

/// GUID严格按宿主文本格式验证，生成回执也必须通过，避免带换行的身份注入。
pub(super) fn valid_guid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 38 && bytes[0] == b'{' && bytes[37] == b'}'
        && (1..37).all(|i| if [9, 14, 19, 24].contains(&i) { bytes[i] == b'-' } else { bytes[i].is_ascii_hexdigit() })
}

/// 粘贴前生成全新item/take/FX实例GUID；源状态内GUID和FX不透明payload保持原样。
pub(super) fn rewrite_item(
    chunk: &str,
    position: f64,
    mut fresh_guid: impl FnMut() -> Result<String, String>,
) -> Result<RewrittenItem, String> {
    if chunk.is_empty() || chunk.len() > MAX_ITEM_CHUNK_BYTES || chunk.contains('\0') {
        return Err("invalid or oversized REAPER item state".into());
    }
    if !position.is_finite() || !(0.0..=1_000_000.0).contains(&position) {
        return Err("invalid paste position".into());
    }
    let mut stack = Vec::<&str>::new();
    let mut item_seen = false;
    let mut item_finished = false;
    let mut positions = 0;
    let mut item_guid = None;
    let mut take_guids = Vec::new();
    let mut generated = BTreeSet::new();
    // 不允许生成器把任何旧对象GUID再次用作新身份；只读检查不重写opaque payload。
    let old_guids: BTreeSet<_> = chunk.lines().filter_map(|line| {
        let mut words = line.split_whitespace();
        match (words.next(), words.next(), words.next()) {
            (Some("IGUID" | "GUID" | "FXID"), Some(guid), None) if valid_guid(guid) => Some(guid.to_ascii_uppercase()),
            _ => None,
        }
    }).collect();
    let mut output = String::with_capacity(chunk.len() + 64);
    for line in chunk.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            output.push_str(line);
            output.push('\n');
            continue;
        }
        if item_finished { return Err("unexpected data after item state".into()); }
        if let Some(open) = trimmed.strip_prefix('<') {
            let kind = open.split_whitespace().next().ok_or("invalid item state block")?;
            if stack.is_empty() {
                if item_seen || kind != "ITEM" { return Err("clipboard must contain exactly one ITEM block".into()); }
                item_seen = true;
            } else if kind == "ITEM" { return Err("nested ITEM state is unsupported".into()); }
            if stack.len() >= 64 { return Err("item state nesting budget exceeded".into()); }
            stack.push(kind);
        } else if trimmed == ">" {
            if stack.pop().is_none() { return Err("unbalanced item state closing block".into()); }
            item_finished = stack.is_empty();
        } else {
            if stack.is_empty() { return Err("data outside item state".into()); }
            let field = trimmed.split_whitespace().next().unwrap();
            let root_field = stack.len() == 1;
            let take_field = root_field || stack.last() == Some(&"TAKE");
            let fx_field = stack.iter().any(|kind| matches!(*kind, "TAKEFX" | "FXCHAIN"));
            let identity = field == "IGUID" && root_field
                || field == "GUID" && take_field
                || field == "FXID" && fx_field;
            let indent = &line[..line.len() - line.trim_start().len()];
            if identity {
                let old = trimmed.strip_prefix(field).unwrap().trim();
                if !valid_guid(old) { return Err("invalid item/take/FX identity in clipboard".into()); }
                if generated.len()>=16384 {return Err("item instance identity budget exceeded".into());}
                let guid = fresh_guid()?.to_ascii_uppercase();
                let normalized = guid.to_ascii_uppercase();
                if !valid_guid(&guid) || old_guids.contains(&normalized) || !generated.insert(normalized) {
                    return Err("generated clipboard GUID is invalid or not unique".into());
                }
                if field == "IGUID" {
                    if item_guid.replace(guid.clone()).is_some() { return Err("duplicate item identity".into()); }
                } else if field == "GUID" {
                    if take_guids.len() >= 512 { return Err("take count budget exceeded".into()); }
                    take_guids.push(guid.clone());
                }
                output.push_str(&format!("{indent}{field} {guid}\n"));
                continue;
            }
            if root_field && field == "POSITION" {
                positions += 1;
                output.push_str(&format!("{indent}POSITION {position:.17}\n"));
                continue;
            }
            if root_field && field == "SEL" {
                output.push_str(&format!("{indent}SEL 0\n"));
                continue;
            }
        }
        output.push_str(line);
        output.push('\n');
        if output.len() > MAX_ITEM_CHUNK_BYTES { return Err("rewritten item state budget exceeded".into()); }
    }
    if !item_seen || !item_finished || !stack.is_empty() || positions != 1 || take_guids.is_empty() {
        return Err("incomplete item state, position, or take identity".into());
    }
    if output.len() > MAX_ITEM_CHUNK_BYTES { return Err("rewritten item state budget exceeded".into()); }
    Ok(RewrittenItem { text: output, item_guid: item_guid.ok_or("missing item identity")?, take_guids, generated_guids:generated.into_iter().collect() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(id: u32) -> String { format!("{{{id:08X}-1111-2222-3333-444444444444}}") }

    /// 身份变化不侵入SOURCE/FX载荷；倍率、源偏移、渐变与多Take原样保留。
    #[test]
    fn item_copy_rewrites_instances_and_position_but_preserves_source_and_fx_data() {
        let chunk = format!("<ITEM\nPOSITION 2.5\nLENGTH 1.25\nSEL 1\nIGUID {}\nSOFFS 0.2\nPLAYRATE 1.5 1 0\nGUID {}\n<SOURCE WAVE\nFILE \"E:/音mad/ka.wav\"\nGUID {}\n>\n<TAKEFX\n<VST opaque\nAAABBB==\n>\nFXID {}\n>\nTAKE\nGUID {}\n<SOURCE WAVE\nFILE \"E:/音mad/n.wav\"\n>\n>\n", guid(1), guid(2), guid(3), guid(4), guid(5));
        let mut next = 100;
        let rewritten = rewrite_item(&chunk, 9.25, || { next += 1; Ok(guid(next)) }).unwrap();
        assert_eq!(rewritten.item_guid, guid(101));
        assert_eq!(rewritten.take_guids, [guid(102), guid(104)]);
        for original in ["LENGTH 1.25", "SOFFS 0.2", "PLAYRATE 1.5 1 0", "AAABBB==", "E:/音mad/ka.wav"] {
            assert!(rewritten.text.contains(original));
        }
        assert!(rewritten.text.contains(&format!("<SOURCE WAVE\nFILE \"E:/音mad/ka.wav\"\nGUID {}", guid(3))));
        assert!(rewritten.text.contains(&format!("FXID {}", guid(103))));
        assert!(rewritten.text.contains("POSITION 9.25000000000000000\n"));
        assert!(rewritten.text.contains("SEL 0\n"));
    }

    /// 未平衡结构、多个item、重复/旧GUID与非有限位置都必须在原生创建前拒绝。
    #[test]
    fn item_copy_rejects_malformed_structure_and_identity_collisions() {
        let chunk = format!("<ITEM\nPOSITION 0\nIGUID {}\nGUID {}\n>\n", guid(1), guid(2));
        let mut next = 100;
        for bad in [chunk.replace("IGUID", "UNKNOWN"), chunk.replace("POSITION 0", "POSITION 0\nPOSITION 1"), chunk.trim_end_matches(">\n").into(), format!("{chunk}{chunk}"), format!("{chunk}not-an-item") ] {
            assert!(rewrite_item(&bad, 1.0, || { next += 1; Ok(guid(next)) }).is_err());
        }
        assert!(rewrite_item(&chunk, f64::NAN, || Ok(guid(100))).is_err());
        assert!(rewrite_item(&chunk, 1.0, || Ok(guid(1))).is_err());
        assert!(rewrite_item(&chunk, 1.0, || Ok(guid(100))).is_err());
        assert!(rewrite_item(&chunk, 1.0, || Ok("not a guid".into())).is_err());
    }
}
