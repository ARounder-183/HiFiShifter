//! 组件 state 的容器格式：编辑状态 + Capture 清单。
//!
//! # 为什么需要一层容器
//!
//! VST3 的组件 state 是一个**不透明字节块**，宿主只负责原样存取。此前它的内容就是
//! `EditState::encode()` 的输出。加入 Capture 清单时若直接把两段拼起来，两个方向
//! 都会坏：旧版本读新块会解析失败（`EditState::restore` 拿到多余的尾巴），新版本读
//! 旧块也会。
//!
//! 用一个带魔数的容器把"这是本插件的载荷"与"裸 `EditState`"分开。没有魔数就按
//! "整块是 `EditState`"处理 —— 那条分支**保留**，但它的理由不是"兼容旧版本"：
//! `decode` 同时是 `EditState` 自己的解码入口，裸字节块在别处仍然可能出现。
//!
//! # 格式
//!
//! ```text
//! magic     8 字节  "HFSSTATE"
//! version   u32     = 1
//! edits_len u32
//! edits     edits_len 字节（`EditState::encode()` 的输出）
//! capture_len u32   （0 = 没有 Capture 清单）
//! capture   capture_len 字节（清单 JSON）
//! cedits_len u32    （0 = 没有 Capture 用户编辑）
//! cedits    cedits_len 字节（`CaptureEdits` 的 JSON）
//! ```
//!
//! 【为什么版本号回到 1、且不再有"v1 兼容"分支】此前 v2 为了能读 v1（一个少了
//! `capture_edits` 段的形状）保留了"按剩余长度判断"的分支。而 VST3 侧从未向任何
//! 用户分发过 —— 宿主工程里不存在这种旧块。为一份不存在的数据保留一条会把"截断"
//! 误读成"没有编辑"的分支，收益为负，所以格式在这里重置为单版本。
//!
//! 【为什么清单是 JSON 而不是二进制】它是几十到几百字节的元数据，且**需要能被诊断
//! 工具读**（"这个 take 的 sidecar 在哪"是排查"重启后录音不见了"的第一个问题）。
//! 二进制格式在这里只换来"多写一个解析器"。

/// 容器魔数。
const MAGIC: &[u8; 8] = b"HFSSTATE";
/// 容器版本（重置后的第一版：编辑 + 清单 + Capture 用户编辑，三段俱全）。
const VERSION: u32 = 1;

/// 一次组件 state 的载荷。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ComponentState {
    /// `EditState::encode()` 的输出。
    pub edits: Vec<u8>,
    /// Capture 清单 JSON；`None` = 这个实例没有 Capture 数据。
    pub capture: Option<Vec<u8>>,
    /// Capture 用户编辑 JSON（见 `editor::capture_edits`）；`None` = 没有编辑。
    pub capture_edits: Option<Vec<u8>>,
}

impl ComponentState {
    /// 编码成宿主存取的字节块。
    pub(crate) fn encode(&self) -> Vec<u8> {
        let capture = self.capture.as_deref().unwrap_or(&[]);
        let capture_edits = self.capture_edits.as_deref().unwrap_or(&[]);
        let mut out = Vec::with_capacity(
            MAGIC.len() + 16 + self.edits.len() + capture.len() + capture_edits.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.edits.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.edits);
        out.extend_from_slice(&(capture.len() as u32).to_le_bytes());
        out.extend_from_slice(capture);
        out.extend_from_slice(&(capture_edits.len() as u32).to_le_bytes());
        out.extend_from_slice(capture_edits);
        out
    }

    /// 解析字节块。**没有魔数时整块按 `EditState` 处理**（版本化之前的历史格式）。
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            // 历史格式：整块就是 `EditState`。
            return Ok(Self {
                edits: bytes.to_vec(),
                capture: None,
                capture_edits: None,
            });
        }
        let mut cursor = MAGIC.len();
        let version = read_u32(bytes, &mut cursor)?;
        // 【为什么只认一个版本】格式已重置（见文件头）。不匹配只可能是两种情况：
        // 更旧的构建写下的（本机残留），或更新的构建写下的（用户降级了应用）。
        // 两者都无从按当前形状解析，明确拒绝并说清方向才是可执行的。
        if version != VERSION {
            return Err(format!(
                "unsupported component state version {version} (this build writes {VERSION}); \
                 the state was written by a different build of this plug-in"
            ));
        }
        let edits_len = read_u32(bytes, &mut cursor)? as usize;
        let edits = take(bytes, &mut cursor, edits_len)?.to_vec();
        let capture_len = read_u32(bytes, &mut cursor)? as usize;
        // 【为什么允许 0 与"没有清单"等价】编码时 `None` 写成 0 长度。区分"空清单"
        // 与"没有清单"没有语义差别 —— 两者都表示这个实例没有 Capture 数据。
        let capture = if capture_len == 0 {
            None
        } else {
            Some(take(bytes, &mut cursor, capture_len)?.to_vec())
        };
        // 第三段：Capture 用户编辑。长度前缀，0 = 没有。
        //
        // 【为什么不再"按剩余长度判断有没有这一段"】那个判据存在的前提是"v1 载荷到
        // 这里就结束了"，而 v1 已随格式重置一起消失。留着它只会把**截断**误读成
        // "没有编辑"—— 那是一次静默的数据丢失，而明确报错还能让用户换回旧构建。
        let cedits_len = read_u32(bytes, &mut cursor)? as usize;
        let capture_edits = if cedits_len == 0 {
            None
        } else {
            Some(take(bytes, &mut cursor, cedits_len)?.to_vec())
        };
        Ok(Self {
            edits,
            capture,
            capture_edits,
        })
    }
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let slice = take(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8], String> {
    let end = cursor
        .checked_add(count)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| "component state ends mid-field".to_owned())?;
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_trip_preserves_both_halves() {
        let state = ComponentState {
            edits: vec![1, 2, 3, 4],
            capture: Some(br#"{"version":1,"takes":[]}"#.to_vec()),
            capture_edits: Some(br#"{"params_by_root_track":{}}"#.to_vec()),
        };
        let decoded = ComponentState::decode(&state.encode()).unwrap();
        assert_eq!(decoded, state);
    }

    /// 没有 Capture 数据的实例往返后仍然是 `None`（而不是空 `Some`）。
    #[test]
    fn an_instance_without_capture_data_round_trips_to_none() {
        let state = ComponentState {
            edits: vec![9, 9],
            capture: None,
            capture_edits: None,
        };
        assert_eq!(ComponentState::decode(&state.encode()).unwrap(), state);
    }

    /// **重置前构建写下的载荷必须被明确拒绝，而不是被"读成没有编辑"。**
    ///
    /// 【为什么这条取代了原来的 v1 兼容测试】格式已重置（见文件头）：v1（少了
    /// `capture_edits` 段）不再存在。留一条"按剩余长度判断"的分支，只会把**截断**
    /// 误读成"这个实例没有 Capture 编辑"—— 那是静默的数据丢失。
    #[test]
    fn a_pre_reset_container_is_rejected_instead_of_losing_the_edits() {
        let edits = b"edit-bytes";
        let capture = br#"{"version":1,"takes":[]}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        // 旧版本号（重置前是 2）。
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(edits.len() as u32).to_le_bytes());
        bytes.extend_from_slice(edits);
        bytes.extend_from_slice(&(capture.len() as u32).to_le_bytes());
        bytes.extend_from_slice(capture);

        let error = ComponentState::decode(&bytes).unwrap_err();
        assert!(
            error.contains("different build of this plug-in"),
            "错误要指向可执行的结论：{error}"
        );
    }

    /// **历史格式必须仍然可读**：用户已经存过的工程里就是裸的 `EditState` 字节。
    ///
    /// 【为什么这是本文件最重要的一条】读不出来的后果是把用户已有的编辑弄丢 ——
    /// 而那个字节块在磁盘上、在宿主工程里，改不回来。
    #[test]
    fn a_legacy_blob_is_read_as_a_bare_edit_state() {
        let legacy = b"some older edit state bytes";
        let decoded = ComponentState::decode(legacy).unwrap();
        assert_eq!(decoded.edits, legacy);
        assert_eq!(decoded.capture, None);
    }

    /// 更短的块（空 state）也不能被误判成容器。
    #[test]
    fn a_short_blob_is_not_mistaken_for_a_container() {
        assert_eq!(ComponentState::decode(b"").unwrap().edits, Vec::<u8>::new());
        assert_eq!(
            ComponentState::decode(b"HFS").unwrap().edits,
            b"HFS".to_vec()
        );
    }

    /// 未来版本必须被明确拒绝，而不是按当前版本硬解析。
    #[test]
    fn a_newer_version_is_rejected_with_an_explanation() {
        let mut bytes = ComponentState {
            edits: vec![1],
            capture: None,
            capture_edits: None,
        }
        .encode();
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        let error = ComponentState::decode(&bytes).unwrap_err();
        assert!(
            error.contains("different build of this plug-in"),
            "错误要指向可执行的结论：{error}"
        );
    }

    /// 截断的容器被明确拒绝，不返回半个 `EditState`。
    #[test]
    fn a_truncated_container_is_rejected() {
        let bytes = ComponentState {
            edits: vec![1, 2, 3, 4, 5, 6],
            capture: Some(vec![7, 8]),
            capture_edits: Some(vec![9, 10, 11]),
        }
        .encode();
        assert!(ComponentState::decode(&bytes[..bytes.len() - 3]).is_err());
    }
}
