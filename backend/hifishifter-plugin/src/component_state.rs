//! 组件 state 的容器格式：编辑状态 + Transfer 清单。
//!
//! # 为什么需要一层容器
//!
//! VST3 的组件 state 是一个**不透明字节块**，宿主只负责原样存取。此前它的内容就是
//! `EditState::encode()` 的输出。加入 Transfer 清单时若直接把两段拼起来，两个方向
//! 都会坏：旧版本读新块会解析失败（`EditState::restore` 拿到多余的尾巴），新版本读
//! 旧块也会。
//!
//! 用一个带魔数的容器把"这是新版"与"这是旧版"分开，且**旧块仍然可读** ——
//! 没有魔数就按"整块是 `EditState`"处理。这条兼容不是可选项：用户已经存过的工程
//! 里就是旧格式，读不出来等于把他们的编辑弄丢了。
//!
//! # 格式
//!
//! ```text
//! magic     8 字节  "HFSSTATE"
//! version   u32     = 2（v1 仍可读）
//! edits_len u32
//! edits     edits_len 字节（`EditState::encode()` 的输出）
//! xfer_len  u32     （0 = 没有 Transfer 清单）
//! xfer      xfer_len 字节（清单 JSON）
//! tedits_len u32    （v2 起；0 = 没有 Transfer 用户编辑）
//! tedits    tedits_len 字节（`TransferEdits` 的 JSON）
//! ```
//!
//! 【为什么 v2 只**追加**一段】v1 的每个字段都是长度前缀，读到 `xfer` 结束就是一个
//! 完整载荷。追加段让旧工程（v1）照样读得出来，新版本也不会把旧载荷误判成截断。
//!
//! 【为什么清单是 JSON 而不是二进制】它是几十到几百字节的元数据，且**需要能被诊断
//! 工具读**（"这个 take 的 sidecar 在哪"是排查"重启后录音不见了"的第一个问题）。
//! 二进制格式在这里只换来"多写一个解析器"。

/// 容器魔数。
const MAGIC: &[u8; 8] = b"HFSSTATE";
/// 容器版本。
///
/// v2 追加 Transfer 用户编辑段（曲线 / 音阶 / Tempo Map / 轨道设置）。
const VERSION: u32 = 2;

/// 一次组件 state 的载荷。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ComponentState {
    /// `EditState::encode()` 的输出。
    pub edits: Vec<u8>,
    /// Transfer 清单 JSON；`None` = 这个实例没有 Transfer 数据。
    pub transfer: Option<Vec<u8>>,
    /// Transfer 用户编辑 JSON（见 `editor::transfer_edits`）；`None` = 没有编辑。
    pub transfer_edits: Option<Vec<u8>>,
}

impl ComponentState {
    /// 编码成宿主存取的字节块。
    pub(crate) fn encode(&self) -> Vec<u8> {
        let transfer = self.transfer.as_deref().unwrap_or(&[]);
        let transfer_edits = self.transfer_edits.as_deref().unwrap_or(&[]);
        let mut out = Vec::with_capacity(
            MAGIC.len() + 16 + self.edits.len() + transfer.len() + transfer_edits.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.edits.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.edits);
        out.extend_from_slice(&(transfer.len() as u32).to_le_bytes());
        out.extend_from_slice(transfer);
        out.extend_from_slice(&(transfer_edits.len() as u32).to_le_bytes());
        out.extend_from_slice(transfer_edits);
        out
    }

    /// 解析字节块。**没有魔数时整块按 `EditState` 处理**（版本化之前的历史格式）。
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            // 历史格式：整块就是 `EditState`。
            return Ok(Self {
                edits: bytes.to_vec(),
                transfer: None,
                transfer_edits: None,
            });
        }
        let mut cursor = MAGIC.len();
        let version = read_u32(bytes, &mut cursor)?;
        // 【为什么 v1 与 v2 都接受、更高版本拒绝】v1 是已经存在于用户工程里的格式，
        // 读不出来等于把他们的编辑弄丢；v3 及以后我们不知道它追加了什么，硬按 v2
        // 解析会把未知段当成截断错误 —— 明确拒绝并说明"来自更新的版本"才是可执行的。
        if !(1..=VERSION).contains(&version) {
            return Err(format!(
                "unsupported component state version {version} (expected <= {VERSION}); \
                 this project was written by a newer build"
            ));
        }
        let edits_len = read_u32(bytes, &mut cursor)? as usize;
        let edits = take(bytes, &mut cursor, edits_len)?.to_vec();
        let transfer_len = read_u32(bytes, &mut cursor)? as usize;
        // 【为什么允许 0 与"没有清单"等价】编码时 `None` 写成 0 长度。区分"空清单"
        // 与"没有清单"没有语义差别 —— 两者都表示这个实例没有 Transfer 数据。
        let transfer = if transfer_len == 0 {
            None
        } else {
            Some(take(bytes, &mut cursor, transfer_len)?.to_vec())
        };
        // v1 到此结束；v2 后面还有一段。**按剩余长度判断**而不是只看版本号：
        // 这样即便某个宿主把载荷补零（或未来出现 v1 的变体），也不会把 0 字节当成
        // 一段有效的空编辑。
        let transfer_edits = if version >= 2 && cursor < bytes.len() {
            let len = read_u32(bytes, &mut cursor)? as usize;
            if len == 0 {
                None
            } else {
                Some(take(bytes, &mut cursor, len)?.to_vec())
            }
        } else {
            None
        };
        Ok(Self {
            edits,
            transfer,
            transfer_edits,
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
            transfer: Some(br#"{"version":1,"takes":[]}"#.to_vec()),
            transfer_edits: Some(br#"{"params_by_root_track":{}}"#.to_vec()),
        };
        let decoded = ComponentState::decode(&state.encode()).unwrap();
        assert_eq!(decoded, state);
    }

    /// 没有 Transfer 数据的实例往返后仍然是 `None`（而不是空 `Some`）。
    #[test]
    fn an_instance_without_transfer_data_round_trips_to_none() {
        let state = ComponentState {
            edits: vec![9, 9],
            transfer: None,
            transfer_edits: None,
        };
        assert_eq!(ComponentState::decode(&state.encode()).unwrap(), state);
    }

    /// **v1 载荷必须仍然可读**：用户已经存过的工程里就是带清单的 v1。
    ///
    /// 【为什么这是本文件第二重要的一条】v2 只是追加一段；把 v1 判成"截断"会让
    /// 所有既有工程在升级后**完全读不出编辑**。
    #[test]
    fn a_v1_container_is_read_without_a_transfer_edits_section() {
        let edits = b"edit-bytes";
        let transfer = br#"{"version":1,"takes":[]}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(edits.len() as u32).to_le_bytes());
        bytes.extend_from_slice(edits);
        bytes.extend_from_slice(&(transfer.len() as u32).to_le_bytes());
        bytes.extend_from_slice(transfer);

        let decoded = ComponentState::decode(&bytes).unwrap();
        assert_eq!(decoded.edits, edits.to_vec());
        assert_eq!(decoded.transfer, Some(transfer.to_vec()));
        assert_eq!(decoded.transfer_edits, None);
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
        assert_eq!(decoded.transfer, None);
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
            transfer: None,
            transfer_edits: None,
        }
        .encode();
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        let error = ComponentState::decode(&bytes).unwrap_err();
        assert!(
            error.contains("newer build"),
            "错误要指向可执行的结论：{error}"
        );
    }

    /// 截断的容器被明确拒绝，不返回半个 `EditState`。
    #[test]
    fn a_truncated_container_is_rejected() {
        let bytes = ComponentState {
            edits: vec![1, 2, 3, 4, 5, 6],
            transfer: Some(vec![7, 8]),
            transfer_edits: Some(vec![9, 10, 11]),
        }
        .encode();
        assert!(ComponentState::decode(&bytes[..bytes.len() - 3]).is_err());
    }
}
