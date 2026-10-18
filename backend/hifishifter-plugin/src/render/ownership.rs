//! renderer 分配键到文档/区域槽位的所有权表，仅在非实时模型线程读写。

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub(crate) type RegionKey = u64;
pub(crate) type DocumentId = u64;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum OwnershipError {
    InvalidIdentity,
    DuplicateRegion,
    MissingRegion,
    CrossDocument,
    EmptyAssignment,
}

#[derive(Default)]
pub(crate) struct RegionOwners {
    regions: HashMap<RegionKey, (DocumentId, usize)>,
    /// 本次进程里"文档已销毁"的次数（诊断：宿主是否按 leaf-first 销毁）。
    reclaimed: u64,
}

impl RegionOwners {
    /// 登记 region 的真实 model-ref 键，不用会随删除变化的 clip 下标代替身份。
    ///
    /// 【跨文档重复是认领，不是错误】key 是宿主持有的指针值，宿主进程内地址会复用。
    /// 上一次文档没有走完 `destroy_document`（宿主不按 leaf-first 销毁时
    /// `PluginRuntime::destroy` 会整份放弃）就会留下无主 key；这一次文档在同一批
    /// 操作序列下拿到同一个地址。把它判成 `DuplicateRegion` 会让
    /// `create_playback_region` 一路冒到宿主脸上 —— 宿主从此不再交付任何 region，
    /// 界面永远停在"等待宿主音频"，而删掉插件再添加就好（新指针值）。
    /// 同**一**份文档内的重复才是真的冲突，仍然拒绝。
    pub fn register(
        &mut self,
        key: RegionKey,
        document: DocumentId,
        slot: usize,
    ) -> Result<(), OwnershipError> {
        if key == 0 || document == 0 {
            return Err(OwnershipError::InvalidIdentity);
        }
        match self.regions.get(&key).copied() {
            Some((owner, _)) if owner == document => return Err(OwnershipError::DuplicateRegion),
            Some((owner, _)) => {
                // 上一个文档的遗留：撤销旧归属，按新文档登记。
                self.reclaimed = self.reclaimed.saturating_add(1);
                log::info!(
                    "[ara] region key {key} reclaimed by document {document} from dead document \
                     {owner}"
                );
            }
            None => {}
        }
        self.regions.insert(key, (document, slot));
        // 【为什么不设硬上限】这张表是**所有权权威**：丢掉一个活 region 的条目会让
        // `resolve` 判它"不存在"，进而把一次正常渲染变成错误。硬性淘汰比增长更糟。
        // 真正保证有界的是两条移除路径（region 销毁、文档销毁）；这里只在明显异常时
        // 留痕，让"宿主既不销毁 region 也不关闭文档"这种泄漏在日志里可见。
        const REGION_OWNERS_SOFT_LIMIT: usize = 65_536;
        if self.regions.len() == REGION_OWNERS_SOFT_LIMIT {
            log::warn!(
                "[ara] {REGION_OWNERS_SOFT_LIMIT} regions are registered across all documents;                  if the host is not calling destroyPlaybackRegion this table only grows"
            );
        }
        Ok(())
    }

    /// 区域销毁时撤销身份。
    pub fn remove(&mut self, key: RegionKey) {
        self.regions.remove(&key);
    }

    /// 文档销毁时撤销全部子区域，禁止悬空映射指向后来打开的文档。
    pub fn remove_document(&mut self, document: DocumentId) {
        self.regions.retain(|_, (owner, _)| *owner != document);
    }

    /// 解析 renderer 分配；跨文档、未知、已销毁或空分配都不能被默认为完整时间线。
    pub fn resolve(&self, keys: &[RegionKey]) -> Result<(DocumentId, Vec<usize>), OwnershipError> {
        let first = keys.first().ok_or(OwnershipError::EmptyAssignment)?;
        let (document, _) = self
            .regions
            .get(first)
            .ok_or(OwnershipError::MissingRegion)?;
        let mut slots = Vec::with_capacity(keys.len());
        for key in keys {
            let (owner, slot) = self.regions.get(key).ok_or(OwnershipError::MissingRegion)?;
            if owner != document {
                return Err(OwnershipError::CrossDocument);
            }
            slots.push(*slot);
        }
        Ok((*document, slots))
    }

    /// 单个 key 的归属文档（未登记或已销毁时为 `None`）。
    ///
    /// 【为什么需要逐 key 查询】`resolve` 是**整批**语义：一批里只要有一条 key 未登记
    /// （跨文档或已销毁），整批就返回 `Err`。而宿主的 `addPlaybackRegion` 会把当前
    /// 分配**整体**重发一遍 —— 分割后它可能把已销毁父段的 key 也一起传进来，于是
    /// 插件把**整批**（包括新 region）清成空 ⇒ 静默"分割后毫无反应"。逐 key 查询让
    /// 调用方能只剔除坏的那条。
    pub fn owner_of(&self, key: RegionKey) -> Option<DocumentId> {
        self.regions.get(&key).map(|(document, _)| *document)
    }

    /// 只保留"归属文档仍然活着"的登记。
    ///
    /// 【为什么需要它】`register` 的跨文档认领只在**下一次 create** 才生效；如果宿主
    /// 重开后创建的是 audio source / region sequence 而 region 那一步始终没走到，
    /// 遗留 key 就一直占位。新文档注册时主动清扫一次，让"重开工程"不依赖
    /// 后续创建顺序就能把上一次的残留清干净。
    ///
    /// 【为什么必须逐个问存活】同一进程里可以同时有多份文档（多个工程窗口）。
    /// 全表清空会打断另一份正在用的工程。
    pub fn retain_alive_documents(&mut self, is_alive: &impl Fn(DocumentId) -> bool) -> usize {
        let before = self.regions.len();
        self.regions
            .retain(|_, (document, _)| is_alive(*document));
        let removed = before.saturating_sub(self.regions.len());
        if removed > 0 {
            log::info!("[ara] swept {removed} region entrie(s) owned by dead document(s)");
        }
        removed
    }

    /// 跨文档认领发生的次数（诊断）。
    pub fn reclaimed(&self) -> u64 {
        self.reclaimed
    }
}

/// 进程内身份索引；音频回调不得锁此表，必须消费预先发布的 renderer 快照。
pub(crate) fn region_owners() -> &'static Mutex<RegionOwners> {
    static OWNERS: OnceLock<Mutex<RegionOwners>> = OnceLock::new();
    OWNERS.get_or_init(|| Mutex::new(RegionOwners::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 两个文档都可能从 slot 0 开始，但 renderer 不得把它们混成同一条轨道。
    #[test]
    fn assignments_resolve_only_their_own_document_and_slots() {
        let mut owners = RegionOwners::default();
        owners.register(101, 1, 0).unwrap();
        owners.register(102, 1, 3).unwrap();
        owners.register(202, 2, 0).unwrap();
        assert_eq!(owners.resolve(&[102, 101]).unwrap(), (1, vec![3, 0]));
        assert_eq!(owners.resolve(&[202]).unwrap(), (2, vec![0]));
        assert_eq!(
            owners.resolve(&[101, 202]),
            Err(OwnershipError::CrossDocument)
        );
        assert_eq!(owners.resolve(&[999]), Err(OwnershipError::MissingRegion));
        assert_eq!(owners.resolve(&[]), Err(OwnershipError::EmptyAssignment));
    }

    /// 删除与文档关闭都使旧键失效，同时不影响其他文档。
    #[test]
    fn destroying_regions_and_documents_revokes_only_their_keys() {
        let mut owners = RegionOwners::default();
        owners.register(101, 1, 0).unwrap();
        owners.register(102, 1, 1).unwrap();
        owners.register(202, 2, 0).unwrap();
        owners.remove(101);
        assert_eq!(owners.resolve(&[101]), Err(OwnershipError::MissingRegion));
        owners.remove_document(1);
        assert_eq!(owners.resolve(&[102]), Err(OwnershipError::MissingRegion));
        assert_eq!(owners.resolve(&[202]).unwrap(), (2, vec![0]));
    }

    /// 不允许空身份；**同一份文档内**的重复登记仍然被拒，并且不覆盖归属。
    ///
    /// 跨文档的同一个 key 不是重复而是认领（见 `register` 的说明），由
    /// `a_dead_documents_key_is_reclaimed_rather_than_rejected` 钉住。
    #[test]
    fn duplicate_or_zero_identity_does_not_replace_the_owner() {
        let mut owners = RegionOwners::default();
        assert_eq!(
            owners.register(0, 1, 0),
            Err(OwnershipError::InvalidIdentity)
        );
        assert_eq!(
            owners.register(101, 0, 0),
            Err(OwnershipError::InvalidIdentity)
        );
        owners.register(101, 1, 0).unwrap();
        assert_eq!(
            owners.register(101, 1, 9),
            Err(OwnershipError::DuplicateRegion)
        );
        assert_eq!(owners.resolve(&[101]).unwrap(), (1, vec![0]));
        assert_eq!(owners.reclaimed(), 0);
    }

    /// 上一个文档遗留的 key 必须能被新文档认领，否则宿主重开工程时拿不到任何 region。
    ///
    /// 【为什么必须是这条语义】REAPER 的 region 分配器在相同操作序列下复用地址；
    /// 而宿主不按 leaf-first 销毁时 `destroy_document` 不执行、遗留 key 一直留在
    /// 这张进程级表里。把跨文档重复判成错误，症状就是"重开只显示等待宿主音频，
    /// 删掉插件再添加就好"。
    #[test]
    fn a_dead_documents_key_is_reclaimed_rather_than_rejected() {
        let mut owners = RegionOwners::default();
        owners.register(101, 1, 0).unwrap();
        // 文档 1 没有走完销毁流程（`remove_document` 从未被调用）。
        assert_eq!(owners.register(101, 2, 4), Ok(()));
        assert_eq!(owners.owner_of(101), Some(2));
        assert_eq!(owners.resolve(&[101]).unwrap(), (2, vec![4]));
        assert_eq!(owners.reclaimed(), 1);
    }

    /// 清扫只动已死文档的登记，活文档的一条都不碰（多工程窗口并存）。
    #[test]
    fn the_alive_sweep_keeps_every_live_documents_regions() {
        let mut owners = RegionOwners::default();
        owners.register(101, 1, 0).unwrap();
        owners.register(102, 1, 1).unwrap();
        owners.register(202, 2, 0).unwrap();
        let swept = owners.retain_alive_documents(&|document| document == 2);
        assert_eq!(swept, 2);
        assert_eq!(owners.owner_of(101), None);
        assert_eq!(owners.owner_of(102), None);
        assert_eq!(owners.resolve(&[202]).unwrap(), (2, vec![0]));
    }
}
