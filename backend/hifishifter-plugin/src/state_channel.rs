//! 插件侧编辑状态；客户端只能提交参数，宿主的clip几何始终权威。
use hifishifter_kernel::state::{TimelineState, Track, TrackParamsState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// 会话轨道键只用于live关联；值完全来自宿主的modification/source persistentID。
pub(crate) type TrackBindings = BTreeMap<String, Vec<(String, String)>>;

/// 插件自有、宿主表达不到的**逐片段**属性。
///
/// 【为什么需要独立存储】clip 颜色与共振峰变形在宿主里没有对应概念（宿主只报告几何与
/// 媒体），而 `session.timeline` 每次 `ensure_loaded` 都会被整体替换。只有落在"有权威
/// 来源、且在共同投影入口重新播种"的地方，才不会被下一次重建冲掉 —— 这正是 `fades`
/// 用同一手法解决的同一个问题（见 [`EditState::fades`] 与 [`EditState::apply`]）。
///
/// 【键为什么是 clip id】`apply` 收到的时间线里 clip id 形如 `ara-item-{item GUID}`，
/// 即宿主的稳定 item 身份；与 `fades` 的键口径一致，冷重建后仍能对上。
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ClipLocalState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// 共振峰变形是**渲染输入**（内核 DSP 读它），所以必须随片段一起播种，
    /// 否则重新加载后声音会退回未变形的版本。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formant_morph: Option<hifishifter_kernel::state::ClipFormantMorph>,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub(crate) struct EditState {
    pub revision: u64,
    pub params: BTreeMap<String, TrackParamsState>,
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub bindings: TrackBindings,
    #[serde(
        default,
        skip_serializing_if = "crate::editor::parameter_atlas::ParameterAtlas::is_empty"
    )]
    pub atlas: crate::editor::parameter_atlas::ParameterAtlas,
    /// REAPER稳定item GUID对应的HFS自有形状；不写回宿主shape/c/S字段。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fades: BTreeMap<String, crate::fade::FadeStyle>,
    /// 插件自有、宿主表达不到的逐片段属性（颜色 / 共振峰变形）。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub clip_local: BTreeMap<String, ClipLocalState>,
    /// 跨组件共享的插件私有参数分组；GUID关系不参与REAPER folder或音频路由。
    #[serde(
        default,
        skip_serializing_if = "crate::editor::private_groups::TrackGroups::is_empty"
    )]
    pub groups: crate::editor::private_groups::TrackGroups,
    /// 组件恢复时延迟到完整ARA图ready后再关联，禁止逐对象创建时误套新序号。
    #[serde(skip)]
    pub needs_rebind: bool,
    /// **没能拿到可持久化身份**的条目（轨道 id）。
    ///
    /// 【为什么必须有这一档】身份缺失此前是一个**全局**判据：只要有一条记录没有绑定，
    /// 整份载荷就被判无效（`restore` 报错、`encode` 报错）。而"这条轨现在没有 region"
    /// 是**正常状态**（用户刚插了插件、刚删光了一条轨上的 item、或者宿主还没交付几何），
    /// 于是最常见的一种工程形状 —— 有一条空轨 —— 就让整份存档失效。
    /// 失效的下游是 `encode_archive` 退回空载荷：用户所有的参数曲线被**写进磁盘的
    /// 空块**覆盖掉。这是数据销毁，不是显示问题。
    ///
    /// 现在身份缺失只影响它自己那一条：数据留着、不套用、不进 `bindings`，
    /// 并在诊断里报成 `unbound=N`。见 [`EditState::apply`] 的过滤器。
    ///
    /// 【为什么不进存档】它就是"bindings 里没有这一条"的同义词，写出去只是重复。
    /// 恢复时由 [`EditState::restore`] 按同一判据重新算出来，因此不占格式代际。
    #[serde(skip)]
    pub unbound: BTreeSet<String>,
    /// 【G5】这份载荷**到达时**我们是否正处在一次 ARA 编辑会话里。
    ///
    /// 【为什么它是"要不要覆盖在线状态"的判据】撤销/重做的还原会话在概念上就是一次
    /// 编辑会话（`ARAInterface.h:2938-2943`），那种场合存档**必须**覆盖（undo 是权威）。
    /// 而切屏/重载 take 时交回来的旧载荷不是 —— 那时在线状态比它新，覆盖就是回退。
    ///
    /// 【为什么不进存档】它是"这一份载荷是怎么来的"这个**事件**的属性，不是用户数据。
    #[serde(skip)]
    pub arrived_in_edit_session: bool,
    /// 【G5】这份载荷来自 **ARA 文档存档**通道（而不是 VST3 组件状态）。
    ///
    /// 【为什么必须区分】两条通道的权威语义不同：
    ///   * 组件状态是**本实例**的存档，宿主用它做 undo / 复制轨 —— 恢复即权威。
    ///   * 文档存档在切屏、重载 take、重开工程时都会到达，其中**只有**重开工程
    ///     （在线状态为空）与撤销（编辑会话里）才是权威；其余携带的是**更旧**的值，
    ///     覆盖就是用户看到的"参数回退"。
    #[serde(skip)]
    pub from_archive: bool,
}

/// 一次冷恢复身份匹配的三种结论（见 [`pick_binding`]）。
#[derive(Clone, Debug, PartialEq, Eq)]
enum BindingPick {
    /// 唯一候选：接受。
    Unique(String),
    /// 多条候选并列：**拒绝**（挑一个等于把参数挂到别人的轨上）。
    Ambiguous,
    /// 一条候选都没有交集：这一轮还没观测到，继续等。
    Unobserved,
}

/// 冷恢复时给一份保存身份找宿主轨道：**唯一最大交集**。
///
/// 【为什么不是相等】保存的身份是"那条轨上所有 region 的 (modification, source) 对"。
/// 宿主每做一次编辑都可能让它多一条、少一条或换掉 modification（分割、删除、REAPER
/// 重建 item），于是"逐项相等"在真实工程里几乎永远不成立 —— 存档再也合不进来。
/// 而交集为零时它仍然是零证据，不能猜。
///
/// 【为什么平局必须拒绝】两条候选同样像，说明这次的模型图不足以区分它们；挑一个就是
/// 把用户的参数挂到错误的轨道上。宁可继续等，也不要那种错误。
fn pick_binding(current: &TrackBindings, saved: &[(String, String)]) -> BindingPick {
    // 【一、逐项相等优先】它是最强的证据：宿主原样把这份身份还了回来。
    let exact: Vec<&String> = current
        .iter()
        .filter(|(_, known)| *known == saved)
        .map(|(id, _)| id)
        .collect();
    match exact.len() {
        1 => return BindingPick::Unique(exact[0].clone()),
        0 => {}
        _ => return BindingPick::Ambiguous,
    }
    // 【二、交集】"多一条 / 少一条 region"是同一条轨的正常形状。
    let pair_overlap =
        |known: &Vec<(String, String)>| known.iter().filter(|pair| saved.contains(pair)).count();
    if let Some(pick) = unique_best(current, pair_overlap) {
        return pick;
    }
    // 【三、仅按源降级】宿主重建 item 时可能给新的 modification id 而 source 不变。
    // 只在"源的交集"唯一最大时接受 —— 同一文件被两个 item 共用时必然平局，仍拒绝。
    let saved_sources: BTreeSet<&str> = saved.iter().map(|pair| pair.1.as_str()).collect();
    let source_overlap = |known: &Vec<(String, String)>| {
        known
            .iter()
            .filter(|pair| saved_sources.contains(pair.1.as_str()))
            .count()
    };
    unique_best(current, source_overlap).unwrap_or(BindingPick::Unobserved)
}

/// 取"得分唯一最大"的候选；最高分为 0 视为没有候选。
fn unique_best(
    current: &TrackBindings,
    score: impl Fn(&Vec<(String, String)>) -> usize,
) -> Option<BindingPick> {
    let mut best = 0usize;
    let mut winners: Vec<&String> = Vec::new();
    for (id, known) in current {
        let value = score(known);
        if value > best {
            best = value;
            winners.clear();
            winners.push(id);
        } else if value == best && value > 0 {
            winners.push(id);
        }
    }
    match winners.len() {
        0 => None,
        1 => Some(BindingPick::Unique(winners[0].clone())),
        _ => Some(BindingPick::Ambiguous),
    }
}

impl EditState {
    /// 这份状态是否**从未装过任何东西**（重开工程 / 刚插上插件）。
    ///
    /// 【为什么它决定"覆盖还是补缺"】冷启动时在线状态是空的，存档就是唯一的数据源 ——
    /// 必须整份装上。而运行中到达的旧载荷不是：它携带的是**更旧**的参数，覆盖就是
    /// 用户看到的"参数回退"。
    pub fn is_pristine(&self) -> bool {
        self.params.is_empty()
            && self.tracks.is_empty()
            && self.atlas.is_empty()
            && self.fades.is_empty()
            && self.clip_local.is_empty()
    }

    /// 形状记录有界且只允许固定GUID键，冷重开不能按会话clip序号关联。
    fn validate_fades(&self) -> Result<(), String> {
        if self.fades.len() > 10000 {
            return Err("fade state budget exceeded".into());
        }
        for (key, style) in &self.fades {
            if key.len() != 38 || !key.starts_with('{') || !key.ends_with('}') {
                return Err("invalid fade item identity".into());
            }
            style.validate()?;
        }
        Ok(())
    }
    /// 逐片段自有属性同样有界：数量与颜色长度都设上限，避免不可信 state 撑爆内存。
    fn validate_clip_local(&self) -> Result<(), String> {
        if self.clip_local.len() > 10000 {
            return Err("clip local state budget exceeded".into());
        }
        for local in self.clip_local.values() {
            if local
                .color
                .as_deref()
                .is_some_and(|color| color.len() > 64 || color.contains('\0'))
            {
                return Err("invalid clip color".into());
            }
        }
        Ok(())
    }
    /// live图已明确归属时刷新成员；restore只接受唯一完整身份集合，失败保留原state。
    ///
    /// 【T-C6：没有身份的条目不再让整份失败】它进 `unbound`：数据留着、不套用、
    /// 不进 `bindings`。此前"没有身份"与"身份对不上"共用一条 `Err`，而那条 `Err`
    /// 的上游是"整份存档作废"，代价与收益完全不成比例 —— 一条空轨就能触发它。
    ///
    /// 【红线没动】**有**身份却匹配不到唯一轨道时仍然 `Err`：按会话序号猜归属会把
    /// 参数挂到别人的轨道上，那是比"少恢复一条"严重得多的错误。两者的区别是
    /// "这条从来没被绑定过"与"这条被绑定过、但现在对不上"。
    pub fn reconcile(&mut self, current: &TrackBindings) -> Result<(), String> {
        // 【G2】一张**空**的绑定图不是证据，是"这一轮什么都没看到"。
        //
        // 【为什么必须在这里挡住】本函数在**每一次** `end_editing` 都被调用。宿主刚
        // 启动、正在重建序列、或这一轮还没交付几何时，`track_bindings` 会是空的；
        // 若让它继续走下去，"每一条记录都查不到宿主身份"会被判成"这些轨都被删了"，
        // 于是整份编辑状态被清空 —— 用户看到的是"什么都没做，参数全没了"。
        //
        // 【为什么只挡 live 分支】冷恢复（`needs_rebind`）的严格契约必须保留：那时
        // "宿主一个身份都没给"就是"还不能合入"，继续走下去会把存档判成"身份缺失"
        // 并整份拒绝（那正是 `restored_edits_refuse_…` 钉住的语义）。而 live 分支
        // 的空图只是"这一轮什么都没看到"。
        if current.is_empty() && !self.needs_rebind {
            return Ok(());
        }
        let mut candidate = self.clone();
        let mut mapping = BTreeMap::new();
        let mut unbound: BTreeSet<String> = BTreeSet::new();
        let ids: BTreeSet<_> = self
            .tracks
            .iter()
            .map(|t| t.id.clone())
            .chain(self.params.keys().cloned())
            .collect();
        for old in &ids {
            // 存档里这一条的身份（缺席与空向量同义：没有可匹配的东西）。
            let saved = self.bindings.get(old).cloned().unwrap_or_default();
            let (new, identity) = if self.needs_rebind {
                if saved.is_empty() {
                    // 没有身份可匹配 —— 不是"对不上"，是"从来没能绑定"。
                    unbound.insert(old.clone());
                    continue;
                }
                // 【T-D1：身份不是"集合相等"，是"唯一最大交集"】
                //
                // 旧判据要求宿主**这一次**给出的身份向量与存档**逐项相等**。而
                // `bindings` 的值是"这条轨上所有 region 的 (modification, source) 对"，
                // 宿主每做一次编辑就可能让它多一条、少一条或换一个 modification：
                // 分割、删除、REAPER 重建 item 全都命中。于是"集合相等"在真实宿主里
                // 几乎**永远不成立**，存档从此再也合不进来 —— 用户看到的每次编辑都
                // 报"没能读回上次保存的参数"，就是这条契约的直接后果。
                //
                // 改判据而不改红线：
                //   * 交集为 0 ⇒ 不做任何猜测，仍报错（等待下一轮，载荷保留）；
                //   * 只有一条有交集 ⇒ 接受（多/少一条 region 是同一批源）；
                //   * 多条并**并列最大** ⇒ 仍然拒绝（那才是真正的歧义）。
                // 平局拒绝这一条是"I2 身份单调"的底线：两条候选同样像，就不能挑一个。
                match pick_binding(current, &saved) {
                    BindingPick::Unique(id) => (id, saved),
                    BindingPick::Ambiguous => {
                        return Err(
                            "ARA saved edit identity is ambiguous; multiple host tracks \
                             overlap it equally"
                                .into(),
                        )
                    }
                    BindingPick::Unobserved => {
                        return Err(
                            "ARA edit identity missing or ambiguous in rebuilt host graph".into(),
                        )
                    }
                }
            } else {
                // live轨道已由真实sequence边明确关联，不因轨内成员增删而丢掉编辑。
                let Some(identity) = current.get(old) else {
                    // 【G2】这条轨这一轮没出现在绑定表里。它可能只是**暂时**看不到
                    // （宿主正在重建 region sequence、几何还没交付、编辑中途），也可能
                    // 真的被删了。两种情况下"保留数据、不套用"都比删除安全：
                    // 删除是**不可逆**的数据销毁，而保留只让这条记录这一轮不参与投影。
                    //
                    // 此前这里 `continue` ⇒ 不建 mapping ⇒ 下面的 filter_map 把它删掉。
                    // 而轨道 id 来自宿主的 **realtime key**（`create_region_sequence`），
                    // 宿主重建序列时必然改变 —— 于是"切个屏回来，整轨参数消失"。
                    unbound.insert(old.clone());
                    continue;
                };
                // mute/空轨不是身份丢失；live序列仍明确，保留此前持久成员供解除/保存。
                // 冷恢复分支仍拒绝空或歧义身份，不能用旧轨道序号猜归属。
                if identity.is_empty() {
                    if saved.is_empty() {
                        // 两侧都没有身份：live 也给不出，存档也给不出 ⇒ 无法持久化。
                        unbound.insert(old.clone());
                        continue;
                    }
                    (old.clone(), saved)
                } else {
                    (old.clone(), identity.clone())
                }
            };
            // live归属已由宿主区域分配证明；复制轨道合法共享modification/source。
            // restore仍要求候选集中恰好一个身份匹配，不能仅按旧轨道序号猜测。
            if identity.is_empty()
                || identity
                    .iter()
                    .any(|pair| pair.0.is_empty() || pair.1.is_empty())
            {
                return Err("ARA edit identity is empty; cannot persist uniquely".into());
            }
            if mapping.values().any(|(known, _)| known == &new) {
                return Err("ARA saved edit identities are ambiguous; multiple records target one host track".into());
            }
            mapping.insert(old.clone(), (new, identity));
        }
        // 【为什么 unbound 的条目要**原样留下**】它们的数据是用户的资产（曲线、
        // 混音、算法选择）。丢掉它们等于"因为认不出是谁的，所以扔了"。留着之后：
        // `apply` 按 `unbound` 跳过（不猜归属），`encode` 原样写回（下一版
        // 插件或下一次宿主几何稳定后仍有机会恢复）。
        candidate.params = self
            .params
            .iter()
            .filter_map(|(id, p)| match mapping.get(id) {
                Some((new, _)) => Some((new.clone(), p.clone())),
                None if unbound.contains(id) => Some((id.clone(), p.clone())),
                None => None,
            })
            .collect();
        candidate.tracks = self
            .tracks
            .iter()
            .filter_map(|track| match mapping.get(&track.id) {
                Some((new, _)) => {
                    let mut track = track.clone();
                    track.id = new.clone();
                    Some(track)
                }
                None if unbound.contains(&track.id) => Some(track.clone()),
                None => None,
            })
            .collect();
        candidate.bindings = mapping.values().cloned().collect();
        candidate.unbound = unbound;
        for record in candidate.atlas.regions.values_mut() {
            if let Some((root, _)) = mapping.get(&record.root) {
                record.root = root.clone();
            }
        }
        for record in candidate.atlas.copy_seeds.values_mut() {
            if let Some((root, _)) = mapping.get(&record.root) {
                record.root = root.clone();
            }
        }
        // 【U-7：`gaps` 必须与 `params` 用同一条规则】
        //
        // "未锁定"的参数线里，**没有 clip 覆盖的那一段**住在这一层（`atlas.gaps`）。
        // round 23 修好了 `params` 的 unbound 保留，却漏了这里：宿主重建 region sequence
        // ⇒ 根轨 id 变 ⇒ 这一层被整片清空。用户看到的是"那个 clip 附近的那段线没了" ——
        // 而那正是他们最可能在两段音频之间画的那一段。
        candidate.atlas.gaps = self
            .atlas
            .gaps
            .iter()
            .filter_map(|(root, params)| match mapping.get(root) {
                Some((new, _)) => Some((new.clone(), params.clone())),
                // 这一轮认不出是谁的 ⇒ 数据留着、不套用（与 `params` 同一纪律）。
                None if candidate.unbound.contains(root) => Some((root.clone(), params.clone())),
                None => None,
            })
            .collect();
        candidate.needs_rebind = false;
        *self = candidate;
        Ok(())
    }

    /// 只复制用户曲线与已有轨道参数，不接受客户端移动/替换clip。
    ///
    /// 【为什么现在只是转调】工作区交接的回写载荷**只装它真正会被读的字段**
    /// （曲线 + 每条轨道的 5 个可写字段），不再传一整份 `TimelineState`。
    /// 校验规则一条没变，只是入口收窄了 —— 于是这里保留成对
    /// [`EditState::merge_params`] 的一层壳，供既有测试与调用点沿用。
    pub fn merge(
        &self,
        host: &TimelineState,
        client: &TimelineState,
        base_revision: u64,
    ) -> Result<Self, String> {
        self.merge_params(
            host,
            &client.params_by_root_track,
            &client.tracks,
            base_revision,
        )
    }

    /// 校验并合并**用户拥有的**两类东西：参数曲线与轨道混音。
    ///
    /// 【为什么不接受 clip 几何】宿主拥有几何。客户端把几何发上来只会被丢弃 ——
    /// 而"发一个会被丢弃的东西、再在接收侧校验它"正是旧协议里最大的一处误解来源
    /// （旧 `Commit` 传整份 `TimelineState`，这里只读其中两项）。
    pub fn merge_params(
        &self,
        host: &TimelineState,
        params: &BTreeMap<String, TrackParamsState>,
        tracks: &[Track],
        base_revision: u64,
    ) -> Result<Self, String> {
        if base_revision != self.revision {
            return Err("Conflict: edit revision changed".into());
        }
        let host_ids: BTreeSet<_> = host.tracks.iter().map(|track| track.id.as_str()).collect();
        let client_ids: BTreeSet<_> = tracks.iter().map(|track| track.id.as_str()).collect();
        if host_ids.len() != host.tracks.len()
            || client_ids.len() != tracks.len()
            || client_ids != host_ids
            || tracks.iter().any(|track| {
                !host.tracks.iter().any(|known| known.id == track.id)
                    || !track.volume.is_finite()
                    || !(0.0..=hifishifter_kernel::gain::GAIN_MAX).contains(&track.volume)
            })
        {
            return Err("invalid or unknown track".into());
        }
        for (id, params) in params {
            if !host.tracks.iter().any(|track| track.id == *id)
                || !params.frame_period_ms.is_finite()
                || !(0.1..=100.0).contains(&params.frame_period_ms)
            {
                return Err("invalid curve track or frame period".into());
            }
            for curve in [
                &params.pitch_orig,
                &params.pitch_edit,
                &params.tension_orig,
                &params.tension_edit,
            ]
            .into_iter()
            .chain(params.extra_curves.values())
            {
                if curve.len() > 1_000_000
                    || curve
                        .iter()
                        .any(|value| !value.is_finite() || value.abs() > 10000.0)
                {
                    return Err("invalid parameter curve".into());
                }
            }
            if params.extra_params.values().any(|value| !value.is_finite()) {
                return Err("invalid static parameter".into());
            }
        }
        let mut merged = self.clone();
        merged.revision = self.revision.checked_add(1).ok_or("revision exhausted")?;
        merged
            .params
            .retain(|id, _| !host_ids.contains(id.as_str()));
        merged.params.extend(params.clone());
        merged
            .tracks
            .retain(|track| !host_ids.contains(track.id.as_str()));
        merged.tracks.extend(tracks.iter().cloned());
        Ok(merged)
    }
    /// 当前宿主几何上叠加持久化参数，旧轨道记录不会创造虚构轨道。
    pub fn apply(&self, timeline: &mut TimelineState) {
        // 未完成身份关联的恢复记录绝不能沿旧会话序号静默应用。
        if self.needs_rebind {
            return;
        }
        // 【为什么按条目过滤而不是整份返回】身份缺失是**逐条**的事实：一条空轨没有
        // region，不代表同一份载荷里另外十条轨的曲线也该被丢掉。判据用
        // [`Self::unbound`]（"这条从来没被绑定过"），而不是 `needs_rebind`
        // （"整份还没关联"）—— 后者一旦为真就什么都不套，正是"全部参数丢失"的
        // 另一种写法。
        timeline.params_by_root_track = self
            .params
            .iter()
            .filter(|(id, _)| {
                !self.unbound.contains(*id) && timeline.tracks.iter().any(|t| t.id == **id)
            })
            .map(|(id, p)| (id.clone(), p.clone()))
            .collect();
        for track in &mut timeline.tracks {
            if self.unbound.contains(&track.id) {
                continue;
            }
            if let Some(edited) = self.tracks.iter().find(|t| t.id == track.id) {
                track.compose_enabled = edited.compose_enabled;
                track.pitch_analysis_algo = edited.pitch_analysis_algo.clone();
                track.volume = edited.volume;
                track.muted = edited.muted;
                track.solo = edited.solo;
            }
        }
        // 插件自有、宿主表达不到的逐片段属性：必须在**共同的投影入口**重新播种，
        // 否则会被下一次 `ensure_loaded` 的整份重建冲掉（颜色消失、共振峰变形失效）。
        for clip in &mut timeline.clips {
            let Some(local) = self.clip_local.get(&clip.id) else {
                continue;
            };
            if let Some(color) = &local.color {
                clip.color = color.clone();
            }
            if local.formant_morph.is_some() {
                clip.formant_morph = local.formant_morph.clone();
            }
        }
    }

    /// 本构建支持的**最高**状态 schema 代际。
    ///
    /// 【为什么 `encode` 不写这个常量】版本号在这里是**向后兼容**手段：只带 atlas 的
    /// 载荷写 2、带 groups 写 5……旧构建（它只认自己那一代及更低）因此仍能读。写死最高
    /// 代际会让所有旧构建一律拒绝，把"能读"变成"全丢"。这个常量只用于 `restore` 判断
    /// "由更新的插件保存"。
    ///
    /// 【为什么第 8 代必须升】第 8 代把权威曲线从"工程帧索引"改成"**源帧**索引"，
    /// 并把 `SourceCurve.basis` 从对象改成 `null`（见 `SourceCurve::basis`）。
    /// 旧构建读新存档时那个字段会解析失败 —— 升代让它得到"这份存档由更新的插件保存"
    /// 这句**可行动**的话，而不是一条看不懂的 serde 错误。反过来，新构建读第 7 代及
    /// 更早的存档仍然完全正确（`basis` 有对象 ⇒ 走旧索引分支），因此**不需要折算**。
    const CURRENT_STATE_VERSION: u64 = 8;

    /// 有版本且有界的组件state；旧空state保持默认编辑。
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.groups.validate()?;
        self.atlas.validate()?;
        self.validate_fades()?;
        self.validate_clip_local()?;
        // 【为什么不再拒绝"没有身份"的条目】此前这里返回 `Err`，而它唯一的调用方
        // `encode_archive` 把错误降级成 `Ok(Vec::new())` —— 宿主于是收到一份**空存档**
        // 并把它写进工程文件。一条空轨（没有 region、因此没有身份）就足以触发它，
        // 而"有一条空轨"是最常见的工程形状之一。
        //
        // 契约应当是**往返对称**：`restore(encode(s))` 对任意 `s` 都能成功。
        // 做不到这一点时，"拒绝写"不是保守，而是用整份数据去赌一条记录。
        // 无法绑定的条目现在由 `unbound` 承载：数据留着、不套用、不进 `bindings`。
        let version = if !self.atlas.authorities.is_empty() {
            8
        } else if !self.clip_local.is_empty() {
            6
        } else if !self.groups.is_empty() {
            5
        } else if !self.fades.is_empty() {
            4
        } else if self.atlas.is_empty() {
            2
        } else {
            3
        };
        // 【为什么不直接序列化 `self`】`atlas.regions` 是**每帧派生**的投影缓存，它的
        // 权威是 `atlas.authorities`。写出派生缓存会让同一份采样在存档里出现两次
        // （曲线是 `Arc<Vec<f32>>`，内存里共享、序列化时各写一份），而它读回来之后
        // 又会被 `follow_geometry` 立刻覆盖。清掉它：存档只带权威。
        let mut trimmed = self.clone();
        trimmed.atlas.regions.clear();
        let bytes = serde_json::to_vec(&serde_json::json!({"version":version,"edits":trimmed}))
            .map_err(|e| e.to_string())?;
        if bytes.len() > hifishifter_host_link::MAX_COMPONENT_STATE {
            return Err("state exceeds transport budget".into());
        }
        Ok(bytes)
    }
    /// 恢复使乐观并发revision前进，防止旧GUI再次覆盖宿主undo/恢复。
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        if bytes.len() > hifishifter_host_link::MAX_COMPONENT_STATE {
            return Err("state too large".into());
        }
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        // 【为什么区分"更新版保存"与"损坏"】两者此前都报同一句
        // `unsupported state version`，用户看到"编辑全没了"却不知道原因。版本高于本
        // 构建时明确说明"由更新的插件保存"，让人知道该升级而不是以为档坏了。
        let version = value["version"].as_u64().unwrap_or(0);
        if version == 0 {
            return Err("unsupported state version".into());
        }
        if version > Self::CURRENT_STATE_VERSION {
            return Err(format!(
                "state was saved by a newer plugin version (state {version} > supported {})",
                Self::CURRENT_STATE_VERSION
            ));
        }
        let mut restored: Self =
            serde_json::from_value(value["edits"].clone()).map_err(|e| e.to_string())?;
        restored.groups.validate()?;
        restored.validate_fades()?;
        restored.atlas = restored.atlas.reserve_restored()?;
        if version == 1 && (!restored.params.is_empty() || !restored.tracks.is_empty()) {
            return Err("legacy ARA edits have no persistent identity; cannot restore by session track number".into());
        }
        // 【为什么逐条降级而不是整份拒绝】此前这里只要有一条记录没有身份就 `Err`，
        // 而两条恢复通道对这个错误的处置都是"整份不恢复"：ARA 存档那条只 `log::error!`
        // 然后 `return Ok(())`（宿主以为成功），VST3 `setState` 那条把错误传上去
        // （整块组件状态一起失败）。两种结局都是"用户所有参数变回默认值"。
        //
        // 现在：没有身份的条目进 `unbound`，其余照常。它们的**数据不丢**（仍在
        // `params` / `tracks` 里，也会被原样写回下一份存档），只是**不套用** ——
        // 按会话序号猜归属是被明确禁止的（见 `reconcile` 的冷恢复分支），
        // 而"保留但不猜"既守住了那条红线，又不再拿整份载荷陪葬。
        restored.unbound = restored
            .tracks
            .iter()
            .map(|t| &t.id)
            .chain(restored.params.keys())
            .filter(|id| restored.bindings.get(*id).is_none_or(Vec::is_empty))
            .cloned()
            .collect();
        restored.needs_rebind = !restored.tracks.is_empty() || !restored.params.is_empty();
        restored.validate_clip_local()?;
        restored.revision = self
            .revision
            .max(restored.revision)
            .checked_add(1)
            .ok_or("revision exhausted")?;
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn host() -> TimelineState {
        serde_json::from_value(serde_json::json!({"tracks":[{"id":"track","name":"host","order":0}],"clips":[],"bpm":120,"project_sec":1})).unwrap()
    }
    #[test]
    fn edits_preserve_host_geometry_and_reject_stale_writer() {
        let mut client = host();
        client.tracks[0].volume = 0.5;
        client.project_sec = 400.;
        let state = EditState::default().merge(&host(), &client, 0).unwrap();
        let mut rendered = host();
        state.apply(&mut rendered);
        assert_eq!(rendered.tracks[0].volume, 0.5);
        assert_eq!(rendered.project_sec, 1.);
        assert!(state.merge(&host(), &client, 0).is_err());
    }
    /// **不变量：插件自有逐片段属性必须在共同投影入口重新播种。**
    ///
    /// 【为什么这是回归测试】`session.timeline` 每次 `ensure_loaded` 都被整体替换；
    /// 不落在"有权威来源 + 在共同投影入口重新播种"的地方，颜色会在下一次重建后消失、
    /// 共振峰变形会失效（它还是渲染输入）。键是 `ara-item-{guid}`，与投影里的 clip id
    /// 同口径，冷重建后仍能对上。
    #[test]
    fn clip_local_properties_are_reseeded_by_apply() {
        let mut edits = EditState::default();
        edits.clip_local.insert(
            "ara-item-{11111111-1111-1111-1111-111111111111}".into(),
            ClipLocalState {
                color: Some("#abcdef".into()),
                formant_morph: Some(hifishifter_kernel::state::ClipFormantMorph {
                    enabled: true,
                    ..Default::default()
                }),
            },
        );
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"ara-item-{11111111-1111-1111-1111-111111111111}",
                      "track_id":"track","name":"c","start_sec":0,"length_sec":1,
                      "takes":[{"id":"t","source_start_sec":0,"source_end_sec":1}]}],
            "bpm":120,"project_sec":1
        }))
        .unwrap();
        let before = timeline.clips[0].color.clone();
        assert_ne!(before, "#abcdef", "前提：宿主时间线没有这条颜色");
        edits.apply(&mut timeline);
        assert_eq!(timeline.clips[0].color, "#abcdef");
        assert!(
            timeline.clips[0].formant_morph.is_some(),
            "共振峰变形是渲染输入，必须随投影播种"
        );
    }
    #[test]
    fn invalid_curve_and_unknown_track_are_rejected() {
        let mut client = host();
        client.params_by_root_track.insert(
            "track".into(),
            TrackParamsState {
                pitch_edit: vec![f32::NAN],
                ..Default::default()
            },
        );
        assert!(EditState::default().merge(&host(), &client, 0).is_err());
        client.params_by_root_track.clear();
        client.tracks[0].id = "foreign".into();
        assert!(EditState::default().merge(&host(), &client, 0).is_err());
    }
    /// live空轨/mute只暂停区域，不丢曲线或持久成员；冷恢复空身份仍拒绝。
    #[test]
    fn ui_inventory_live_empty_members_keep_parameters_and_previous_persistent_identity() {
        let mut client = host();
        client.params_by_root_track.insert(
            "track".into(),
            TrackParamsState {
                frame_period_ms: 5.,
                pitch_edit: vec![61., 63.],
                ..Default::default()
            },
        );
        let mut state = EditState::default().merge(&host(), &client, 0).unwrap();
        let members = vec![("mod".into(), "source".into())];
        state
            .reconcile(&BTreeMap::from([("track".into(), members.clone())]))
            .unwrap();
        state
            .reconcile(&BTreeMap::from([("track".into(), vec![])]))
            .unwrap();
        assert_eq!(state.params["track"].pitch_edit, [61., 63.]);
        assert_eq!(state.bindings["track"], members);
        let mut restored = EditState::default();
        restored.restore(&state.encode().unwrap()).unwrap();
        assert!(restored
            .reconcile(&BTreeMap::from([("track".into(), vec![])]))
            .is_err());
    }

    #[test]
    fn local_view_merge_preserves_other_track_curves_and_volume_after_restore() {
        let a = host();
        let mut b = host();
        b.tracks[0].id = "other".into();
        let mut edited_a = a.clone();
        edited_a.tracks[0].volume = 0.5;
        edited_a.params_by_root_track.insert(
            "track".into(),
            TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![61.0, 63.0],
                ..Default::default()
            },
        );
        let first = EditState::default().merge(&a, &edited_a, 0).unwrap();
        let mut edited_b = b.clone();
        edited_b.tracks[0].volume = 0.25;
        edited_b.params_by_root_track.insert(
            "other".into(),
            TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![70.0],
                ..Default::default()
            },
        );
        let mut second = first.merge(&b, &edited_b, 1).unwrap();
        let identities = BTreeMap::from([
            ("track".into(), vec![("mod-a".into(), "source".into())]),
            ("other".into(), vec![("mod-b".into(), "source".into())]),
        ]);
        second.reconcile(&identities).unwrap();
        assert_eq!(second.params["track"].pitch_edit, [61.0, 63.0]);
        assert_eq!(
            second
                .tracks
                .iter()
                .find(|t| t.id == "track")
                .unwrap()
                .volume,
            0.5
        );
        let mut restored = EditState::default();
        restored.restore(&second.encode().unwrap()).unwrap();
        restored.reconcile(&identities).unwrap();
        let mut applied = a.clone();
        restored.apply(&mut applied);
        assert_eq!(
            applied.params_by_root_track["track"].pitch_edit,
            [61.0, 63.0]
        );
        assert_eq!(applied.tracks[0].volume, 0.5);
    }

    #[test]
    fn duplicate_track_ids_cannot_impersonate_a_complete_authorized_view() {
        let mut known = host();
        let mut other = known.tracks[0].clone();
        other.id = "other".into();
        known.tracks.push(other);
        let mut client = known.clone();
        client.tracks[1] = client.tracks[0].clone();
        assert!(EditState::default().merge(&known, &client, 0).is_err());
    }

    #[test]
    fn restored_edits_refuse_ambiguous_empty_and_missing_host_identity_without_rebinding() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        let original = BTreeMap::from([("track".into(), vec![("mod".into(), "source".into())])]);
        state.reconcile(&original).unwrap();
        let bytes = state.encode().unwrap();
        for current in [
            BTreeMap::new(),
            BTreeMap::from([("new".into(), vec![])]),
            BTreeMap::from([
                ("a".into(), vec![("mod".into(), "source".into())]),
                ("b".into(), vec![("mod".into(), "source".into())]),
            ]),
        ] {
            let mut restored = EditState::default();
            restored.restore(&bytes).unwrap();
            assert!(restored
                .reconcile(&current)
                .unwrap_err()
                .contains("identity"));
            let mut timeline = host();
            restored.apply(&mut timeline);
            assert_eq!(timeline.tracks[0].volume, 1.0, "未解决身份不套用临时序号");
            assert!(restored.needs_rebind);
        }
        // live 空轨可能是 mute，沿已确认的身份保留；上述冷恢复仍拒绝空/歧义身份。
        state
            .reconcile(&BTreeMap::from([("track".into(), vec![])]))
            .unwrap();
        assert_eq!(state.bindings, original);
        assert_eq!(state.tracks[0].volume, 0.25);
    }

    /// **T-D1：多一条 region 不该让整条轨的身份失效。**
    ///
    /// 【为什么这是症状三的主回归】保存的身份是"这条轨上所有 region 的对"。宿主分割
    /// 一次就多一条，而旧判据要求逐项相等 —— 于是存档从此再也合不进来，用户每次编辑
    /// 都看到"没能读回上次保存的参数"。交集唯一时接受，才符合"身份是同一批源"的事实。
    /// **这一轮看不到的轨必须留着数据，而不是被删掉。**
    ///
    /// 【为什么这是"移动一下 item / 切个屏回来，参数全没了"的回归测试】轨道 id 来自
    /// 宿主的 **realtime key**，而宿主重建 region sequence 时它必然改变。旧实现在
    /// live 分支里对"查不到的旧轨"直接 `continue` ⇒ 不建 mapping ⇒ 下面被 filter 掉。
    /// 删除是**不可逆**的数据销毁；保留只让这条记录这一轮不参与投影。
    #[test]
    fn a_temporarily_missing_track_keeps_its_parameters() {
        let mut state = EditState {
            params: BTreeMap::from([("a".to_owned(), params_fixture())]),
            tracks: vec![track_fixture("a")],
            bindings: BTreeMap::from([("a".to_owned(), vec![("mod-a".into(), "src".into())])]),
            ..EditState::default()
        };
        // 宿主这一轮只交出了另一条轨（或者什么都没交出来）。
        let current = TrackBindings::from([(
            "other".to_owned(),
            vec![("mod-b".to_owned(), "src".to_owned())],
        )]);
        state.reconcile(&current).unwrap();
        assert!(state.params.contains_key("a"), "数据必须留着");
        assert!(state.tracks.iter().any(|track| track.id == "a"));
        assert!(state.unbound.contains("a"), "留着但不套用");
        // `apply` 必须跳过它：猜归属比"这一轮不显示"严重得多。
        let mut host = host();
        host.tracks.push(track_fixture("a"));
        state.apply(&mut host);
        assert!(!host.params_by_root_track.contains_key("a"));
    }

    /// **一张空的绑定图不是证据**：它只说明"这一轮什么都没看到"。
    #[test]
    fn an_empty_binding_graph_never_erases_the_edit_state() {
        let state = EditState {
            params: BTreeMap::from([("a".to_owned(), params_fixture())]),
            tracks: vec![track_fixture("a")],
            bindings: BTreeMap::from([("a".to_owned(), vec![("mod-a".into(), "src".into())])]),
            ..EditState::default()
        };
        let mut live = state.clone();
        live.reconcile(&TrackBindings::new()).unwrap();
        assert_eq!(live.params.len(), 1, "空图不得清空参数");
        assert_eq!(live.tracks.len(), 1);
        assert!(live.unbound.is_empty(), "什么都没看到 ≠ 认不出这一条");
    }
    #[test]
    fn an_extra_region_does_not_invalidate_the_track_identity() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        state
            .reconcile(&BTreeMap::from([(
                "track".into(),
                vec![("mod".into(), "source".into())],
            )]))
            .unwrap();
        let bytes = state.encode().unwrap();
        let mut restored = EditState::default();
        restored.restore(&bytes).unwrap();
        // 宿主这次交付了**两条**同源 region（分割后的形状）。
        let current = BTreeMap::from([(
            "track".into(),
            vec![
                ("mod".into(), "source".into()),
                ("mod-2".into(), "source".into()),
            ],
        )]);
        restored.reconcile(&current).unwrap();
        assert_eq!(restored.tracks[0].volume, 0.25);
        assert!(!restored.needs_rebind);
    }

    /// **T-D2：modification 被宿主重建、源没变时，按源唯一匹配。**
    ///
    /// 【为什么要这一档】REAPER 重建 item 会给新的 modification id，而音频源还是同一个
    /// 文件。只按 (modification, source) 对匹配等于"每次重建都失联"；只按源匹配又要
    /// 防止同一文件被两条轨共用 —— 所以只在唯一最大时接受。
    #[test]
    fn a_rebuilt_modification_still_matches_by_source_when_unique() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        state
            .reconcile(&BTreeMap::from([(
                "track".into(),
                vec![("mod".into(), "source".into())],
            )]))
            .unwrap();
        let bytes = state.encode().unwrap();
        let mut restored = EditState::default();
        restored.restore(&bytes).unwrap();
        let current = BTreeMap::from([(
            "track".into(),
            vec![("mod-rebuilt".into(), "source".into())],
        )]);
        restored.reconcile(&current).unwrap();
        assert_eq!(restored.tracks[0].volume, 0.25);
    }

    /// **T-D1 的反面：两条轨并列最大时仍然拒绝。**
    ///
    /// 【为什么必须有这一条】放宽"集合相等"之后，唯一还能保护用户的是"平局不猜"。
    /// 两个候选同样像，说明这一轮的图不足以区分它们。
    #[test]
    fn equal_overlap_between_two_tracks_is_still_refused() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        // 保存的是**两条** region。
        state
            .reconcile(&BTreeMap::from([(
                "track".into(),
                vec![
                    ("mod".into(), "source".into()),
                    ("mod-2".into(), "source-2".into()),
                ],
            )]))
            .unwrap();
        let bytes = state.encode().unwrap();
        let mut restored = EditState::default();
        restored.restore(&bytes).unwrap();
        // 现场拆成两条轨，各带其中一条 —— 交集都是 1、没有任何一条"逐项相等"，
        // 所以必须拒绝（"挑一个"等于把参数挂到别人的轨上）。
        let current = BTreeMap::from([
            ("a".into(), vec![("mod".into(), "source".into())]),
            ("b".into(), vec![("mod-2".into(), "source-2".into())]),
        ]);
        let error = restored.reconcile(&current).unwrap_err();
        assert!(error.contains("identit"), "{error}");
        assert!(restored.needs_rebind, "拒绝后必须保留待绑标记");
    }

    #[test]
    fn live_membership_change_updates_saved_identity_and_legacy_nonempty_state_is_rejected() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        let mut identity =
            BTreeMap::from([("track".into(), vec![("mod-one".into(), "source".into())])]);
        state.reconcile(&identity).unwrap();
        identity
            .get_mut("track")
            .unwrap()
            .push(("mod-two".into(), "source".into()));
        state.reconcile(&identity).unwrap();
        assert_eq!(state.tracks[0].volume, 0.25);
        assert_eq!(state.bindings["track"].len(), 2);
        let mut legacy =
            serde_json::from_slice::<serde_json::Value>(&state.encode().unwrap()).unwrap();
        legacy["version"] = serde_json::json!(1);
        assert!(state
            .restore(&serde_json::to_vec(&legacy).unwrap())
            .unwrap_err()
            .contains("legacy"));
        assert_eq!(state.revision, 1, "拒绝旧state保留现有编辑");
    }

    /// 版本号是"最低够用的代际"（**向后兼容**手段），且"更新版保存的"与"损坏的"
    /// 给出可区分的报错。
    #[test]
    fn a_newer_state_version_is_reported_distinctly_from_corruption() {
        let mut state = EditState::default();
        let value: serde_json::Value = serde_json::from_slice(&state.encode().unwrap()).unwrap();
        assert_eq!(
            value["version"].as_u64(),
            Some(2),
            "空编辑只带 atlas 段 ⇒ 写最低代际 2，旧构建也能读"
        );
        // 更高版本：明确说"由更新的插件保存"，而不是含混的 unsupported。
        let mut newer = value.clone();
        newer["version"] = serde_json::json!(EditState::CURRENT_STATE_VERSION + 1);
        let error = state
            .restore(&serde_json::to_vec(&newer).unwrap())
            .unwrap_err();
        assert!(error.contains("newer plugin version"), "{error}");
        // 缺版本：仍是"不支持"。
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("version");
        assert!(state
            .restore(&serde_json::to_vec(&missing).unwrap())
            .unwrap_err()
            .contains("unsupported"));
    }

    fn params_fixture() -> TrackParamsState {
        serde_json::from_value(serde_json::json!({"frame_period_ms": 5.0})).unwrap()
    }
    fn track_fixture(id: &str) -> Track {
        serde_json::from_value(serde_json::json!({"id": id, "name": id, "order": 0})).unwrap()
    }

    /// **I3-3：往返对称 —— 任意一份编辑都必须能被自己读回来。**
    ///
    /// 【为什么这条曾经不成立】`encode` 与 `restore` 对"没有身份"用了两套严格度：
    /// 写的那一侧允许 `reconcile` 留下空绑定，读的那一侧却要求每条都有非空绑定，
    /// 于是 `restore(encode(s))` 对 `s` 报错。而 `encode` 自己也会因为同一条记录
    /// 失败 —— 它的调用方 `encode_archive` 把失败降级成**空载荷**，宿主据此把用户的
    /// 全部参数写成"没有编辑"。一条空轨就足够触发：它没有 region ⇒ 没有身份。
    #[test]
    fn encode_then_restore_round_trips_every_key() {
        let mut state = EditState::default();
        for id in ["T_empty", "T_a", "T_b"] {
            state.tracks.push(track_fixture(id));
            state.params.insert(id.to_owned(), params_fixture());
        }
        state
            .bindings
            .insert("T_a".into(), vec![("m1".into(), "s1".into())]);
        state
            .bindings
            .insert("T_b".into(), vec![("m2".into(), "s2".into())]);
        let bytes = state.encode().expect("一条空轨不得让整份存档失败");
        let mut restored = EditState::default();
        restored
            .restore(&bytes)
            .expect("自己写的存档必须自己读得回来");
        let keys = |value: &EditState| value.params.keys().cloned().collect::<BTreeSet<_>>();
        assert_eq!(keys(&restored), keys(&state));
        assert_eq!(restored.tracks.len(), state.tracks.len());
        assert_eq!(
            restored.unbound,
            BTreeSet::from(["T_empty".to_owned()]),
            "只有真正没有身份的那一条进 unbound"
        );
        // 再存一次仍然带着全部三条（数据不因"认不出是谁的"而被丢掉）。
        let again = EditState::default().encode().unwrap();
        assert!(!again.is_empty());
        assert_eq!(restored.params.len(), 3);
    }

    /// **I3-3 的另一半：没有身份的条目不得被"套用"到一条同名的 live 轨上。**
    ///
    /// 【为什么这条是红线】`ara-track-N` 是会话内的槽位号，跨会话并不稳定。
    /// 按它猜归属会把参数挂到别人的轨道上 —— 那比"少恢复一条"严重得多。
    /// 正确的处置是：数据留着、不套用、在诊断里报成 `unbound=N`。
    #[test]
    fn a_track_without_regions_is_kept_but_never_applied_by_slot_number() {
        let mut state = EditState::default();
        state.tracks.push(track_fixture("ara-track-2"));
        state.tracks[0].volume = 0.25;
        state
            .params
            .insert("ara-track-2".to_owned(), params_fixture());
        // live 图：这条轨存在，但没有任何 region ⇒ 身份是空向量，存档里也没有记录。
        state
            .reconcile(&BTreeMap::from([("ara-track-2".into(), vec![])]))
            .expect("没有身份不是失败");
        assert!(state.unbound.contains("ara-track-2"));
        assert!(state.bindings.is_empty(), "不得凭空造一个身份");
        assert_eq!(state.params.len(), 1, "数据必须留着");
        state.encode().expect("存档必须成功");
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks": [{"id": "ara-track-2", "name": "t", "order": 0}],
            "clips": [], "bpm": 120, "project_sec": 1
        }))
        .unwrap();
        state.apply(&mut timeline);
        assert_eq!(timeline.tracks[0].volume, 1.0, "按槽位号猜归属是红线");
        assert!(timeline.params_by_root_track.is_empty());
    }

    #[test]
    fn duplicate_saved_track_identities_cannot_collapse_into_one_rebuilt_track() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        let identity = vec![("mod".into(), "source".into())];
        state
            .reconcile(&BTreeMap::from([("track".into(), identity.clone())]))
            .unwrap();
        let mut saved =
            serde_json::from_slice::<serde_json::Value>(&state.encode().unwrap()).unwrap();
        let mut duplicate = saved["edits"]["tracks"][0].clone();
        duplicate["id"] = serde_json::json!("other-old-track");
        saved["edits"]["tracks"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        saved["edits"]["bindings"]["other-old-track"] = serde_json::to_value(&identity).unwrap();
        let mut restored = EditState::default();
        restored
            .restore(&serde_json::to_vec(&saved).unwrap())
            .unwrap();
        assert!(
            restored
                .reconcile(&BTreeMap::from([("new-track".into(), identity)]))
                .is_err(),
            "保存数据中的重复身份不能合并串轨"
        );
        assert!(restored.needs_rebind);
    }
}
