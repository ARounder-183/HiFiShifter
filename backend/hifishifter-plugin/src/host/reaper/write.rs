//! 核对锁定官方REAPER API的主线程片段写入口；GUID来自真实直接take绑定，不猜活动工程。
use super::{checked, Guid, HostVtbl, NativeHost};
use crate::host::geometry::HostClipGeometry;
use crate::sync::MutexExt;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

/// 宿主拒绝重建这个 item 时的**原文**错误（`SetItemStateChunk` 返回 false）。
///
/// 【为什么单独一个常量】它不是"写坏了"，而是宿主的一条策略：带**在线 ARA FX** 的 item
/// 一律不接受整份 chunk 重建（真机取证见 [`super::take_env`]）。调用方要靠它区分
/// "这条路在宿主上不存在"与"我们的 chunk 拼错了"，前者才允许退到宿主动作。
pub(crate) const ITEM_STATE_REFUSED: &str = "REAPER rejected the item state";

pub(super) type SetValue = unsafe extern "C" fn(*mut c_void, *const c_char, f64) -> bool;
pub(super) type SetString =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char, bool) -> bool;
pub(super) type Track = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
pub(super) type Split = unsafe extern "C" fn(*mut c_void, f64) -> *mut c_void;
pub(super) struct SplitApi {
    pub split: Split,
    pub active_take: Track,
}
pub(super) type Move = unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool;
pub(super) type Begin = unsafe extern "C" fn(*mut c_void);
pub(super) type End = unsafe extern "C" fn(*mut c_void, *const c_char, i32);
pub(super) type Update = unsafe extern "C" fn(*mut c_void);
pub(super) type Arrange = unsafe extern "C" fn();
pub(super) type UndoAction = unsafe extern "C" fn(*mut c_void) -> i32;
pub(super) type UndoLabel = unsafe extern "C" fn(*mut c_void) -> *const c_char;
pub(super) type UndoEntry = unsafe extern "C" fn(*mut c_void, i32) -> *const c_char;
pub(super) struct HistoryApi {
    pub undo: UndoAction,
    pub redo: UndoAction,
    pub can_undo: UndoLabel,
    pub can_redo: UndoLabel,
    pub current: UndoAction,
    pub count: UndoAction,
    pub entry: UndoEntry,
}

pub(super) struct WriteApi {
    pub set_item: SetValue,
    pub set_take: SetValue,
    pub set_take_string: SetString,
    pub item_track: Track,
    pub move_item: Move,
    pub begin: Begin,
    pub end: End,
    pub update: Update,
    pub arrange: Arrange,
}

/// 地址只在所属UI线程使用；拥有host接口引用不等同item/take生命周期，每次写前都验证。
#[derive(Clone)]
pub(crate) struct HostClipTarget {
    host: Arc<NativeHost>,
    project: usize,
    item: usize,
    take: usize,
    track: usize,
    pub geometry: HostClipGeometry,
}
/// 只由已验证parent轨道枚举器构建；当前take/item GUID与项目归属再次核对。
pub(super) fn inventory_target(
    host: Arc<NativeHost>,
    project: usize,
    track: usize,
    item: usize,
    take: usize,
    geometry: HostClipGeometry,
    authorized: &impl Fn() -> bool,
) -> Result<HostClipTarget, String> {
    let target = HostClipTarget {
        host,
        project,
        track,
        item,
        take,
        geometry,
    };
    target.verify(authorized)?;
    Ok(target)
}
impl HostClipTarget {
    /// 删除前核对project/item/take/GUID；删除后不再对已失效对象调用UpdateItem。
    pub(crate) fn delete_item(&self, authorized: &impl Fn() -> bool) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.delete_item(authorized))?;
        }
        self.verify(authorized)?;
        let api = self
            .host
            .media
            .as_ref()
            .ok_or("host item deletion API unavailable")?;
        if !checked(authorized, || unsafe {
            (api.delete_item)(self.track as *mut c_void, self.item as *mut c_void)
        })? {
            return Err("REAPER rejected item deletion".into());
        }
        let write = self.api()?.1;
        checked(authorized, || unsafe { (write.arrange)() })
    }
    /// 复制捕获真实item状态；宿主变化/对象租约失效即拒绝，不保存指针充当剪贴板。
    pub(crate) fn capture_item_state(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<String, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.capture_item_state(authorized))?;
        }
        self.verify(authorized)?;
        let api = self
            .host
            .item_state
            .as_ref()
            .ok_or("host item state API unavailable")?;
        let version = self.host.geometry_revision(authorized)?;
        self.verify(authorized)?;
        let text =
            read_complete_item_state(api, self.item as *mut c_void, 16 * 1024, false, authorized)?;
        self.verify(authorized)?;
        if self.host.geometry_revision(authorized)? != version {
            return Err("host item changed during clipboard capture".into());
        }
        Ok(text)
    }

    /// 冻结UI对象不保活item，写入前按GUID查验并重新读取当前几何。
    pub(crate) fn current(&self, authorized: &impl Fn() -> bool) -> Result<Self, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.current(authorized))?;
        }
        self.verify(authorized)?;
        let mut result = self.clone();
        result.geometry = self
            .host
            .geometry_for_take(self.take as *mut c_void, authorized)?;
        Ok(result)
    }
}

/// 一个真实宿主Undo块；关闭视图/错误也收尾。析构不在错误线程调用宿主API。
pub(crate) struct HostUndoBlock {
    host: Arc<NativeHost>,
    project: usize,
}
impl Drop for HostUndoBlock {
    fn drop(&mut self) {
        // 【不在宿主线程也必须把块关掉】`Undo_EndBlock2` 只能在宿主线程调用，而
        // `Drop` 可能在任意线程上跑（actor 线程、命令线程、错误路径）。旧实现遇到
        // 这种情形直接 `return` —— 一声不响。
        //
        // 后果不是"少一条撤销记录"，而是**块一直开着**：宿主的下一次 `Undo_DoUndo`
        // 会把之后所有编辑并成一步，用户按一次撤销就毁掉一大片；而自动渲染的
        // `host_undo.pending` 也永远为真，于是合成永久饥饿。
        //
        // 正确处置是**排进宿主线程的队列**，由那一边的下一次 tick 关掉。
        //
        // 【为什么不是就地 `on_host`】那会**阻塞**等宿主线程来取任务，而 `Drop` 可能
        // 正跑在宿主线程等待的那把锁上 —— 实测一个 workspace 用例因此永久挂起。
        // 排队是异步的：调用方立刻返回，块由宿主线程在下一次 tick 关掉。
        if std::thread::current().id() != self.host.thread {
            self.host.queue_undo_end(self.project);
            return;
        }
        self.host.end_project_undo(self.project);
    }
}

impl NativeHost {
    /// 可选写能力与只读几何独立；缺setter时GUI保持只读，不调用未查到的函数。
    pub(crate) fn can_edit_clips(&self) -> bool {
        self.write.is_some() && self.geometry.is_some()
    }
    /// 原生分割和取active take两项均存在，才向GUI声明分割能力。
    pub(crate) fn can_split_clips(&self) -> bool {
        self.can_edit_clips() && self.split.is_some()
    }
    pub(crate) fn has_project_history(&self) -> bool {
        self.history.is_some() && self.can_edit_clips()
    }
    /// take 的**播放方向**（倒放）能否写回。
    ///
    /// 【为什么不需要 `geometry`】倒放只改结构（SOURCE SECTION + SOFFS），与逐字段
    /// setter 无关。要求 `geometry` 会把一条本来可用的路径挡在门外（反之亦然）。
    ///
    /// 【为什么判据仍是 `item_state`】真正落地的是**两条**实现：chunk 整份改写，以及
    /// 宿主自己的 toggle 动作（见 `host_edit::write_take_direction`）。后者是前者被宿主
    /// 拒绝时的退路，而"现在到底是哪个方向"的核对（`PCM_Source_GetSectionInfo` 之外的第二
    /// 判据）与 item chunk 那条主路都在这一族 API 上 —— 缺它就没有可信的起点。
    pub(crate) fn can_reverse_clips(&self) -> bool {
        self.item_state.is_some()
    }
    /// take 的**音高包络**能否读写（REAPER 的 PITCHENV）。
    ///
    /// 【为什么与倒放同一条判据】写的那一步有更近的路（`GetTakeEnvelopeByName` +
    /// `SetEnvelopeStateChunk`，让写回绕开 chunk），但没有包络时"建一条"要借宿主动作，
    /// 而"到底有没有包络、有没有点"的判据、以及没有包络 API 时的整条退路，都建在 item
    /// state 上。缺 `Get/SetItemStateChunk` 时两条路一起不可用，有它时两条路一起可用 ——
    /// 拆成两位只会制造两个必然同值的布尔。
    pub(crate) fn can_bridge_take_pitch(&self) -> bool {
        self.can_edit_clips() && self.item_state.is_some()
    }
    /// 参数和几何共用所属project历史；没有item时也能为首次导入/参数编辑开启Undo。
    pub(crate) fn begin_project_undo(
        self: &Arc<Self>,
        authorized: &impl Fn() -> bool,
    ) -> Result<HostUndoBlock, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.begin_project_undo(authorized))?;
        }
        if !self.has_project_history() {
            return Err("REAPER project history API unavailable".into());
        }
        let project = self.project(authorized)?;
        if !checked(authorized, || unsafe {
            (self.geometry.as_ref().unwrap().validate)(project, project, c"ReaProject*".as_ptr())
        })? {
            return Err("invalid undo project".into());
        }
        let block = HostUndoBlock {
            host: self.clone(),
            project: project as usize,
        };
        checked(authorized, || unsafe {
            (self.write.as_ref().unwrap().begin)(project)
        })?;
        Ok(block)
    }
    /// 把一个"需要在宿主线程结束的 Undo 块"排队。
    ///
    /// 【为什么是队列而不是阻塞投递】见 [`HostUndoBlock::drop`]：阻塞投递会在
    /// "`Drop` 跑在宿主线程正等的那把锁上"时自锁。队列由宿主线程的下一次
    /// [`Self::drain_undo_ends`] 排空。
    pub(crate) fn queue_undo_end(&self, project: usize) {
        let mut queued = self.undo_ends.lock_or_recover();
        // 同一个项目只留一次：一个块只会结束一次，重复排队没有意义。
        if !queued.contains(&project) {
            queued.push(project);
        }
    }

    /// 在宿主线程上排空待结束的 Undo 块。**不在宿主线程时什么都不做**（留给下一次）。
    pub(crate) fn drain_undo_ends(&self) {
        if !self.on_host_thread() {
            return;
        }
        let queued = std::mem::take(&mut *self.undo_ends.lock_or_recover());
        for project in queued {
            self.end_project_undo(project);
        }
    }

    /// 结束一个已开启的宿主 Undo 块（**只允许在宿主线程调用**）。
    ///
    /// 抽成方法是为了让 [`HostUndoBlock::drop`] 在**别的线程**上也能把块关掉：
    /// 那时它把这次调用投递到宿主线程，而不是就地放弃（见那里的说明）。
    pub(crate) fn end_project_undo(&self, project: usize) {
        let Some(api) = &self.geometry else {
            return;
        };
        let Some(write) = &self.write else {
            return;
        };
        let project = project as *mut c_void;
        if !unsafe { (api.validate)(project, project, c"ReaProject*".as_ptr()) } {
            return;
        }
        unsafe {
            // 【为什么是品牌名而不是文案】后端不持有 UI 文案（见 `model.rs` 的同名
            // 约定）。这里写进的是**宿主自己的**撤销栈，宿主会把它显示在它自己的
            // 界面里；一个稳定的产品名比一句可能过期的英文描述更合适。前端只对
            // `backend === "local"` 的记录做 `history_op_` 本地化，宿主记录原样显示。
            (write.end)(project, c"HiFiShifter".as_ptr(), -1);
        }
        // `Undo_EndBlock2` 之后，刚创建的条目就是当前条目 —— 在这里记下它的墙钟时间。
        // 时间戳只是显示用的缓存，读不到就算了（不阻断任何写路径）。
        if let Some(history) = &self.history {
            let index = unsafe { (history.current)(project) };
            if index >= 0 {
                self.note_undo_entry_time(index);
            }
        }
    }
    /// 记下一个由本插件创建的 Undo 条目的墙钟时间（Unix 毫秒）。
    ///
    /// 【为什么先丢弃更大的索引】REAPER 的撤销栈在回退点截断：用户回退到索引 `i` 之后
    /// 再编辑，`i+1` 起的新条目会顶掉旧条目。丢掉所有大于 `index` 的记录，就不会让
    /// 旧时间错配到新条目上。
    ///
    /// 【容量】宿主条目数已有 10000 上限（见 [`Self::project_history`]）；本表只记本插件
    /// 自己创建的条目，规模受同样约束，并在 `close` 时随文档一起丢弃。
    pub(crate) fn note_undo_entry_time(&self, index: i32) {
        let mut times = self.undo_entry_times.lock_or_recover();
        times.retain(|known, _| *known < index);
        times.insert(index, hifishifter_kernel::state::now_unix_ms());
    }
    /// 官方Undo枚举提供真实深度和行，而不是用本地参数history冒充宿主历史。
    pub(crate) fn project_history(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<serde_json::Value, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.project_history(authorized))?;
        }
        let token = self.history_token(authorized)?;
        let project = self.project(authorized)?;
        let history = self.history.as_ref().ok_or("REAPER history unavailable")?;
        let read = |pointer: *const c_char| -> Result<Option<String>, String> {
            if pointer.is_null() {
                return Ok(None);
            }
            let value = unsafe { CStr::from_ptr(pointer) }
                .to_str()
                .map_err(|_| "host undo label is not UTF-8")?;
            if value.len() > 8192 {
                return Err("host undo label budget exceeded".into());
            }
            Ok(Some(value.to_owned()))
        };
        let position = checked(authorized, || unsafe { (history.current)(project) })?;
        let count = checked(authorized, || unsafe { (history.count)(project) })?;
        if !(0..=10000).contains(&count) {
            return Err("host undo entry budget exceeded".into());
        }
        let mut records = Vec::new();
        // 宿主越界描述可能返回非空指针的空字符串，不能靠null枚举结束。
        for index in 0..count {
            let pointer = checked(authorized, || unsafe { (history.entry)(project, index) })?;
            let label = read(pointer)?.ok_or("host undo changed while reading")?;
            // REAPER 不提供时间；只有**本插件创建**的条目有记录（见 `HostUndoBlock::drop`）。
            // 未知一律 `null`（前端隐藏时间列），绝不回落到 0 —— 那会显示 1970。
            let at = self
                .undo_entry_times
                .lock_or_recover()
                .get(&index)
                .copied()
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null);
            records.push(serde_json::json!({"label":label,"atMs":at}));
        }
        let position = if position < 0 && records.is_empty() {
            0
        } else {
            usize::try_from(position).map_err(|_| "invalid host undo index")?
        };
        if !records.is_empty() && position >= records.len() {
            return Err("host undo changed while reading".into());
        }
        if token != self.history_token(authorized)? {
            return Err("host undo changed while reading".into());
        }
        Ok(
            serde_json::json!({"ok":true,"backend":"host","position":position,"undoDepth":position,
            "redoDepth":records.len().saturating_sub(position+1),"records":records}),
        )
    }
    /// 轻量历史缓存键；UI timer不反复枚举整份Undo记录或广播相同状态。
    pub(crate) fn history_token(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<(i32, i32), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.history_token(authorized))?;
        }
        let change = self.geometry_revision(authorized)?;
        let project = self.project(authorized)?;
        let history = self.history.as_ref().ok_or("REAPER history unavailable")?;
        Ok((
            change,
            checked(authorized, || unsafe { (history.current)(project) })?,
        ))
    }
    /// 必须先收尾本应用的Undo块；Undo回流继续由真实ARA模型/组件setState接收。
    pub(crate) fn history_jump(
        &self,
        redo: bool,
        authorized: &impl Fn() -> bool,
    ) -> Result<bool, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.history_jump(redo, authorized))?;
        }
        self.history_token(authorized)?;
        let project = self.project(authorized)?;
        let history = self.history.as_ref().ok_or("REAPER history unavailable")?;
        let can = checked(authorized, || unsafe {
            if redo {
                (history.can_redo)(project)
            } else {
                (history.can_undo)(project)
            }
        })?;
        if can.is_null() {
            return Ok(false);
        }
        let done = checked(authorized, || unsafe {
            if redo {
                (history.redo)(project)
            } else {
                (history.undo)(project)
            }
        })?;
        Ok(done != 0)
    }
    /// 原操作记录窗口的跳转也使用真实project索引，不回放旧私有几何快照。
    pub(crate) fn history_jump_to(
        &self,
        target: i32,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.history_jump_to(target, authorized))?;
        }
        if !(0..10000).contains(&target) {
            return Err("host history position out of range".into());
        }
        let project = self.project(authorized)?;
        let history = self.history.as_ref().ok_or("REAPER history unavailable")?;
        if target >= checked(authorized, || unsafe { (history.count)(project) })? {
            return Err("host history position does not exist".into());
        }
        for _ in 0..10000 {
            let current = self.history_token(authorized)?.1;
            if current == target {
                return Ok(());
            }
            if !self.history_jump(current < target, authorized)? {
                return Err("host history jump stopped before target".into());
            }
        }
        Err("host history jump budget exceeded".into())
    }
    /// 冻结唯一直接take身份；调用方先验证该host对应真实assigned region。
    pub(crate) fn clip_target(
        self: &Arc<Self>,
        authorized: impl Fn() -> bool,
    ) -> Result<HostClipTarget, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.clip_target(authorized))?;
        }
        if !self.can_edit_clips() {
            return Err("REAPER clip write API unavailable".into());
        }
        let geometry = self.geometry(&authorized)?;
        let project = self.project(&authorized)?;
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        let take = checked(&authorized, || unsafe { (table.parent)(pointer, 2) })?;
        let api = self.geometry.as_ref().unwrap();
        let item = checked(&authorized, || unsafe { (api.item)(take) })?;
        let track = checked(&authorized, || unsafe {
            (self.write.as_ref().unwrap().item_track)(item)
        })?;
        let target = HostClipTarget {
            host: self.clone(),
            project: project as usize,
            item: item as usize,
            take: take as usize,
            track: track as usize,
            geometry,
        };
        target.verify(&authorized)?;
        Ok(target)
    }
}

/// 读一份 item state，直到拿到**一整份** chunk。
///
/// 【为什么"宿主说读成功"不等于"读完了"】`GetItemStateChunk` 能写多少由调用方给的
/// 缓冲区决定。宿主对"放不下"有两种表现：返回 `false`（调用方据此扩容），或者把前缀
/// 写满后**照常返回 true**。后者是这次 take 音高转换失败的根：一份 200 KB 的 item chunk
/// （长片段上千个 `PITCHENV` 点，或 take FX 里那坨插件状态）被当成"宿主给的 chunk 坏了"，
/// 于是每一次读写都在解析阶段失败 —— 而用户看到的只是"转换没反应"。
///
/// 判据只能是**结构**：能不能解析成一整份 `<ITEM>` 块（见 `take_chunk`）。截断必然
/// 留下未闭合的块，完好的 chunk 一定闭合 —— 这比再猜一个字节数可靠得多，也不会因为
/// 宿主换了写法（多一个自闭合叶子块）而误判。
///
/// 【为什么两种失败一起重试】把"`!ok`"也当成"可能太小"是唯一不依赖宿主文档的选择：
/// 真的读不出来时最多多读几次（最后一次仍在预算内），而误判成"坏了"会让一个完好的
/// item 永久写不进去。
fn read_complete_item_state(
    api: &super::item_chunk::ItemStateApi,
    item: *mut c_void,
    initial_capacity: usize,
    is_undo: bool,
    authorized: &(impl Fn() -> bool + ?Sized),
) -> Result<String, String> {
    let mut capacity = initial_capacity.max(1024);
    let mut observed: Option<String> = None;
    loop {
        let mut buffer = vec![0_u8; capacity];
        let ok = checked(authorized, || unsafe {
            (api.get)(item, buffer.as_mut_ptr().cast(), capacity as i32, is_undo)
        })?;
        if ok {
            // 宿主把 chunk 写成 NUL 结尾的 C 串；没有 NUL 就说明这一段没有边界。
            if let Some(end) = buffer.iter().position(|byte| *byte == 0) {
                if end > 0 {
                    if let Ok(text) = std::str::from_utf8(&buffer[..end]) {
                        // 【判据只有结构这一条】不引入"NUL 之前还有余量就算完整"这类启发式：
                        // 一旦成立就把完整性押在"宿主不会悄悄截断"上，而这次的故障正是它。
                        // 扫描是 O(字节)，读一次 chunk 本来就要整份解析；item chunk 的常见
                        // 尺寸是几十 KB，take FX 那份大的也只有**一个** clip 会走到这里。
                        if super::take_chunk::is_complete_item_state(text) {
                            return Ok(text.to_owned());
                        }
                        observed = Some(text.to_owned());
                    }
                }
            }
        }
        if capacity >= super::item_chunk::MAX_ITEM_CHUNK_BYTES {
            break;
        }
        capacity = (capacity * 2).min(super::item_chunk::MAX_ITEM_CHUNK_BYTES);
    }
    // 【失败时留下形状，而不是只留一句话】"读不出来"有三种完全不同的成因：宿主截断、
    // 宿主改了写法（多一种块）、指针早已失效。三者只能从"拿到了什么"里分开，而这条
    // 日志是用户下一次报障时唯一的一手证据。限流：同一个键只留一行。
    if let Some(text) = observed {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.item-chunk",
            format_args!(
                "[ara] the host item state is unusable: {}",
                super::take_chunk::describe_shape(&text)
            ),
        );
    }
    Err("host item state is not a complete item chunk within the supported budget".into())
}

impl HostClipTarget {
    /// 按已验证item在绝对项目秒域分割；返回两个真实GUID与几何，不猜新片段身份。
    pub(crate) fn split_at(
        &self,
        position: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<(HostClipGeometry, HostClipGeometry), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.split_at(position, authorized))?;
        }
        let current = self.current(authorized)?;
        let api = self
            .host
            .split
            .as_ref()
            .ok_or("REAPER split API unavailable")?;
        if !position.is_finite()
            || position <= current.geometry.start_sec + 1e-6
            || position >= current.geometry.start_sec + current.geometry.duration_sec - 1e-6
        {
            return Err("split position must be inside the item".into());
        }
        let right = checked(authorized, || unsafe {
            (api.split)(self.item as *mut c_void, position)
        })?;
        if right.is_null() {
            return Err("REAPER rejected item split".into());
        }
        current.valid(right as usize, c"MediaItem*", authorized)?;
        let take = self.split_right_take(right, position, authorized)?;
        let right_geometry = self.host.geometry_for_take(take, authorized)?;
        let right_target = HostClipTarget {
            host: self.host.clone(),
            project: self.project,
            item: right as usize,
            take: take as usize,
            track: self.track,
            geometry: right_geometry,
        };
        right_target.verify(authorized)?;
        current.update(authorized)?;
        right_target.update(authorized)?;
        Ok((
            current.current(authorized)?.geometry,
            right_target.current(authorized)?.geometry,
        ))
    }
    /// 分割后右半段的 take：按"同一序号 + 源起点连续"解析。
    ///
    /// 【为什么不能直接取 active take】`SplitMediaItem` 之后 `GetActiveTake(right)`
    /// 给的是**宿主选中的**那条 take。take FX 恰好可以挂在非 active 的 take 上 ——
    /// 那时旧代码会把右半段的写回目标指到别人的 take，用户看到的是"分割后音高/方向
    /// 落到了另一条 take 上"。
    ///
    /// 【判据】REAPER 在分割出的两个 item 上保持 take 的**顺序**，因此"本条 take 在原
    /// item 里的序号"是最强的身份证据；再用源起点连续性（右 = 左 + 位移 × 播放速率）
    /// 复核一次。序号对得上、连续性也对得上才写；否则拒绝（宿主 Undo 仍然可用），
    /// 因为"猜一个 take"等于改错别人的音频。
    fn split_right_take(
        &self,
        right: *mut c_void,
        position: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<*mut c_void, String> {
        let left = self.current(authorized)?;
        let delta = position - left.geometry.start_sec;
        let Some(api) = self.host.take_enum() else {
            return self.single_take_after_split(right, authorized);
        };
        let count = checked(authorized, || unsafe {
            (api.count)(left.item as *mut c_void)
        })?;
        if !(0..=1024).contains(&count) {
            return Err("split item has an implausible take count".into());
        }
        let mut index = None;
        for slot in 0..count {
            let candidate = checked(authorized, || unsafe {
                (api.at)(left.item as *mut c_void, slot)
            })?;
            if candidate as usize == left.take {
                index = Some(slot);
                break;
            }
        }
        let Some(index) = index else {
            return Err("split item no longer holds the bound take".into());
        };
        let candidate = checked(authorized, || unsafe { (api.at)(right, index) })?;
        if candidate.is_null() {
            return Err("split item has no take at the bound index".into());
        }
        // 源起点连续性：右侧的 SOFFS 必须等于"左侧 SOFFS + 位移 × 播放速率"。
        // 这条读数由宿主自己算出来，改错了一条 take 就会对不上。
        let rate = if left.geometry.play_rate_signed.is_finite() {
            left.geometry.play_rate_signed
        } else {
            1.0
        };
        let expected = left.geometry.source_start_sec + delta * rate;
        let actual = self.read_take_start_offs(candidate, authorized)?;
        if (actual - expected).abs() > 1e-3 {
            return Err("split take sequence does not match the bound take".into());
        }
        Ok(candidate)
    }

    /// 读一条 take 的 `D_STARTOFFS`。
    fn read_take_start_offs(
        &self,
        take: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> Result<f64, String> {
        let geometry = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let value = checked(authorized, || unsafe {
            (geometry.take_value)(take, c"D_STARTOFFS".as_ptr())
        })?;
        if !value.is_finite() {
            return Err("host take start offset is not finite".into());
        }
        Ok(value)
    }

    /// 宿主没有多 take 枚举接口时：退回改动前的行为（取 active take）。
    ///
    /// 【为什么这里诚实退化而不是拒绝】没有 `GetMediaItemNumTakes` 的宿主上，插件连
    /// "一个 item 有几条 take"都看不到 —— 多 take 的 take FX 在那样的宿主上本来也
    /// 绑不出正确结果。此时保留旧行为，等于把改动的影响面限制在"宿主确实提供了枚举
    /// 接口"的那一档。
    fn single_take_after_split(
        &self,
        right: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> Result<*mut c_void, String> {
        let api = self
            .host
            .split
            .as_ref()
            .ok_or("REAPER split API unavailable")?;
        let take = checked(authorized, || unsafe { (api.active_take)(right) })?;
        if take.is_null() {
            return Err("split item has no active take; host Undo remains available".into());
        }
        Ok(take)
    }

    /// 同批所有对象须来自同一project，不以“当前活动项目”或相同轨名推断。
    pub(crate) fn same_project(&self, other: &Self) -> bool {
        self.project == other.project
    }
    /// 这个 target 所在的宿主能不能写回 take 方向。
    ///
    /// 【为什么委托给 host】能力位是**宿主级**的（`Get/SetItemStateChunk` 存不存在），
    /// 不是 item 级的；target 只是"这个宿主上的一个对象"。
    pub(crate) fn can_reverse_clips(&self) -> bool {
        self.host.can_reverse_clips()
    }
    /// 是不是同一个宿主 item（回滚时用来避免同一对象被撤两次）。
    pub(crate) fn same_item(&self, other: &Self) -> bool {
        self.project == other.project && self.item == other.item
    }
    pub(crate) fn import_target(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<super::HostTrackTarget, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.import_target(authorized))?;
        }
        self.verify(authorized)?;
        self.host.track_target(self.project, self.track, authorized)
    }
    /// 写能力束。
    ///
    /// 【为什么返回 `Result` 而不是 `unwrap`】`geometry` 与 `write` 是**两个独立的
    /// `Option`**（`can_edit_clips` 要求两者同时存在，但构造/校验路径不止一条）。
    /// 这里跑在宿主进程里，一次 panic 会带走整个 DAW 会话 —— 缺能力时如实报错。
    fn api(&self) -> Result<(&super::GeometryApi, &WriteApi), String> {
        Ok((
            self.host
                .geometry
                .as_ref()
                .ok_or("REAPER geometry API unavailable")?,
            self.host
                .write
                .as_ref()
                .ok_or("REAPER write API unavailable")?,
        ))
    }
    fn valid(
        &self,
        pointer: usize,
        kind: &CStr,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if pointer == 0 {
            return Err("missing host edit object".into());
        }
        let api = self.api()?;
        if !checked(authorized, || unsafe {
            (api.0.validate)(
                self.project as *mut c_void,
                pointer as *mut c_void,
                kind.as_ptr(),
            )
        })? {
            return Err("host edit object no longer belongs to project".into());
        }
        Ok(())
    }
    fn guid(
        &self,
        pointer: usize,
        getter: Guid,
        expected: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if expected.len() != 38 {
            return Err("invalid captured host GUID".into());
        }
        let mut bytes = [0_u8; 64];
        if !checked(authorized, || unsafe {
            getter(
                pointer as *mut c_void,
                c"GUID".as_ptr(),
                bytes.as_mut_ptr().cast(),
                false,
            )
        })? || bytes[..expected.len()] != *expected.as_bytes()
            || bytes[expected.len()] != 0
        {
            return Err("host item/take identity changed before write".into());
        }
        Ok(())
    }
    /// 即使地址被重用也不能写新对象。自身setter造成ARA模型代次推进不等同租约失效。
    pub(crate) fn verify(&self, authorized: &impl Fn() -> bool) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.verify(authorized))?;
        }
        if std::thread::current().id() != self.host.thread {
            return Err("host edit off UI thread".into());
        }
        self.valid(self.project, c"ReaProject*", authorized)?;
        self.valid(self.item, c"MediaItem*", authorized)?;
        self.valid(self.take, c"MediaItem_Take*", authorized)?;
        let api = self.api()?.0;
        if checked(authorized, || unsafe {
            (api.item)(self.take as *mut c_void)
        })? as usize
            != self.item
        {
            return Err("host take parent changed".into());
        }
        self.guid(self.item, api.item_guid, &self.geometry.item_id, authorized)?;
        self.guid(self.take, api.take_guid, &self.geometry.take_id, authorized)
    }
    /// 批次开始前调用；拒绝跨工程/失效对象，不在document锁内调用Undo API。
    pub(crate) fn begin_undo(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<HostUndoBlock, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.begin_undo(authorized))?;
        }
        self.verify(authorized)?;
        let block = HostUndoBlock {
            host: self.host.clone(),
            project: self.project,
        };
        let api = self.api()?;
        checked(authorized, || unsafe {
            (api.1.begin)(self.project as *mut c_void)
        })?;
        Ok(block)
    }
    /// 仅明确白名单的数值字段可写；重入撤销租约后停止后续setter。
    pub(crate) fn set_item(
        &self,
        name: &CStr,
        value: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.set_item(name, value, authorized))?;
        }
        self.verify(authorized)?;
        if !matches!(
            name.to_bytes(),
            b"D_POSITION"
                | b"D_LENGTH"
                | b"D_VOL"
                | b"B_MUTE"
                | b"B_LOOPSRC"
                | b"I_GROUPID"
                | b"D_SNAPOFFSET"
                | b"D_FADEINLEN"
                | b"D_FADEOUTLEN"
                | b"D_FADEINLEN_AUTO"
                | b"D_FADEOUTLEN_AUTO"
                | b"C_FADEINSHAPE"
                | b"C_FADEOUTSHAPE"
                | b"D_FADEINDIR"
                | b"D_FADEOUTDIR"
                | b"D_FADEINDIR_NEW"
                | b"D_FADEOUTDIR_NEW"
                | b"D_FADEINDIR2_NEW"
                | b"D_FADEOUTDIR2_NEW"
        ) {
            return Err("unsupported host item field".into());
        }
        if !value.is_finite() {
            return Err("nonfinite host item edit".into());
        }
        if name == c"D_VOL" && value < 0. {
            return Err("negative item volume is not supported".into());
        }
        let api = self.api()?;
        if !checked(authorized, || unsafe {
            (api.1.set_item)(self.item as *mut c_void, name.as_ptr(), value)
        })? {
            return Err(format!(
                "REAPER rejected item field {}",
                name.to_string_lossy()
            ));
        }
        Ok(())
    }
    pub(crate) fn set_take(
        &self,
        name: &CStr,
        value: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.set_take(name, value, authorized))?;
        }
        self.verify(authorized)?;
        if !matches!(
            name.to_bytes(),
            b"D_STARTOFFS" | b"D_PLAYRATE" | b"B_PPITCH" | b"I_CHANMODE" | b"D_PITCH"
        ) {
            return Err("unsupported host take field".into());
        }
        if !value.is_finite() {
            return Err("nonfinite host take edit".into());
        }
        // 【为什么要给 D_PITCH 设上界】音高迁移要把宿主侧清零；而清零是一个**写**。
        // 没有上界时一次误算（例如把"音分"当半音）会把 take 推到几十个八度之外 ——
        // 用户听到的是噪声，看不出是一个单位错误。±240 半音 = ±20 八度，
        // 远超任何音乐用途，又足以拦住量级错误。
        if name == c"D_PITCH" && value.abs() > 240. {
            return Err("take pitch outside the supported range -240..240 semitones".into());
        }
        let api = self.api()?;
        if !checked(authorized, || unsafe {
            (api.1.set_take)(self.take as *mut c_void, name.as_ptr(), value)
        })? {
            return Err(format!(
                "REAPER rejected take field {}",
                name.to_string_lossy()
            ));
        }
        Ok(())
    }
    /// 读取该 item 的完整 state chunk（写回结构前必须先在宿主线程拿到原文）。
    ///
    /// 【为什么在 target 上而不是直接拿 host 的 API】item 指针是冻结并校验过的
    /// （`verify` 会核对 GUID）；绕开它去拿"当前选中 item"的 chunk 会写错对象。
    pub(crate) fn item_state_chunk(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<String, String> {
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.item_state_chunk(authorized))?;
        }
        self.verify(authorized)?;
        let api = self
            .host
            .item_state
            .as_ref()
            .ok_or("REAPER item state API unavailable")?;
        read_complete_item_state(api, self.item as *mut c_void, 64 * 1024, true, authorized)
    }
    /// 把改写后的 state chunk 交回宿主。
    ///
    /// 与 `item_state_chunk` 成对出现：调用方必须先读原文、只在**确实要改**时才写，
    /// 因为 `SetItemStateChunk` 会让宿主重建这个 item（选择态、undo 点都会受影响）。
    pub(crate) fn set_item_state_chunk(
        &self,
        chunk: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.set_item_state_chunk(chunk, authorized))?;
        }
        self.verify(authorized)?;
        if chunk.is_empty()
            || chunk.len() > super::item_chunk::MAX_ITEM_CHUNK_BYTES
            || chunk.contains('\0')
        {
            return Err("invalid item state to write".into());
        }
        let api = self
            .host
            .item_state
            .as_ref()
            .ok_or("REAPER item state API unavailable")?;
        let text = std::ffi::CString::new(chunk).map_err(|_| "item state contains NUL")?;
        if !checked(authorized, || unsafe {
            (api.set)(self.item as *mut c_void, text.as_ptr(), true)
        })? {
            return Err(ITEM_STATE_REFUSED.into());
        }
        Ok(())
    }
    /// 读该 take 的音高包络。
    ///
    /// 【为什么优先读包络对象】take FX 放置下，要读的那个 item 恰好就是挂着本插件的 item，
    /// 而它的 item chunk 里塞着整个插件状态（真机工程里是几百 KB）。包络自己的 chunk 只有
    /// 几十个点、几十字节 —— 而且这条读数在"item chunk 写回被宿主拒绝"时仍然可用。
    pub(crate) fn take_pitch_points(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<super::take_pitch::TakePitch, String> {
        if let Some(envelope) = self
            .host
            .take_pitch_envelope(self.take as *mut c_void, authorized)?
        {
            let text = self.host.envelope_state_chunk(envelope, authorized)?;
            return super::take_pitch::read_block(&text);
        }
        let chunk = self.item_state_chunk(authorized)?;
        super::take_pitch::read_take_pitch(&chunk, &self.geometry.take_id)
    }
    /// 用 `points` 整块替换该 take 的音高包络（空列表 = 删除）。
    ///
    /// 【三条路径，按"最不打扰宿主"排序】
    /// 1. **包络对象已存在** ⇒ `SetEnvelopeStateChunk`。真机实测这条路在 take FX（本插件
    ///    在线、item chunk 写回被拒）时仍然成功，而且不重建 item —— 轨道 FX 下也更快、
    ///    更安全（不会顺手重建同一条 item 上的别的 take FX）。
    /// 2. **没有包络** ⇒ 借宿主自己的动作建一条（[`super::host_action`]）再走 1。
    ///    两种放置通用，同样不重建 item。
    /// 3. 动作不可用（老宿主/精简导出）⇒ 回到 item chunk 的逐字节路径。这条是历史行为，
    ///    也是唯一不依赖宿主动作的"创建"，因此必须留着。
    ///
    /// 【为什么不再"先试 chunk、失败再建包络"】那条顺序在 take FX 下每次都要读一份几百 KB
    /// 的 chunk 再吃一次必然的拒绝（宿主不允许重建带在线 ARA FX 的 item），最后才走包络 ——
    /// 用户看到的是几秒钟的卡顿加一条吓人的日志。现在失败的那条路只在最后兜底。
    pub(crate) fn write_take_pitch_points(
        &self,
        points: &[hifishifter_kernel::pitch_bridge::EnvelopePoint],
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        let take = self.take as *mut c_void;
        if let Some(envelope) = self.host.take_pitch_envelope(take, authorized)? {
            return self.host.set_envelope_state_chunk(
                envelope,
                &super::take_pitch::block_text(points),
                authorized,
            );
        }
        if points.is_empty() {
            // 没有包络就没有可删的东西：连 item chunk 都不必读。
            return Ok(());
        }
        let created =
            self.host
                .create_take_pitch_envelope(self.item as *mut c_void, take, authorized);
        match created {
            Ok(envelope) => self.host.set_envelope_state_chunk(
                envelope,
                &super::take_pitch::block_text(points),
                authorized,
            ),
            Err(action_error) => {
                let chunk = self.item_state_chunk(authorized).and_then(|chunk| {
                    let rewritten = super::take_pitch::rewrite_take_pitch(
                        &chunk,
                        &self.geometry.take_id,
                        points,
                    )?;
                    if rewritten == chunk {
                        return Ok(());
                    }
                    self.set_item_state_chunk(&rewritten, authorized)
                });
                chunk.map_err(|chunk_error| format!("{action_error}; {chunk_error}"))
            }
        }
    }
    /// 宿主此刻报告的方向位（`PCM_Source_GetSectionInfo`）。
    ///
    /// 写回之后必须用它做一次"写→读→比"自校验：SECTION MODE 的取值属于宿主，
    /// 仓库里没有第二份证据能替代这次实测。
    pub(crate) fn reversed_now(&self, authorized: &impl Fn() -> bool) -> Result<bool, String> {
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.reversed_now(authorized))?;
        }
        self.verify(authorized)?;
        // 【读不到方向位 = 正放，不是错误】`PCM_Source_GetSectionInfo` 对**没有
        // SECTION/REVERSE 包装**的 take 返回 false —— 那是"从未倒放过"的正常形态，
        // 与几何读取侧（`project_host_take_facts_locked` 对 `None` 按正放处理）是
        // 同一条纪律。把它当错误会让 take FX 的动作路径在第一步（读 `before`）就
        // 整批失败：真机日志 "the host did not report the take direction" 正是它，
        // 而轨道 FX 走 chunk 路径没有这次预读，所以"轨道正常、take FX 被拒"。
        //
        // 方向判定与读取侧同一口径：方向位（若有）**或**负速率（REAPER 表达倒放的
        // 另一个来源，见 `HostClipGeometry::play_rate_signed`）。
        if let Some(facts) = self
            .host
            .take_direction_facts(self.take as *mut c_void, authorized)
        {
            return Ok(facts.reversed);
        }
        let geometry = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let rate = checked(authorized, || unsafe {
            (geometry.take_value)(self.take as *mut c_void, c"D_PLAYRATE".as_ptr())
        })?;
        Ok(rate.is_finite() && rate < 0.0)
    }
    /// 这个 item 是不是**本实例自己挂着的那个**（take FX 下必然如此）。
    ///
    /// 【为什么它就是"item chunk 写不进去"的判据】真机取证（REAPER 7.85，见
    /// [`super::take_env`] 的模块文档）：一个带着**在线 ARA FX** 的 item，
    /// `SetItemStateChunk` 连"原样写回"都拒绝。take FX 放置下我们要改方向/包络的那个
    /// item 正是挂着本插件的那个 —— 判据来自放置位置缓存（`placement().item()`），
    /// 不需要再往宿主跑一趟，也不会把"其它 ARA 实例的 item"误判成自己的。
    ///
    /// 【读的是缓存】`Unbound`（还没求值）时返回 `false`：那就照历史路径先写 chunk，
    /// 真被拒绝时调用方还有动作那条退路 —— 两条判据叠起来不会漏。
    pub(crate) fn hosts_this_instance(&self) -> bool {
        self.host.placement().item() == Some(self.item)
    }

    /// 借宿主自己的动作把这条 take 的**播放方向**切到 `reversed`（已经是它就不动）。
    ///
    /// 【为什么不自己写结构】倒放没有 setter：它是 `<SOURCE SECTION MODE 3>` 这层包装，
    /// 而 chunk 那条路在 take FX 下被宿主拒绝。宿主的 `Item properties: Toggle take
    /// reverse` 写出的结构与我们的编码逐字节一致（真机对照过，见
    /// [`super::source_direction`]），于是"借它的动作"与"写同一份结构"等价。
    ///
    /// 【为什么先比再动】动作是 toggle：已经是对的方向时再跑一次会把用户刚改的倒放翻回去。
    pub(crate) fn apply_take_direction(
        &self,
        reversed: bool,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if self.reversed_now(authorized)? == reversed {
            return Ok(());
        }
        self.host.run_item_action(
            self.item as *mut c_void,
            self.take as *mut c_void,
            &super::host_action::TAKE_REVERSE,
            authorized,
        )
    }

    /// 按 take GUID 在**同一 item**上解析另一条 take 的指针（逐 take 写回用）。
    ///
    /// 【为什么在写回时才解析】UI 清单的 `UiItem.takes` 只保存几何、不保存指针；
    /// 而且 take 指针在两次清单刷新之间可能已被宿主重建。写回时刻逐槽枚举、按
    /// GUID 认领 —— 与 `verify` 对 item/take 的"GUID 相等才算同一个对象"同一条
    /// 纪律。找不到唯一匹配就报错：猜错 take 是工程损坏。
    pub(crate) fn take_pointer_by_guid(
        &self,
        guid: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<usize, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            let guid = guid.to_owned();
            return self
                .host
                .on_host(move || self.take_pointer_by_guid(&guid, authorized))?;
        }
        self.valid(self.project, c"ReaProject*", authorized)?;
        self.valid(self.item, c"MediaItem*", authorized)?;
        let enum_api = self
            .host
            .take_enum()
            .ok_or("REAPER take enumeration API unavailable")?;
        let count = checked(authorized, || unsafe {
            (enum_api.count)(self.item as *mut c_void)
        })?;
        if !(0..=super::ui_inventory::MAX_TAKES).contains(&count) {
            return Err("take inventory budget exceeded".into());
        }
        let geometry = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        if guid.len() != 38 {
            return Err("invalid captured host take GUID".into());
        }
        for slot in 0..count {
            let take = checked(authorized, || unsafe {
                (enum_api.at)(self.item as *mut c_void, slot)
            })?;
            if take.is_null() {
                continue;
            }
            let mut bytes = [0_u8; 64];
            let matched = checked(authorized, || unsafe {
                (geometry.take_guid)(
                    take,
                    c"GUID".as_ptr(),
                    bytes.as_mut_ptr().cast(),
                    false,
                )
            })? && bytes[..guid.len()] == *guid.as_bytes()
                && bytes[guid.len()] == 0;
            if matched {
                return Ok(take as usize);
            }
        }
        Err("the requested take is gone from the host item".to_string())
    }

    /// 对**任意一条 take**（按指针）写白名单数值字段。
    ///
    /// 与 [`Self::set_take`] 的差别只有目标：那是绑定 take，这一条服务逐 take 写回
    /// （声道模式落在用户点的那一条上，而不是永远落在 active take）。字段白名单
    /// 两侧同源，指针在使用前由 [`Self::take_pointer_by_guid`] 按 GUID 现场认领。
    pub(crate) fn set_take_pointer_value(
        &self,
        take: usize,
        name: &CStr,
        value: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            let name = name.to_owned();
            return self.host.on_host(move || {
                self.set_take_pointer_value(take, &name, value, authorized)
            })?;
        }
        self.valid(self.project, c"ReaProject*", authorized)?;
        self.valid(self.item, c"MediaItem*", authorized)?;
        self.valid(take, c"MediaItem_Take*", authorized)?;
        if !matches!(name.to_bytes(), b"I_CHANMODE" | b"D_STARTOFFS") {
            return Err("unsupported host take field".into());
        }
        if !value.is_finite() {
            return Err("nonfinite host take edit".into());
        }
        let api = self.api()?;
        if !checked(authorized, || unsafe {
            (api.1.set_take)(take as *mut c_void, name.as_ptr(), value)
        })? {
            return Err(format!(
                "REAPER rejected take field {}",
                name.to_string_lossy()
            ));
        }
        Ok(())
    }

    /// 读**任意一条 take**此刻的方向位（逐 take 倒放的自校验用）。
    ///
    /// 与 [`Self::reversed_now`] 同一口径：方向位（若有）或负速率；读不到方向位
    /// 按正放处理（"从未倒放过"是正常形态，不是错误）。
    pub(crate) fn take_reversed_now(
        &self,
        take: usize,
        authorized: &impl Fn() -> bool,
    ) -> Result<bool, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(move || self.take_reversed_now(take, authorized))?;
        }
        self.valid(take, c"MediaItem_Take*", authorized)?;
        if let Some(facts) = self.host.take_direction_facts(take as *mut c_void, authorized) {
            return Ok(facts.reversed);
        }
        let geometry = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let rate = checked(authorized, || unsafe {
            (geometry.take_value)(take as *mut c_void, c"D_PLAYRATE".as_ptr())
        })?;
        Ok(rate.is_finite() && rate < 0.0)
    }

    /// 借宿主自己的动作把**任意一条 take**的方向切到 `reversed`（已经是它就不动）。
    ///
    /// [`Self::apply_take_direction`] 只作用于绑定 take；逐 take 倒放的目标是兄弟
    /// take —— 宿主动作作用于 item 的 active take，所以入口先把 `I_CURTAKE` 切到
    /// 目标、跑完切回来（[`super::host_action::NativeHost::run_item_action`] 的
    /// 既有纪律，音高包络创建走的就是同一条路）。
    pub(crate) fn apply_take_direction_on(
        &self,
        take: usize,
        reversed: bool,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if self.take_reversed_now(take, authorized)? == reversed {
            return Ok(());
        }
        self.host.run_item_action(
            self.item as *mut c_void,
            take as *mut c_void,
            &super::host_action::TAKE_REVERSE,
            authorized,
        )
    }

    /// 宿主几何 API 读数束（兄弟 take 的 SOFFS/LENGTH/PLAYRATE/I_CHANMODE 读用）。
    ///
    /// 【为什么不是 `api()`】`api()` 还要求写 API 存在；兄弟 take 的**读数**在
    /// 只读宿主上也成立，不该被写能力门挡住。
    pub(crate) fn host_geometry_api(
        &self,
    ) -> Option<&super::GeometryApi> {
        self.host.geometry.as_ref()
    }

    /// 读任意一条 take 的方向事实（`PCM_Source_GetSectionInfo`）。
    pub(crate) fn host_take_direction_facts(
        &self,
        take: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> Option<super::TakeDirectionFacts> {
        self.host.take_direction_facts(take, authorized)
    }

    /// 读任意一条 take 的媒体总时长（`GetMediaSourceLength`）。
    pub(crate) fn host_take_source_length(
        &self,
        take: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> Option<f64> {
        self.host.take_source_length(take, authorized)
    }

    /// 绑定 take 的指针（回滚要区分"这条写的是绑定 take 还是兄弟 take"）。
    pub(crate) fn bound_take(&self) -> usize {
        self.take
    }

    /// 宿主是否具备 take 字段写能力（逐 take 声道模式预检用；语义同
    /// `NativeHost::can_edit_clips`）。
    pub(crate) fn host_can_write_take_fields(&self) -> bool {
        self.host.can_edit_clips()
    }

    /// 只重命名当前真实active take；字符串通过REAPER官方setter写入，不改HFS孤立名称。
    pub(crate) fn set_take_name(
        &self,
        name: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.set_take_name(name, authorized))?;
        }
        self.verify(authorized)?;
        if name.len() > 8192 || name.contains('\0') {
            return Err("take name budget or NUL rejected".into());
        }
        let mut bytes = std::ffi::CString::new(name)
            .map_err(|_| "take name contains NUL")?
            .into_bytes_with_nul();
        let api = self.api()?;
        if !checked(authorized, || unsafe {
            (api.1.set_take_string)(
                self.take as *mut c_void,
                c"P_NAME".as_ptr(),
                bytes.as_mut_ptr().cast(),
                true,
            )
        })? {
            return Err("REAPER rejected take name".into());
        }
        Ok(())
    }
    /// 目标轨道由另一真实region的直接绑定证明，不接受任意JS指针/轨道编号。
    pub(crate) fn move_to(
        &self,
        destination: &Self,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.move_to(destination, authorized))?;
        }
        self.verify(authorized)?;
        destination.verify(authorized)?;
        if !self.same_project(destination) {
            return Err("cannot move item across host projects".into());
        }
        self.valid(destination.track, c"MediaTrack*", authorized)?;
        let api = self.api()?;
        if !checked(authorized, || unsafe {
            (api.1.move_item)(self.item as *mut c_void, destination.track as *mut c_void)
        })? {
            return Err("REAPER rejected item track move".into());
        }
        Ok(())
    }
    /// 触发宿主刷新和真实ARA回流，不直接篡改插件timeline几何。
    pub(crate) fn update(&self, authorized: &impl Fn() -> bool) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.update(authorized))?;
        }
        self.verify(authorized)?;
        let api = self.api()?;
        checked(authorized, || unsafe {
            (api.1.update)(self.item as *mut c_void)
        })?;
        checked(authorized, || unsafe { (api.1.arrange)() })
    }
}
