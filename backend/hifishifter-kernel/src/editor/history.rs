//! 原线性撤销历史的共享操作；状态惰性补齐与redo分支语义不因运行模式改变。
use crate::state::*;

/// 登记操作前状态，截断redo分支并保留现有100步上限；notes只在旧实现需要时读取。
pub fn checkpoint(
    h: &mut TimelineHistory,
    snapshot: &TimelineState,
    label: String,
    mut notes: impl FnMut() -> Option<String>,
) {
    let now = now_unix_ms();
    if h.started_at_ms == 0 {
        h.started_at_ms = now;
    }
    let started_at_ms = h.started_at_ms;
    if h.records.is_empty() {
        // 初始状态行：此刻的实时状态就是第 0 个状态（工程打开 /
        // 新建之后、第一次编辑之前）。
        h.records.push(HistoryRecord {
            label: None,
            at_ms: started_at_ms,
            state: Some(snapshot.clone()),
            notes_markdown: notes(),
            param_selection: None,
        });
        h.position = 0;
    } else {
        let position = h.position;
        // 新操作丢弃当前位置之后的分支（旧 redo 栈）。
        h.records.truncate(position + 1);
        // 当前位置的记录是占位（None）：用实时时间线补齐 —— 它此刻
        // 正是这个状态本身，之后跳回该位置要靠这份快照。
        //
        // 记事本同理：本步**离开**的那个状态，其记事本内容就是即将
        // 被这次操作改掉之前的值。显式取 `notes_markdown` 入参（记事本
        // 编辑路径传入"编辑前"的文本）而不依赖"调用方还没写新值"的
        // 时序 —— 后者一旦被重构打乱，撤销就会静默丢内容。
        //
        // 选区快照（`param_selection`）**必须清空**：位置被复用意味着
        // 这里要放的是**新**一步，而旧分支那一步的选区快照不属于它 ——
        // 留着会让撤销这步时把选区跳到一条已被丢弃的分支上。
        if let Some(current) = h.records.get_mut(position) {
            current.state = Some(snapshot.clone());
            if current.notes_markdown.is_none() {
                current.notes_markdown = notes();
            }
            current.param_selection = None;
        }
    }
    // 追加这一步：label/at 描述紧随其后的操作，快照留待下一次
    // 打点或跳转时补齐（届时它就是实时时间线本身）。
    h.records.push(HistoryRecord {
        label: Some(label),
        at_ms: now,
        state: None,
        notes_markdown: None,
        param_selection: None,
    });
    h.position = h.records.len() - 1;
    // 上限：丢最旧的状态（当前位置随之左移）。
    while h.records.len() > MAX_UNDO_HISTORY + 1 {
        h.records.remove(0);
        h.position = h.position.saturating_sub(1);
    }
}

/// 历史跳转的纯状态部分；窗口、设备、dirty等副作用由各宿主执行。
pub fn jump(
    h: &mut TimelineHistory,
    current: &TimelineState,
    target: usize,
    intent: HistoryJumpIntent,
    notes: Option<String>,
) -> Option<(TimelineState, Option<String>, Option<Vec<[f32; 2]>>)> {
    if target >= h.records.len() || target == h.position {
        return None;
    }
    let position = h.position;
    if let Some(record) = h.records.get_mut(position) {
        record.state = Some(current.clone());
        if record.notes_markdown.is_none() {
            record.notes_markdown = notes;
        }
    }
    let next = h.records.get(target)?.state.clone()?;
    h.position = target;
    let notes = h.records.get(target).and_then(|r| r.notes_markdown.clone());
    let selection = param_selection_restore_for(h, target, intent);
    Some((next, notes, selection))
}
