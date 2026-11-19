//! 原生QI/vtable/raw函数回归；fixture不是实际REAPER parent/API返回的验收证据。
use super::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

thread_local! {static ACTIVE:Cell<usize>=const {Cell::new(0)};}

/// 假宿主里的一条 FX 链项（名字 / 启用 / 离线）。
///
/// 【为什么用 `&'static str` 而不是 `String`】夹具是 `Cell`/`RefCell` 里的**常量**
/// 数据，用例在断言里写的名字必须与宿主读到的一字不差；静态串让"名字写错"变成编译期
/// 问题，而不是一条永远匹配不上的运行时谜题。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FxEntry {
    pub name: &'static str,
    pub enabled: bool,
    pub offline: bool,
}

impl FxEntry {
    /// 本插件的 FX（类名匹配 `HiFiShifter`）。
    pub const HFS: Self = Self {
        name: "VST3: HiFiShifter",
        enabled: true,
        offline: false,
    };
    /// 一个与本插件无关的原生 FX。
    pub const OTHER: Self = Self {
        name: "VST3: ReaEQ",
        enabled: true,
        offline: false,
    };
}

#[repr(C)]
pub struct Fixture {
    table: *const HostVtbl,
    refs: AtomicU32,
    take_token: u8,
    item_token: u8,
    track_token: u8,
    right_item_token: u8,
    right_take_token: u8,
    split_enabled: Cell<bool>,
    right_values: RefCell<Option<BTreeMap<&'static str, f64>>>,
    writer_enabled: Cell<bool>,
    take_volume: Cell<f64>,
    pub inventory_enabled: Cell<bool>,
    pub inventory_empty: Cell<bool>,
    /// 多 take 枚举接口（`GetMediaItemNumTakes` / `GetMediaItemTake` 等）是否注册。
    /// 默认关 —— 缺省下清单只呈现 active take，与本次改动之前完全一致。
    pub takes_enabled: Cell<bool>,
    /// 覆盖 `GetMediaItemNumTakes` 的返回值（越界/畸形预算用）。
    pub take_count_override: Cell<Option<i32>>,
    /// 第 2..N 个 take 的稳定地址（第 1 个复用 `take_token`）。
    pub extra_takes: RefCell<Vec<Box<u8>>>,
    /// `PCM_Source_GetSectionInfo` 报告的方向；配合 `section_reports` 使用。
    pub take_reversed: Cell<bool>,
    /// `PCM_Source_GetSectionInfo` 是否返回成功（false = 读不出方向 → `None`）。
    pub section_reports: Cell<bool>,
    /// `PCM_Source_GetSectionInfo` 报告的段边界（`SetItemStateChunk` 按 `MODE` 位
    /// 更新，与 REAPER 同一条规则：`MODE 3` 报整份媒体、`MODE 0/2` 报
    /// `STARTPOS`/`LENGTH`、非段块报 0/0）。
    pub take_section_start: Cell<f64>,
    pub take_section_length: Cell<f64>,
    /// `GetMediaSourceFileName` 报告的路径；`None` = 读不出文件名（回退身份不可用）。
    pub source_file_name: RefCell<Option<String>>,
    undo_records: RefCell<Vec<std::ffi::CString>>,
    undo_position: Cell<i32>,
    media_enabled: Cell<bool>,
    media_sources: RefCell<Vec<Box<u8>>>,
    take_source: Cell<usize>,
    fail_media_take: Cell<bool>,
    media_path: RefCell<String>,
    extra_tracks: RefCell<Vec<Box<u8>>>,
    new_fx: RefCell<std::collections::HashSet<usize>>,
    fx_guid_bytes: [u8; 16],
    picker_paths: RefCell<Vec<String>>,
    pub valid: Cell<bool>,
    connection_host: Cell<bool>,
    /// 假宿主**只**认另一种 IID 布局（模拟 Wine / wrapper 转发层）。
    alternate_iid_layout: Cell<bool>,
    calls: RefCell<Vec<String>>,
    revoke_at: Cell<usize>,
    change_at: Cell<usize>,
    missing: Cell<Option<&'static str>>,
    no_take: Cell<bool>,
    no_project: Cell<bool>,
    bad_type: Cell<Option<&'static str>>,
    values: RefCell<BTreeMap<&'static str, f64>>,
    markers: RefCell<Vec<(f64, f64, f64)>>,
    count_override: Cell<Option<i32>>,
    bad_marker: Cell<bool>,
    bad_guid: Cell<bool>,
    change: Cell<i32>,
    /// 轨道级 `GetMediaTrackInfo_Value` 的读数。
    ///
    /// 夹具不区分字段名，`IP_TRACKNUMBER` 与 `I_FOLDERDEPTH` 共用这一个值；默认 1
    /// 与本次改动之前完全一致（原先硬编码返回 1）。需要区分"folder 父轨"与"普通轨"
    /// 的用例必须显式设置它。
    pub track_value: Cell<f64>,
    pub hook: RefCell<Option<(String, Box<dyn Fn()>)>>,
    /// `TrackFX_*` / `TakeFX_*` / `I_FXEN` 这部分读数是否注册。
    ///
    /// 【为什么单独一个开关】旁通读数是**可选**能力：不注册时 `fx_bypass_state` 恒为
    /// `Unknown`（照改动前照常渲染）。把它挂在 `media_enabled` 上会让"能不能导入音频"
    /// 决定"能不能读旁通"，两组用例互相牵连。
    pub fx_api: Cell<bool>,
    /// 轨道 FX 链的内容（`TrackFX_*`）。
    pub fx_track_chain: RefCell<Vec<FxEntry>>,
    /// take FX 链的内容（`TakeFX_*`）。
    pub fx_take_chain: RefCell<Vec<FxEntry>>,
    /// `I_FXEN`（轨道整链总开关）。
    pub fx_chain_on: Cell<bool>,
    /// 是否注册 `TakeFX_*` 一族（缺省关：老宿主形态）。
    pub take_fx_api: Cell<bool>,
    /// 动作表里"Take: Toggle take pitch envelope"报的命令号（缺省 41612）。
    ///
    /// 【为什么要能改】"按名字解析"这条纪律只有在**名字与命令号对不上**时才能被验证：
    /// 用例把它改成一个只在夹具里存在的号码，实现却仍然跑对了，才说明它读的是名字。
    pub pitch_action_id: Cell<i32>,
    /// `parent(2)` 报哪条 take；0 = 主 take（多 take 用例把它指到别的 take 上）。
    pub parent_take: Cell<usize>,
    /// `GetItemStateChunk` / `SetItemStateChunk` 是否注册。
    pub item_state_enabled: Cell<bool>,
    /// `SetItemStateChunk` 是否像 take FX 下的真机那样**拒绝**写回。
    pub item_state_write_refused: Cell<bool>,
    /// take 包络那一族（`GetTakeEnvelopeByName` / `*EnvelopeStateChunk` / 动作）是否注册。
    pub take_env_enabled: Cell<bool>,
    /// 当前那条 take 的 Pitch 包络 chunk；`None` = 没有包络。
    pub take_env_chunk: RefCell<Option<String>>,
    /// item 的选择位（`B_UISEL`），动作路径会临时改它又改回来。
    pub item_ui_selected: Cell<bool>,
    /// 夹具当前持有的 item state chunk（`SetItemStateChunk` 会写进这里）。
    pub item_state_chunk: RefCell<String>,
}
impl Fixture {
    /// 夹具的指针会交给 `NativeHost` 持有（`IMainFactory` 那条 QI 通路），而
    /// `NativeHost` 的析构顺序由各测试的声明顺序决定 —— 测试里 `host` 往往比
    /// `owner` 先声明，于是它先被释放，之后 `Interface::drop` 会对着已释放的
    /// 内存调 `release`。Windows 的堆不擦除首字，这个 use-after-free 表现为
    /// "碰巧还能跑"；glibc 会把 tcache 指针写进首字，于是 vtbl 指针变成一个奇数
    /// 地址，debug 构建的未对齐检查直接 abort。
    ///
    /// 夹具本来就是进程级的（`ACTIVE` 是 thread_local、`fixture()` 返回
    /// `&'static Fixture`），那就让它真的活到进程结束：泄漏一份，换掉一整类
    /// 与执行顺序耦合的偶发崩溃。
    pub fn new() -> &'static Self {
        let value: &'static Self = Box::leak(Box::new(Self {
            table: &TABLE,
            refs: AtomicU32::new(1),
            take_token: 1,
            item_token: 2,
            track_token: 3,
            writer_enabled: Cell::new(false),
            take_volume: Cell::new(1.),
            right_item_token: 4,
            right_take_token: 5,
            split_enabled: Cell::new(false),
            right_values: RefCell::new(None),
            inventory_enabled: Cell::new(false),
            inventory_empty: Cell::new(false),
            takes_enabled: Cell::new(false),
            take_count_override: Cell::new(None),
            extra_takes: RefCell::new(Vec::new()),
            take_reversed: Cell::new(false),
            section_reports: Cell::new(false),
            take_section_start: Cell::new(0.),
            take_section_length: Cell::new(0.),
            source_file_name: RefCell::new(None),
            undo_records: RefCell::new(vec![std::ffi::CString::new("Initial state").unwrap()]),
            undo_position: Cell::new(0),
            media_enabled: Cell::new(false),
            media_sources: RefCell::new(Vec::new()),
            take_source: Cell::new(0),
            fail_media_take: Cell::new(false),
            media_path: RefCell::new(String::new()),
            extra_tracks: RefCell::new(Vec::new()),
            new_fx: RefCell::new(Default::default()),
            fx_guid_bytes: [4; 16],
            picker_paths: RefCell::new(Vec::new()),
            valid: Cell::new(true),
            connection_host: Cell::new(false),
            alternate_iid_layout: Cell::new(false),
            calls: RefCell::new(Vec::new()),
            revoke_at: Cell::new(0),
            change_at: Cell::new(0),
            missing: Cell::new(None),
            no_take: Cell::new(false),
            no_project: Cell::new(false),
            bad_type: Cell::new(None),
            values: RefCell::new(BTreeMap::from([
                ("D_POSITION", 1.),
                ("D_VOL", 1.),
                ("D_LENGTH", 4.),
                ("D_SNAPOFFSET", 0.),
                ("I_GROUPID", 0.),
                ("D_STARTOFFS", 0.),
                // 动作路径会临时切 active take（多 take item 上"给这条 take 建包络"）。
                ("I_CURTAKE", 0.),
                ("D_PLAYRATE", 0.5),
                ("B_PPITCH", 1.),
                ("I_CHANMODE", 2.),
                ("D_PITCH", 3.),
                ("C_BEATATTACHMODE", 1.),
                ("C_AUTOSTRETCH", 1.),
                ("B_MUTE", 0.),
                ("B_MUTE_ACTUAL", 0.),
                // 循环源是 item 属性（与 RPP 的 LOOP 行同源）。
                ("B_LOOPSRC", 0.),
                ("D_FADEINLEN", 0.2),
                ("D_FADEOUTLEN", 0.3),
                ("D_FADEINLEN_AUTO", 0.4),
                ("D_FADEOUTLEN_AUTO", 0.5),
                ("C_FADEINSHAPE", 1.),
                ("C_FADEOUTSHAPE", 6.),
                ("D_FADEINDIR", -0.1),
                ("D_FADEOUTDIR", 0.1),
                ("D_FADEINDIR_NEW", -0.2),
                ("D_FADEOUTDIR_NEW", 0.2),
                ("D_FADEINDIR2_NEW", -0.3),
                ("D_FADEOUTDIR2_NEW", 0.3),
            ])),
            markers: RefCell::new(vec![(-1., -2., 2.5), (6., 3., -2.5)]),
            count_override: Cell::new(None),
            bad_marker: Cell::new(false),
            bad_guid: Cell::new(false),
            change: Cell::new(-9),
            track_value: Cell::new(1.),
            hook: RefCell::new(None),
            fx_api: Cell::new(false),
            fx_track_chain: RefCell::new(Vec::new()),
            fx_take_chain: RefCell::new(Vec::new()),
            fx_chain_on: Cell::new(true),

            take_fx_api: Cell::new(false),
            pitch_action_id: Cell::new(41612),
            parent_take: Cell::new(0),
            item_state_enabled: Cell::new(false),
            item_state_write_refused: Cell::new(false),
            take_env_enabled: Cell::new(false),
            take_env_chunk: RefCell::new(None),
            item_ui_selected: Cell::new(false),
            item_state_chunk: RefCell::new(String::new()),
        }));
        ACTIVE.set(value as *const Self as usize);
        value
    }
    pub fn context(&self) -> *mut c_void {
        (self as *const Self as *mut Self).cast()
    }
    /// 仅分割定向夹具启用；不改变其余宿主fixture的可选能力。
    pub(crate) fn enable_split(&self) {
        self.split_enabled.set(true);
        self.writer_enabled.set(true);
    }
    /// take音量独立于item，允许用负号验证极性不被clip音量修改覆盖。
    pub(crate) fn set_take_volume(&self, value: f64) {
        self.take_volume.set(value);
        self.change.set(self.change.get().wrapping_add(1));
    }
    fn right_item(&self) -> *mut c_void {
        (&self.right_item_token as *const u8 as *mut u8).cast()
    }
    fn right_take(&self) -> *mut c_void {
        (&self.right_take_token as *const u8 as *mut u8).cast()
    }
    fn project(&self) -> *mut c_void {
        self.context()
    }
    fn take(&self) -> *mut c_void {
        (&self.take_token as *const u8 as *mut u8).cast()
    }
    /// `parent(2)` 此刻报的那条 take（默认为主 take）。
    fn parent_take(&self) -> *mut c_void {
        let override_ = self.parent_take.get();
        if override_ == 0 {
            self.take()
        } else {
            override_ as *mut c_void
        }
    }
    fn item(&self) -> *mut c_void {
        (&self.item_token as *const u8 as *mut u8).cast()
    }
    fn track(&self) -> *mut c_void {
        (&self.track_token as *const u8 as *mut u8).cast()
    }
    pub fn enable_writer(&self) {
        self.writer_enabled.set(true);
    }
    /// 启用结构读写（`Get/SetItemStateChunk`）并装入一份初始 chunk。
    ///
    /// 【为什么顺带打开媒体与多 take 枚举】方向写回要三样东西：chunk 文本、
    /// 媒体总长（镜像轴）、以及 `PCM_Source_GetSectionInfo` 的方向位（写后自校验）。
    /// 缺任何一个都只会得到"能力不可用"，测不到真正的写回路径。
    pub fn enable_item_state(&self, chunk: &str) {
        self.enable_media();
        self.takes_enabled.set(true);
        self.take_source.set(0x1000);
        self.section_reports.set(true);
        self.clear_markers();
        self.item_state_chunk.replace(chunk.to_string());
        self.item_state_enabled.set(true);
    }
    /// 一键装好"单 take 音频 item"的 chunk（GUID 与夹具的 take 身份一致）。
    pub fn single_take_chunk(&self) -> String {
        "<ITEM\nPOSITION 0\nLENGTH 0.00009070294784580499\nSEL 1\n\
         IGUID {11111111-1111-1111-1111-111111111111}\n\
         SOFFS 0\nPLAYRATE 1 1 0\nGUID {22222222-2222-2222-2222-222222222222}\n\
         <SOURCE WAVE\nFILE \"ara://source\"\n>\n>\n"
            .to_string()
    }
    pub fn enable_media(&self) {
        self.enable_writer();
        self.media_enabled.set(true);
    }
    /// 注册多 take 枚举接口；`takes` 为额外 take 数（不含 active）。
    pub fn enable_takes(&self, extra: usize) {
        self.inventory_enabled.set(true);
        self.takes_enabled.set(true);
        // 方向读取需要一个非空的 PCM_source；本组用例不启用媒体导入，故直接给哨兵值。
        self.take_source.set(0x1000);
        self.extra_takes.borrow_mut().clear();
        for _ in 0..extra {
            self.extra_takes.borrow_mut().push(Box::new(11));
        }
    }
    /// 模拟"宿主把同一个处理器搬到另一条链上"：take 链清空、轨道链上放本插件。
    pub fn forget_and_rearm_as_track_fx(&self) {
        self.fx_take_chain.borrow_mut().clear();
        *self.fx_track_chain.borrow_mut() = vec![FxEntry::HFS];
    }
    pub fn num_takes(&self) -> i32 {
        self.take_count_override
            .get()
            .unwrap_or(1 + self.extra_takes.borrow().len() as i32)
    }
    /// 第 `index` 个 take 的地址；0 = 主 take。
    pub fn take_at(&self, index: i32) -> *mut c_void {
        if index == 0 {
            return self.take();
        }
        let takes = self.extra_takes.borrow();
        takes
            .get(index as usize - 1)
            .map_or(std::ptr::null_mut(), |take| {
                (&**take as *const u8 as *mut u8).cast()
            })
    }
    /// 额外 take 的下标（1-based）；主 take 或未知指针返回 None。
    pub fn extra_take_index(&self, pointer: *mut c_void) -> Option<usize> {
        self.extra_takes
            .borrow()
            .iter()
            .position(|take| std::ptr::eq(&**take as *const u8, pointer as *const u8))
            .map(|index| index + 1)
    }
    pub fn clear_markers(&self) {
        self.markers.borrow_mut().clear();
    }
    pub fn client(&self) -> NativeHost {
        unsafe { NativeHost::from_context(self.context(), || self.valid.get()) }.unwrap()
    }
    pub fn reset(&self) {
        self.calls.borrow_mut().clear();
        self.revoke_at.set(0);
        self.change_at.set(0);
        self.valid.set(true);
    }
    pub fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }
    /// `GetProjectStateChangeCount` 当前读数。用例用它把"宿主 change count 不动"
    /// 这一前提钉住（take 级事实变化必须另有兜底路径穿透，见 `FACTS_REFRESH_INTERVAL`）。
    pub fn change_count(&self) -> i32 {
        self.change.get()
    }
    pub fn references(&self) -> u32 {
        self.refs.load(Ordering::Acquire)
    }
    pub fn enable_connection_host(&self) {
        self.connection_host.set(true);
    }
    /// 让假宿主按"另一种布局"比较 IID（见 `NativeHost::from_context` 的兜底重试）。
    pub fn force_alternate_iid_layout(&self) {
        self.alternate_iid_layout.set(true);
    }
    pub fn set_value(&self, name: &'static str, value: f64) {
        self.values.borrow_mut().insert(name, value);
        self.change.set(self.change.get().wrapping_add(1));
    }
    /// 读回夹具里某个字段的当前值（写口与 `set_value` 共用同一张表）。
    ///
    /// 【为什么需要】只看 `calls()` 能断言"写了哪个字段"，但断言不了"写成了什么值"。
    /// 形状预设落成 `(curvature, S)` 这一对数值时，后者才是要钉住的东西。
    /// 注意 `set_item` 只更新**已存在**的键，所以这里取不到的键等于"宿主写口忽略了它"。
    pub fn item_value(&self, name: &'static str) -> Option<f64> {
        self.values.borrow().get(name).copied()
    }
    fn record(&self, name: impl Into<String>) {
        assert!(
            self.valid.get(),
            "raw getter called after prior getter revoked authorization"
        );
        let name = name.into();
        self.calls.borrow_mut().push(name.clone());
        let index = self.calls.borrow().len();
        if self.revoke_at.get() == index {
            self.valid.set(false);
        }
        if self.change_at.get() == index {
            self.change.set(self.change.get().wrapping_add(1));
        }
        let matches = self
            .hook
            .borrow()
            .as_ref()
            .is_some_and(|(label, _)| *label == name);
        if matches {
            let (_, hook) = self.hook.borrow_mut().take().unwrap();
            hook();
        }
    }
    fn geometry(
        &self,
        client: &NativeHost,
    ) -> Result<crate::host::geometry::HostClipGeometry, String> {
        client.geometry(|| self.valid.get())
    }
}
fn fixture() -> &'static Fixture {
    unsafe { &*(ACTIVE.get() as *const Fixture) }
}
unsafe extern "system" fn query(this: *mut c_void, iid: *const u8, out: *mut *mut c_void) -> i32 {
    let f = unsafe { &*this.cast::<Fixture>() };
    f.record("QI");
    let requested = unsafe { std::slice::from_raw_parts(iid, 16) };
    // 真实宿主按平台布局比较 IID（Windows：COM/GUID；Linux / macOS：每字大端）。
    // 夹具默认照这个来；`force_alternate_iid_layout` 把它换成"另一套"，用来验证
    // `from_context` 的兜底重试不是摆设。
    let own_iid = if f.alternate_iid_layout.get() {
        crate::vst3_abi::alternate_iid(IID)
    } else {
        crate::vst3_abi::platform_iid(IID)
    };
    let host_iid = crate::vst3_abi::platform_iid([0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5]);
    if requested != own_iid && !(f.connection_host.get() && requested == host_iid) {
        unsafe {
            *out = std::ptr::null_mut();
        }
        return crate::vst3_abi::K_NO_INTERFACE;
    }
    unsafe {
        *out = this;
        add(this);
    }
    K_RESULT_OK
}
unsafe extern "system" fn add(this: *mut c_void) -> u32 {
    unsafe {
        (&*this.cast::<Fixture>())
            .refs
            .fetch_add(1, Ordering::AcqRel)
            + 1
    }
}
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    unsafe {
        (&*this.cast::<Fixture>())
            .refs
            .fetch_sub(1, Ordering::AcqRel)
            - 1
    }
}
unsafe extern "system" fn parent(this: *mut c_void, selector: u32) -> *mut c_void {
    let f = unsafe { &*this.cast::<Fixture>() };
    f.record(format!("parent:{selector}"));
    match selector {
        3 if !f.no_project.get() => f.project(),
        2 if !f.no_take.get() => f.parent_take(),
        1 if f.media_enabled.get() => f.track(),
        _ => std::ptr::null_mut(),
    }
}
unsafe extern "system" fn api(_: *mut c_void, name: *const c_char) -> *mut c_void {
    let f = fixture();
    let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_str().unwrap();
    f.record(format!("api:{name}"));
    if f.missing.get() == Some(name) {
        return std::ptr::null_mut();
    }
    let p = match name {
        "GetPlayPositionEx" | "GetCursorPositionEx" => position as *const (),
        "GetPlayStateEx" => state as *const (),
        "GetAppVersion" => version as *const (),
        "ValidatePtr2" => validate as *const (),
        "GetMediaItemTake_Item" => item as *const (),
        "GetMediaItemInfo_Value" => item_value as *const (),
        "GetMediaItemTakeInfo_Value" => take_value as *const (),
        "GetSetMediaItemInfo_String" => item_guid as *const (),
        "GetSetMediaItemTakeInfo_String" => take_guid as *const (),
        "GetTakeNumStretchMarkers" => count as *const (),
        "GetTakeStretchMarker" => marker as *const (),
        "GetTakeStretchMarkerSlope" => slope as *const (),
        "GetProjectStateChangeCount" => change as *const (),
        "SetMediaItemInfo_Value" if f.writer_enabled.get() => set_item as *const (),
        "SetMediaItemTakeInfo_Value" if f.writer_enabled.get() => set_take as *const (),
        "GetMediaItem_Track" if f.writer_enabled.get() => item_track as *const (),
        "MoveMediaItemToTrack" if f.writer_enabled.get() => move_item as *const (),
        "SplitMediaItem" if f.split_enabled.get() => split_item as *const (),
        "GetActiveTake" if f.split_enabled.get() => split_active_take as *const (),
        "Undo_BeginBlock2" if f.writer_enabled.get() => undo_begin as *const (),
        "Undo_EndBlock2" if f.writer_enabled.get() => undo_end as *const (),
        "UpdateItemInProject" if f.writer_enabled.get() => update_item as *const (),
        "UpdateArrange" if f.writer_enabled.get() => update_arrange as *const (),
        "Undo_DoUndo2" if f.writer_enabled.get() => do_undo as *const (),
        "Undo_DoRedo2" if f.writer_enabled.get() => do_redo as *const (),
        "Undo_CanUndo2" if f.writer_enabled.get() => can_undo as *const (),
        "Undo_CanRedo2" if f.writer_enabled.get() => can_redo as *const (),
        "Undo_GetCurEntry" if f.writer_enabled.get() => undo_current as *const (),
        "Undo_GetNumEntries" if f.writer_enabled.get() => undo_count as *const (),
        "Undo_GetEntryDesc" if f.writer_enabled.get() => undo_entry as *const (),
        "PCM_Source_CreateFromFileEx" if f.media_enabled.get() => media_source as *const (),
        "PCM_Source_Destroy" if f.media_enabled.get() => media_destroy as *const (),
        "GetMediaSourceLength" if f.media_enabled.get() => media_length as *const (),
        "AddMediaItemToTrack" if f.media_enabled.get() => media_item as *const (),
        "AddTakeToMediaItem" if f.media_enabled.get() => media_take as *const (),
        "GetSetMediaItemTakeInfo" if f.media_enabled.get() => media_info as *const (),
        "DeleteTrackMediaItem" if f.media_enabled.get() => media_delete as *const (),
        "GetSetMediaTrackInfo_String" if f.media_enabled.get() => media_track_guid as *const (),
        "GetTrackMediaItem" if f.inventory_enabled.get() => inventory_item as *const (),
        "GetActiveTake" if f.inventory_enabled.get() => inventory_take as *const (),
        "GetTakeName" if f.inventory_enabled.get() => inventory_name as *const (),
        "GetMediaItemNumTakes" if f.takes_enabled.get() => num_takes as *const (),
        "GetMediaItemTake" if f.takes_enabled.get() => take_at as *const (),
        "GetMediaItemTake_Source" if f.takes_enabled.get() => take_source as *const (),
        "PCM_Source_GetSectionInfo" if f.takes_enabled.get() => section_info as *const (),
        "GetMediaSourceFileName" if f.takes_enabled.get() => source_file_name as *const (),
        "GetMediaTrackInfo_Value" if f.inventory_enabled.get() || f.fx_api.get() => {
            inventory_track_value as *const ()
        }
        "GetItemStateChunk" if f.item_state_enabled.get() => get_item_state as *const (),
        "GetTakeEnvelopeByName" if f.take_env_enabled.get() => get_take_envelope as *const (),
        "GetEnvelopeStateChunk" if f.take_env_enabled.get() => get_envelope_state as *const (),
        "SetEnvelopeStateChunk" if f.take_env_enabled.get() => set_envelope_state as *const (),
        "kbd_enumerateActions" if f.take_env_enabled.get() => enumerate_actions as *const (),
        "CountSelectedMediaItems" if f.take_env_enabled.get() => count_selected as *const (),
        "GetSelectedMediaItem" if f.take_env_enabled.get() => selected_item as *const (),
        "SelectAllMediaItems" if f.take_env_enabled.get() => select_all as *const (),
        "SetMediaItemSelected" if f.take_env_enabled.get() => set_item_selected as *const (),
        "IsMediaItemSelected" if f.take_env_enabled.get() => item_is_selected as *const (),
        "SectionFromUniqueID" if f.take_env_enabled.get() => section_from_unique_id as *const (),
        "kbd_getTextFromCmd" if f.take_env_enabled.get() => action_text as *const (),
        "Main_OnCommandEx" if f.take_env_enabled.get() => run_command_ex as *const (),
        "Main_OnCommand" if f.take_env_enabled.get() => run_command as *const (),
        "SetItemStateChunk" if f.item_state_enabled.get() => set_item_state as *const (),
        "genGuid" if f.item_state_enabled.get() => gen_guid as *const (),
        "guidToString" if f.item_state_enabled.get() => guid_to_string as *const (),
        "GetUserFileNameForRead" if f.media_enabled.get() => media_picker as *const (),
        "GetUserFileName" if f.media_enabled.get() => multi_picker as *const (),
        "InsertTrackInProject" if f.media_enabled.get() => insert_track as *const (),
        "CountTracks" if f.media_enabled.get() => count_tracks as *const (),
        "GetTrack" if f.media_enabled.get() => get_track as *const (),
        "TrackFX_AddByName" if f.media_enabled.get() => add_fx as *const (),
        "DeleteTrack" if f.media_enabled.get() => delete_track as *const (),
        "CountTrackMediaItems" if f.media_enabled.get() => count_track_items as *const (),
        "TrackFX_GetCount" if f.fx_api.get() || f.media_enabled.get() => fx_count as *const (),
        "TrackFX_GetFXGUID" if f.media_enabled.get() => fx_guid as *const (),
        "TrackFX_GetEnabled" if f.fx_api.get() => fx_enabled as *const (),
        "TrackFX_GetOffline" if f.fx_api.get() => fx_offline as *const (),
        "TrackFX_GetFXName" if f.fx_api.get() => fx_name as *const (),
        "TakeFX_GetCount" if f.fx_api.get() && f.take_fx_api.get() => take_fx_count as *const (),
        "TakeFX_GetEnabled" if f.fx_api.get() && f.take_fx_api.get() => {
            take_fx_enabled as *const ()
        }
        "TakeFX_GetOffline" if f.fx_api.get() && f.take_fx_api.get() => {
            take_fx_offline as *const ()
        }
        "TakeFX_GetFXName" if f.fx_api.get() && f.take_fx_api.get() => take_fx_name as *const (),
        _ => std::ptr::null(),
    };
    p as *mut c_void
}
// 夹具显式采用目标REAPER7.81的新轴，不用缺省字段猜宿主版本。
//
// 【为什么是线程局部而不是 fixture 的字段】`version()` 会在夹具装好之前被调用，
// 而 `fixture()` 解引用的是 `ACTIVE` 里那个原始指针 —— 空指针直接段错误。
thread_local! {
    static APP_VERSION: Cell<&'static std::ffi::CStr> = const { Cell::new(c"7.81/x64") };
}

unsafe extern "C" fn version() -> *const c_char {
    APP_VERSION.with(|value| value.get().as_ptr())
}

/// B_MUTE才是item solo覆盖后的有效静音；原始mute仍开时solo也可让该clip播放。
#[test]
fn effective_item_mute_uses_host_solo_override_not_raw_mute() {
    let fixture = Fixture::new();
    let client = fixture.client();
    assert!(!fixture.geometry(&client).unwrap().muted);
    fixture.set_value("B_MUTE", 1.);
    fixture.set_value("B_MUTE_ACTUAL", 1.);
    assert!(fixture.geometry(&client).unwrap().muted);
    fixture.set_value("B_MUTE", 0.);
    assert!(
        !fixture.geometry(&client).unwrap().muted,
        "item solo覆盖必须与宿主有效状态一致"
    );
}
unsafe extern "system" fn extended(
    _: *mut c_void,
    _: u32,
    _: *mut c_void,
    _: *mut c_void,
    _: *mut c_void,
) -> *mut c_void {
    panic!("unverified extended opcode must not be queried")
}
static TABLE: HostVtbl = HostVtbl {
    base: UnknownVtbl {
        query,
        add,
        release,
    },
    api,
    parent,
    extended,
};
unsafe extern "C" fn position(project: *mut c_void) -> f64 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("position");
    2.5
}
unsafe extern "C" fn state(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("state");
    1
}
unsafe extern "C" fn change(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("change");
    f.change.get()
}
unsafe extern "C" fn validate(
    project: *mut c_void,
    object: *mut c_void,
    kind: *const c_char,
) -> bool {
    let f = fixture();
    assert_eq!(project, f.project());
    let kind = unsafe { std::ffi::CStr::from_ptr(kind) }.to_str().unwrap();
    f.record(format!("validate:{kind}"));
    if f.bad_type.get() == Some(kind) {
        return false;
    }
    match kind {
        "ReaProject*" => object == f.project(),
        "MediaItem_Take*" => {
            object == f.take()
                || (f.right_values.borrow().is_some() && object == f.right_take())
                || f.extra_take_index(object).is_some()
        }
        "MediaItem*" => {
            object == f.item() || (f.right_values.borrow().is_some() && object == f.right_item())
        }
        "PCM_source*" => object == f.take_source.get() as *mut c_void,
        "MediaTrack*" => {
            object == f.track()
                || f.extra_tracks
                    .borrow()
                    .iter()
                    .any(|track| std::ptr::eq(&**track as *const u8, object as *const u8))
        }
        _ => false,
    }
}
unsafe extern "C" fn item(take: *mut c_void) -> *mut c_void {
    let f = fixture();
    if take == f.right_take() {
        f.record("right-item");
        return f.right_item();
    }
    // 同一 item 的其余 take 共享同一 item 指针 —— 这正是 take 枚举的语义。
    if f.extra_take_index(take).is_some() {
        f.record("take-item");
        return f.item();
    }
    assert_eq!(take, f.take());
    f.record("item");
    f.item()
}
fn value(object: *mut c_void, name: *const c_char, is_take: bool) -> f64 {
    let f = fixture();
    if is_take && unsafe { CStr::from_ptr(name) } == c"D_VOL" {
        f.record("D_VOL");
        return f.take_volume.get();
    }
    // 【宿主对**未知**参数一律返回 0.0】这不是夹具偷懒，而是 REAPER 的真实行为：take FX 那次
    // "没被旁通却一点声音都没有"，根因就是 `B_TAKEFX` 这个参数在 take 上根本不存在，宿主
    // 照例回了 0，于是"整链总开关"读成"关"，渲染入口整段跳过。夹具照抄这条行为，这样
    // "再按轨道的形状去读一个 take 参数"会被用例当场抓住，而不是等下一次真机报障。
    if is_take {
        let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
        if !f.values.borrow().contains_key(name) {
            f.record(format!("unknown-take-param:{name}"));
            return 0.0;
        }
    }
    if object
        == if is_take {
            f.right_take()
        } else {
            f.right_item()
        }
    {
        let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
        f.record(name);
        return f.right_values.borrow().as_ref().unwrap()[name];
    }
    // 额外 take 与主 take 共用同一份数值表：本组用例只验证"枚举/身份/方向"，
    // 不靠逐 take 不同的数值来伪造结论。
    if is_take && f.extra_take_index(object).is_some() {
        let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
        f.record(name);
        return f.values.borrow()[name];
    }
    assert_eq!(object, if is_take { f.take() } else { f.item() });
    let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_str().unwrap();
    f.record(name);
    // 动作路径会读/写 item 的选择位（`B_UISEL`）：夹具按一个可断言的开关回答。
    if !is_take && name == "B_UISEL" {
        return if f.item_ui_selected.get() { 1. } else { 0. };
    }
    f.values.borrow()[name]
}
unsafe extern "C" fn item_value(object: *mut c_void, name: *const c_char) -> f64 {
    value(object, name, false)
}
unsafe extern "C" fn take_value(object: *mut c_void, name: *const c_char) -> f64 {
    value(object, name, true)
}
/// 真实typed setter边界夹具：记录宿主写入，不通过插件timeline偷偷模拟成功。
unsafe extern "C" fn set_item(object: *mut c_void, name: *const c_char, value: f64) -> bool {
    let f = fixture();
    assert_eq!(object, f.item());
    let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
    f.record(format!("write-item:{name}"));
    if name == "B_UISEL" {
        f.item_ui_selected.set(value != 0.0);
        return true;
    }
    let mut values = f.values.borrow_mut();
    let key = values.keys().find(|key| **key == name).copied();
    if let Some(key) = key {
        values.insert(key, value);
    }
    f.change.set(f.change.get().wrapping_add(1));
    true
}
unsafe extern "C" fn set_take(object: *mut c_void, name: *const c_char, value: f64) -> bool {
    let f = fixture();
    assert_eq!(object, f.take());
    let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
    f.record(format!("write-take:{name}"));
    let mut values = f.values.borrow_mut();
    let key = values.keys().find(|key| **key == name).copied();
    if let Some(key) = key {
        values.insert(key, value);
    }
    f.change.set(f.change.get().wrapping_add(1));
    true
}
unsafe extern "C" fn item_track(item: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert!(item == f.item() || item == f.right_item());
    f.record("item-track");
    f.track()
}
/// 模拟官方分割语义：左item保持GUID，右item拥有新GUID与按倍率推进的源起点。
unsafe extern "C" fn split_item(item: *mut c_void, position: f64) -> *mut c_void {
    let f = fixture();
    assert_eq!(item, f.item());
    f.record("split-item");
    let mut left = f.values.borrow_mut();
    let delta = position - left["D_POSITION"];
    let mut right = left.clone();
    right.insert("D_POSITION", position);
    right.insert("D_LENGTH", left["D_LENGTH"] - delta);
    right.insert(
        "D_STARTOFFS",
        left["D_STARTOFFS"] + delta * left["D_PLAYRATE"],
    );
    left.insert("D_LENGTH", delta);
    *f.right_values.borrow_mut() = Some(right);
    f.change.set(f.change.get().wrapping_add(1));
    f.right_item()
}
unsafe extern "C" fn split_active_take(item: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert_eq!(item, f.right_item());
    f.record("split-active-take");
    f.right_take()
}
unsafe extern "C" fn move_item(item: *mut c_void, track: *mut c_void) -> bool {
    let f = fixture();
    assert_eq!(item, f.item());
    assert_eq!(track, f.track());
    f.record("move-item");
    true
}
unsafe extern "C" fn undo_begin(project: *mut c_void) {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("undo-begin");
}
unsafe extern "C" fn undo_end(project: *mut c_void, label: *const c_char, flags: i32) {
    let f = fixture();
    assert_eq!(project, f.project());
    assert_eq!(flags, -1);
    f.record("undo-end");
    let mut records = f.undo_records.borrow_mut();
    records.truncate(f.undo_position.get() as usize + 1);
    records.push(unsafe { CStr::from_ptr(label) }.to_owned());
    f.undo_position.set(records.len() as i32 - 1);
    f.change.set(f.change.get().wrapping_add(1));
}
unsafe extern "C" fn undo_current(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("undo-current");
    f.undo_position.get()
}
unsafe extern "C" fn undo_count(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.undo_records.borrow().len() as i32
}
unsafe extern "C" fn undo_entry(project: *mut c_void, index: i32) -> *const c_char {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record(format!("undo-entry:{index}"));
    if index < 0 {
        return c"".as_ptr();
    }
    f.undo_records
        .borrow()
        .get(index as usize)
        .map_or(c"".as_ptr(), |s| s.as_ptr())
}
unsafe extern "C" fn can_undo(project: *mut c_void) -> *const c_char {
    let f = fixture();
    assert_eq!(project, f.project());
    if f.undo_position.get() > 0 {
        unsafe { undo_entry(project, f.undo_position.get()) }
    } else {
        std::ptr::null()
    }
}
unsafe extern "C" fn can_redo(project: *mut c_void) -> *const c_char {
    let f = fixture();
    assert_eq!(project, f.project());
    if f.undo_position.get() + 1 < f.undo_records.borrow().len() as i32 {
        unsafe { undo_entry(project, f.undo_position.get() + 1) }
    } else {
        std::ptr::null()
    }
}
unsafe extern "C" fn do_undo(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("undo-action");
    if f.undo_position.get() > 0 {
        f.undo_position.set(f.undo_position.get() - 1);
        f.change.set(f.change.get().wrapping_add(1));
        1
    } else {
        0
    }
}
unsafe extern "C" fn do_redo(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("redo-action");
    if f.undo_position.get() + 1 < f.undo_records.borrow().len() as i32 {
        f.undo_position.set(f.undo_position.get() + 1);
        f.change.set(f.change.get().wrapping_add(1));
        1
    } else {
        0
    }
}
unsafe extern "C" fn media_source(path: *const c_char, force: bool) -> *mut c_void {
    let f = fixture();
    assert!(force);
    f.record("media-source-create");
    *f.media_path.borrow_mut() = unsafe { CStr::from_ptr(path) }.to_str().unwrap().into();
    let mut source = Box::new(7_u8);
    let pointer = (&mut *source as *mut u8).cast();
    f.media_sources.borrow_mut().push(source);
    pointer
}
unsafe extern "C" fn media_destroy(pointer: *mut c_void) {
    let f = fixture();
    f.record("media-source-destroy");
    let mut sources = f.media_sources.borrow_mut();
    let index = sources
        .iter()
        .position(|source| std::ptr::eq(&**source as *const u8, pointer as *const u8))
        .expect("source must be owned and destroyed once");
    sources.remove(index);
}
unsafe extern "C" fn media_length(_source: *mut c_void, qn: *mut bool) -> f64 {
    fixture().record("media-length");
    unsafe { *qn = false };
    2.
}
unsafe extern "C" fn media_item(track: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert_eq!(track, f.track());
    f.record("media-item-create");
    f.item()
}
unsafe extern "C" fn media_take(item: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert_eq!(item, f.item());
    f.record("media-take-create");
    if f.fail_media_take.get() {
        std::ptr::null_mut()
    } else {
        f.take()
    }
}
unsafe extern "C" fn media_info(
    take: *mut c_void,
    name: *const c_char,
    new: *mut c_void,
) -> *mut c_void {
    let f = fixture();
    assert_eq!(take, f.take());
    assert_eq!(unsafe { CStr::from_ptr(name) }, c"P_SOURCE");
    if new.is_null() {
        f.record("media-source-read");
        f.take_source.get() as *mut c_void
    } else {
        f.record("media-source-attach");
        f.take_source.replace(new as usize) as *mut c_void
    }
}
unsafe extern "C" fn media_delete(track: *mut c_void, item: *mut c_void) -> bool {
    let f = fixture();
    assert_eq!(track, f.track());
    assert_eq!(item, f.item());
    f.record("media-item-delete");
    let source = f.take_source.replace(0);
    if source != 0 {
        unsafe { media_destroy(source as *mut c_void) };
    }
    true
}
unsafe extern "C" fn media_track_guid(
    track: *mut c_void,
    name: *const c_char,
    buffer: *mut c_char,
    set: bool,
) -> bool {
    let f = fixture();
    if set {
        assert_eq!(unsafe { CStr::from_ptr(name) }, c"P_NAME");
        f.record("track-name");
        return true;
    }
    if unsafe { CStr::from_ptr(name) } == c"P_NAME" {
        let text = c"Inventory track";
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), buffer, text.to_bytes_with_nul().len());
        }
        return true;
    }
    assert_eq!(unsafe { CStr::from_ptr(name) }, c"GUID");
    f.record("track-guid");
    let guid = if track == f.track() {
        std::ffi::CString::new("{33333333-3333-3333-3333-333333333333}").unwrap()
    } else {
        let tracks = f.extra_tracks.borrow();
        let index = tracks
            .iter()
            .position(|t| std::ptr::eq(&**t as *const u8, track as *const u8))
            .unwrap();
        std::ffi::CString::new(format!("{{44444444-4444-4444-4444-{:012}}}", index + 1)).unwrap()
    };
    unsafe {
        std::ptr::copy_nonoverlapping(guid.as_ptr(), buffer, guid.to_bytes_with_nul().len());
    }
    true
}
unsafe extern "C" fn media_picker(
    _path: *mut c_char,
    _title: *const c_char,
    _ext: *const c_char,
) -> bool {
    fixture().record("media-picker");
    false
}
unsafe extern "C" fn multi_picker(
    mode: i32,
    _caption: *const c_char,
    _initial: *const c_char,
    _extensions: *const c_char,
    buffer: *mut c_char,
    size: i32,
) -> bool {
    let f = fixture();
    assert!((1..=2).contains(&mode));
    f.record("media-multi-picker");
    let paths = f.picker_paths.borrow();
    if paths.is_empty() {
        return false;
    }
    let value = std::ffi::CString::new(paths.join("|")).unwrap();
    assert!(value.to_bytes_with_nul().len() < size as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(value.as_ptr(), buffer, value.to_bytes_with_nul().len());
    }
    true
}
unsafe extern "C" fn count_tracks(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("count-tracks");
    1 + f.extra_tracks.borrow().len() as i32
}
unsafe extern "C" fn get_track(project: *mut c_void, index: i32) -> *mut c_void {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record(format!("get-track:{index}"));
    if index == 0 {
        f.track()
    } else {
        f.extra_tracks
            .borrow()
            .get(index as usize - 1)
            .map_or(std::ptr::null_mut(), |track| {
                (&**track as *const u8 as *mut u8).cast()
            })
    }
}
unsafe extern "C" fn insert_track(project: *mut c_void, index: i32, flags: i32) {
    let f = fixture();
    assert_eq!(project, f.project());
    assert_eq!(flags, 0);
    // 夹具里轨道 0 恒为 FX 轨，因此合法插入点从 1 开始（"插在 FX 轨之后"）。
    let len = f.extra_tracks.borrow().len() as i32;
    assert!(
        (1..=len + 1).contains(&index),
        "insert index must land at/after the FX track: {index}"
    );
    f.record(format!("insert-track:{index}"));
    f.extra_tracks
        .borrow_mut()
        .insert(index as usize - 1, Box::new(9));
    f.change.set(f.change.get().wrapping_add(1));
}
unsafe extern "C" fn add_fx(
    track: *mut c_void,
    name: *const c_char,
    record: bool,
    index: i32,
) -> i32 {
    let f = fixture();
    assert_eq!(unsafe { CStr::from_ptr(name) }, c"VST3:HiFiShifter");
    assert!(!record);
    assert_eq!(index, -1000);
    f.record("add-hfs-first-fx");
    f.new_fx.borrow_mut().insert(track as usize);
    0
}
unsafe extern "C" fn delete_track(track: *mut c_void) {
    let f = fixture();
    f.record("delete-created-track");
    let mut tracks = f.extra_tracks.borrow_mut();
    let index = tracks
        .iter()
        .position(|t| std::ptr::eq(&**t as *const u8, track as *const u8))
        .unwrap();
    tracks.remove(index);
    f.new_fx.borrow_mut().remove(&(track as usize));
    f.change.set(f.change.get().wrapping_add(1));
}
unsafe extern "C" fn count_track_items(_track: *mut c_void) -> i32 {
    let f = fixture();
    i32::from(f.inventory_enabled.get() && !f.inventory_empty.get())
}
unsafe extern "C" fn inventory_item(track: *mut c_void, index: i32) -> *mut c_void {
    let f = fixture();
    assert_eq!(track, f.track());
    assert_eq!(index, 0);
    f.item()
}
unsafe extern "C" fn inventory_take(item: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert_eq!(item, f.item());
    f.take()
}
unsafe extern "C" fn inventory_name(_take: *mut c_void) -> *const c_char {
    c"short voice".as_ptr()
}
unsafe extern "C" fn inventory_track_value(track: *mut c_void, name: *const c_char) -> f64 {
    let f = fixture();
    assert_eq!(track, f.track());
    // 【为什么在这里分流】`GetMediaTrackInfo_Value` 同时服务 folder 读数与整链总开关；
    // 夹具按**字段名**回答，才能让"folder 深度"与"FX 链被关掉"是两个独立的事实。
    if unsafe { CStr::from_ptr(name) } == c"I_FXEN" {
        f.record("fx-chain-enabled");
        return if f.fx_chain_on.get() { 1. } else { 0. };
    }
    f.track_value.get()
}
unsafe extern "C" fn num_takes(item: *mut c_void) -> i32 {
    let f = fixture();
    // 分割出来的右半段与左半段 take 数相同（REAPER 两个新 item 各带全部 take）。
    if item == f.right_item() {
        f.record("num-takes-right");
        return f.num_takes();
    }
    assert_eq!(item, f.item());
    f.record("num-takes");
    f.num_takes()
}
unsafe extern "C" fn take_at(item: *mut c_void, index: i32) -> *mut c_void {
    let f = fixture();
    if item == f.right_item() {
        f.record(format!("take-at-right:{index}"));
        // 夹具只造**一条**右半段 take：它占据与本条序号相同的那一格（本组用例把实例
        // 挂在最后一条 take 上），其余序号返回空 —— 调用方必须据此拒绝，而不是退回
        // active take。
        if index == f.num_takes() - 1 {
            return f.right_take();
        }
        return std::ptr::null_mut();
    }
    assert_eq!(item, f.item());
    f.record(format!("take-at:{index}"));
    if index < 0 || index >= f.num_takes() {
        return std::ptr::null_mut();
    }
    f.take_at(index)
}
unsafe extern "C" fn take_source(take: *mut c_void) -> *mut c_void {
    let f = fixture();
    f.record("take-source");
    assert!(
        take == f.take() || f.extra_take_index(take).is_some() || take == f.right_take(),
        "source read must be for an enumerated take"
    );
    f.take_source.get() as *mut c_void
}
unsafe extern "C" fn section_info(
    source: *mut c_void,
    offset: *mut f64,
    length: *mut f64,
    reversed: *mut bool,
) -> bool {
    let f = fixture();
    assert_eq!(source, f.take_source.get() as *mut c_void);
    f.record("section-info");
    if !f.section_reports.get() {
        return false;
    }
    unsafe {
        *offset = f.take_section_start.get();
        *length = f.take_section_length.get();
        *reversed = f.take_reversed.get();
    }
    true
}

/// 夹具里的包络句柄：只要非空就代表"这条 take 有 Pitch 包络"。
const ENVELOPE_HANDLE: usize = 0xE70E;

/// `GetTakeEnvelopeByName(MediaItem_Take*, const char*) -> TrackEnvelope*`。
unsafe extern "C" fn get_take_envelope(
    take: *mut c_void,
    name: *const std::ffi::c_char,
) -> *mut c_void {
    let f = fixture();
    assert_eq!(take, f.take());
    let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
    f.record(format!("get-take-envelope:{name}"));
    if f.take_env_chunk.borrow().is_some() {
        ENVELOPE_HANDLE as *mut c_void
    } else {
        std::ptr::null_mut()
    }
}

/// `GetEnvelopeStateChunk(TrackEnvelope*, char*, int, bool)`。
unsafe extern "C" fn get_envelope_state(
    envelope: *mut c_void,
    buffer: *mut std::ffi::c_char,
    size: i32,
    _is_undo: bool,
) -> bool {
    let f = fixture();
    assert_eq!(envelope as usize, ENVELOPE_HANDLE);
    f.record("get-envelope-state");
    let stored = f.take_env_chunk.borrow().clone();
    let Some(text) = stored else {
        return false;
    };
    let bytes = text.as_bytes();
    if size <= 0 || bytes.len() + 1 > size as usize {
        return false;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().cast::<std::ffi::c_char>(),
            buffer,
            bytes.len(),
        );
        *buffer.add(bytes.len()) = 0;
    }
    true
}

/// `SetEnvelopeStateChunk(TrackEnvelope*, const char*, bool)`。
///
/// 【为什么"只有头没有点"要存成 `None`】真机行为：这样的块会让宿主删掉整条包络。
unsafe extern "C" fn set_envelope_state(
    envelope: *mut c_void,
    text: *const std::ffi::c_char,
    _is_undo: bool,
) -> bool {
    let f = fixture();
    assert_eq!(envelope as usize, ENVELOPE_HANDLE);
    f.record("set-envelope-state");
    if text.is_null() {
        return false;
    }
    let chunk = unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .to_string();
    let has_points = chunk
        .lines()
        .any(|line| line.split_whitespace().next() == Some("PT"));
    f.take_env_chunk
        .replace(if has_points { Some(chunk) } else { None });
    true
}

/// 夹具里的主动作段句柄。
const SECTION_HANDLE: usize = 0x5EC7;

/// 夹具的动作表：一条无关动作 + 本插件要的两条。
///
/// 【为什么"无关动作"必须在最前面】真实现是按索引逐条枚举、按名字匹配；只有表里存在
/// 不匹配的条目，"匹配错位置"才会被用例抓住。
fn action_table() -> [(&'static std::ffi::CStr, i32); 3] {
    [
        (c"File: New project", 40023),
        (
            c"Take: Toggle take pitch envelope",
            fixture().pitch_action_id.get(),
        ),
        (c"Item properties: Toggle take reverse", 41051),
    ]
}

/// `KbdSectionInfo* SectionFromUniqueID(int uniqueID)`：只认主动作段（0）。
unsafe extern "C" fn section_from_unique_id(unique: i32) -> *mut c_void {
    if unique == 0 {
        SECTION_HANDLE as *mut c_void
    } else {
        std::ptr::null_mut()
    }
}

/// `int kbd_enumerateActions(KbdSectionInfo*, int idx, const char** nameOut)`。
///
/// 【签名为什么要照真机】出参是**指向名字的指针**（`const char**`），不是字符缓冲区。
/// 夹具照抄这一点，才可能抓住"把出参当缓冲区读"的写法（真机上的现场就是它）。
unsafe extern "C" fn enumerate_actions(
    section: *mut c_void,
    index: i32,
    name_out: *mut *const std::ffi::c_char,
) -> i32 {
    let f = fixture();
    assert_eq!(section as usize, SECTION_HANDLE, "必须传主动作段的指针");
    f.record(format!("enumerate:{index}"));
    let table = action_table();
    let Some((name, command)) = table.get(index as usize) else {
        return 0;
    };
    if !name_out.is_null() {
        unsafe { *name_out = name.as_ptr() };
    }
    *command
}

/// 夹具里的选择态只有一个 item（`item_ui_selected`），足以验证"只影响目标 item"。
unsafe extern "C" fn count_selected(_project: *mut c_void) -> i32 {
    i32::from(fixture().item_ui_selected.get())
}

unsafe extern "C" fn selected_item(_project: *mut c_void, index: i32) -> *mut c_void {
    let f = fixture();
    if index == 0 && f.item_ui_selected.get() {
        f.item()
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn select_all(_project: *mut c_void, selected: bool) {
    fixture().item_ui_selected.set(selected);
}

unsafe extern "C" fn set_item_selected(item: *mut c_void, selected: bool) {
    let f = fixture();
    assert_eq!(item, f.item());
    f.item_ui_selected.set(selected);
}

unsafe extern "C" fn item_is_selected(item: *mut c_void) -> bool {
    let f = fixture();
    assert_eq!(item, f.item());
    f.item_ui_selected.get()
}

/// `const char* kbd_getTextFromCmd(int cmd, KbdSectionInfo* section)`。
unsafe extern "C" fn action_text(command: i32, _section: *mut c_void) -> *const std::ffi::c_char {
    for (name, id) in action_table() {
        if id == command {
            return name.as_ptr();
        }
    }
    std::ptr::null()
}

/// 动作的**效果**：建一条 take pitch 包络 / 翻转 take 方向。
fn action_effect(command: i32) {
    let f = fixture();
    if command == f.pitch_action_id.get() && f.take_env_chunk.borrow().is_none() {
        f.take_env_chunk.replace(Some(
            "<PITCHENV\nACT 1 -1\nVIS 1 1 1\nLANEHEIGHT 0 0\nARM 0\nDEFSHAPE 0 -1 -1\nPT 0 0 0\n>\n"
                .to_string(),
        ));
    }
    if command == 41051 {
        let reversed = !f.take_reversed.get();
        f.take_reversed.set(reversed);
    }
}

/// `Main_OnCommandEx(int command, int flag, ReaProject* proj)`。
unsafe extern "C" fn run_command_ex(command: i32, _flag: i32, project: *mut c_void) {
    let f = fixture();
    assert_eq!(project, f.project(), "动作必须落在本实例的工程上");
    f.record(format!("run-command-ex:{command}"));
    action_effect(command);
}

/// `Main_OnCommand(int, int)`：老宿主形态。
unsafe extern "C" fn run_command(command: i32, _flag: i32) {
    let f = fixture();
    f.record(format!("run-command:{command}"));
    action_effect(command);
}

/// `GetItemStateChunk(MediaItem*, char* buf, int size, bool isUndo)`。
unsafe extern "C" fn get_item_state(
    item: *mut c_void,
    buffer: *mut std::ffi::c_char,
    size: i32,
    _is_undo: bool,
) -> bool {
    let f = fixture();
    assert!(item == f.item() || item == f.right_item());
    f.record("get-item-state");
    let chunk = f.item_state_chunk.borrow().clone();
    let bytes = chunk.as_bytes();
    // 与 REAPER 一致：放不下就返回 false（调用方据此扩容重试）。
    if size <= 0 || bytes.len() + 1 > size as usize {
        return false;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().cast::<std::ffi::c_char>(),
            buffer,
            bytes.len(),
        );
        *buffer.add(bytes.len()) = 0;
    }
    true
}

/// `SetItemStateChunk(MediaItem*, const char* chunk, bool isUndo)`。
///
/// 【为什么夹具要"解析"方向】写回之后调用方会立刻用
/// `PCM_Source_GetSectionInfo` 做一次"写→读→比"。夹具若只是存下文本，
/// 那一步永远读到旧值，自校验就成了摆设 —— 只有真的按 chunk 里的 MODE 更新
/// 方向位，这条回归才有意义。
unsafe extern "C" fn set_item_state(
    item: *mut c_void,
    text: *const std::ffi::c_char,
    _is_undo: bool,
) -> bool {
    let f = fixture();
    assert!(item == f.item() || item == f.right_item());
    f.record("set-item-state");
    // 【take FX 下的真机行为】宿主拒绝重写一个挂着本实例的 item（见 `take_env` 的模块文档）。
    if f.item_state_write_refused.get() {
        return false;
    }
    if text.is_null() {
        return false;
    }
    let chunk = unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .to_string();
    if chunk.is_empty() {
        return false;
    }
    let sectioned = chunk.contains("<SOURCE SECTION");
    let field = |name: &str| -> Option<f64> {
        chunk
            .lines()
            .find(|line| line.trim_start().starts_with(&format!("{name} ")))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<f64>().ok())
    };
    let mode = field("MODE").map_or(0, |value| value as i32);
    // 与 REAPER 7.85 一致：`MODE` 是位 —— 1 = 段边界无效（整份媒体）、2 = 反向。
    f.take_reversed.set(sectioned && mode & 2 != 0);
    if sectioned && mode & 1 == 0 {
        f.take_section_start.set(field("STARTPOS").unwrap_or(0.));
        f.take_section_length.set(field("LENGTH").unwrap_or(0.));
    } else if sectioned {
        // `MODE 1/3`：宿主报的是整份媒体（见 `source_direction` 的真机读数表）。
        f.take_section_start.set(0.);
        f.take_section_length.set(2.);
    } else {
        f.take_section_start.set(0.);
        f.take_section_length.set(0.);
    }
    f.item_state_chunk.replace(chunk);
    f.change.set(f.change.get().wrapping_add(1));
    true
}

/// `genGuid(GUID*)`：夹具不需要真 GUID，给一个确定的零值即可（身份校验走
/// `GetSetMediaItem*Info_String`，不读这里）。
unsafe extern "C" fn gen_guid(out: *mut super::item_chunk::NativeGuid) {
    unsafe { std::ptr::write(out, super::item_chunk::NativeGuid::default()) };
}

/// `guidToString(const GUID*, char*)`。
unsafe extern "C" fn guid_to_string(
    _guid: *const super::item_chunk::NativeGuid,
    buffer: *mut std::ffi::c_char,
) {
    let text = c"{00000000-0000-0000-0000-000000000000}";
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer, text.to_bytes_with_nul().len());
    }
}

/// `GetMediaSourceFileName(PCM_source*, char* buf, int sz)`：把夹具里记的路径写进缓冲。
unsafe extern "C" fn source_file_name(
    source: *mut c_void,
    buffer: *mut std::ffi::c_char,
    size: i32,
) {
    let f = fixture();
    assert_eq!(source, f.take_source.get() as *mut c_void);
    f.record("source-file-name");
    if buffer.is_null() || size <= 0 {
        return;
    }
    let path = f.source_file_name.borrow();
    let Some(path) = path.as_deref() else {
        // 读不到：写一个空串（调用方按"空 = 不可用"处理）。
        unsafe { *buffer = 0 };
        return;
    };
    let bytes = path.as_bytes();
    let count = bytes.len().min(size as usize - 1);
    // SAFETY: 调用方给出 size 字节的可写缓冲；只写 count+1 字节（含 NUL）。
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), count);
        *buffer.add(count) = 0;
    }
}

/// folder 判据走**单值读取**：FX 轨自己的 `I_FOLDERDEPTH` 就是结论，不枚举整个工程。
///
/// 【为什么值得单测】"本 FX 挂在 folder 父轨上"这句提示一旦说错，用户会照着一个
/// 错误方向去改工程。所以这里既钉住读到的值原样透传（0 / 1 / -1 都不改写），也钉住
/// 越界值**报错而不是当成 0** —— 报错让调用方保持"不知道"，退化成"等待中"提示。
#[test]
fn folder_depth_is_read_from_the_fx_track_alone() {
    let f = Fixture::new();
    f.enable_media();
    f.inventory_enabled.set(true);
    let host = std::sync::Arc::new(f.client());
    let target = host.direct_track_target(&|| true).unwrap();
    for depth in [0., 1., -1.] {
        f.track_value.set(depth);
        assert_eq!(target.folder_depth(&|| true).unwrap(), depth as i32);
    }
    for broken in [2., 1.5, f64::NAN] {
        f.track_value.set(broken);
        assert!(
            target.folder_depth(&|| true).is_err(),
            "越界/非整数深度必须报错而不是当成 0：{broken}"
        );
    }
}
unsafe extern "C" fn fx_count(track: *mut c_void) -> i32 {
    let f = fixture();
    // 【两套语义】`fx_api` 打开时按显式的链内容回答（旁通用例）；否则沿用"新建轨道
    // 自带一个 FX"的旧行为（建轨用例）。
    if f.fx_api.get() {
        f.record("track-fx-count");
        return f.fx_track_chain.borrow().len() as i32;
    }
    i32::from(f.new_fx.borrow().contains(&(track as usize)))
}

unsafe extern "C" fn fx_enabled(track: *mut c_void, index: i32) -> bool {
    let f = fixture();
    assert_eq!(track, f.track());
    f.record(format!("track-fx-enabled:{index}"));
    f.fx_track_chain.borrow()[index as usize].enabled
}

unsafe extern "C" fn fx_offline(track: *mut c_void, index: i32) -> bool {
    let f = fixture();
    assert_eq!(track, f.track());
    f.record(format!("track-fx-offline:{index}"));
    f.fx_track_chain.borrow()[index as usize].offline
}

unsafe extern "C" fn fx_name(
    track: *mut c_void,
    index: i32,
    buffer: *mut c_char,
    size: i32,
) -> bool {
    let f = fixture();
    assert_eq!(track, f.track());
    f.record(format!("track-fx-name:{index}"));
    copy_fx_name(f.fx_track_chain.borrow()[index as usize].name, buffer, size)
}

unsafe extern "C" fn take_fx_count(take: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(
        take,
        f.parent_take(),
        "take FX chain read must use the bound take"
    );
    f.record("take-fx-count");
    f.fx_take_chain.borrow().len() as i32
}

unsafe extern "C" fn take_fx_enabled(take: *mut c_void, index: i32) -> bool {
    let f = fixture();
    assert_eq!(take, f.parent_take());
    f.record(format!("take-fx-enabled:{index}"));
    f.fx_take_chain.borrow()[index as usize].enabled
}

unsafe extern "C" fn take_fx_offline(take: *mut c_void, index: i32) -> bool {
    let f = fixture();
    assert_eq!(take, f.parent_take());
    f.record(format!("take-fx-offline:{index}"));
    f.fx_take_chain.borrow()[index as usize].offline
}

unsafe extern "C" fn take_fx_name(
    take: *mut c_void,
    index: i32,
    buffer: *mut c_char,
    size: i32,
) -> bool {
    let f = fixture();
    assert_eq!(take, f.parent_take());
    f.record(format!("take-fx-name:{index}"));
    copy_fx_name(f.fx_take_chain.borrow()[index as usize].name, buffer, size)
}

/// 把 FX 显示名写进宿主缓冲（NUL 结尾，放不下就截断 —— 与宿主同一条纪律）。
fn copy_fx_name(name: &str, buffer: *mut c_char, size: i32) -> bool {
    if buffer.is_null() || size <= 0 {
        return false;
    }
    let bytes = name.as_bytes();
    let count = bytes.len().min(size as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), count);
        *buffer.add(count) = 0;
    }
    true
}
unsafe extern "C" fn fx_guid(track: *mut c_void, index: i32) -> *const u8 {
    let f = fixture();
    assert_eq!(index, 0);
    if f.new_fx.borrow().contains(&(track as usize)) {
        f.fx_guid_bytes.as_ptr()
    } else {
        std::ptr::null()
    }
}

#[test]
fn host_media_new_track_uses_project_guid_difference_and_first_plugin_and_rolls_back_empty_failure()
{
    let f = Fixture::new();
    f.enable_media();
    let host = std::sync::Arc::new(f.client());
    {
        let _track = host.create_audio_track("元音", None, &|| true).unwrap();
        assert_eq!(f.extra_tracks.borrow().len(), 1);
        assert!(f.calls().contains(&"add-hfs-first-fx".into()));
    }
    assert!(f.extra_tracks.borrow().is_empty());
    assert!(f.calls().contains(&"delete-created-track".into()));
    let mut track = host.create_audio_track("保留", None, &|| true).unwrap();
    track.commit();
    drop(track);
    assert_eq!(f.extra_tracks.borrow().len(), 1);
}

/// 插入锚点决定新轨道落点；锚点缺失或越界一律退回工程末尾，不报错。
#[test]
fn host_media_new_track_honours_the_anchor_and_falls_back_to_the_end() {
    let f = Fixture::new();
    f.enable_media();
    let host = std::sync::Arc::new(f.client());
    // 锚点 = `IP_TRACKNUMBER` 1 的唯一轨道 → 新轨道占据 0-based index 1。
    let mut track = host
        .create_audio_track("锚点后", Some(1), &|| true)
        .unwrap();
    track.commit();
    drop(track);
    assert!(f.calls().contains(&"insert-track:1".into()));
    // 越界锚点（工程里只有 2 条轨道）→ 退回末尾 index 2。
    let mut track = host.create_audio_track("末尾", Some(99), &|| true).unwrap();
    track.commit();
    drop(track);
    assert!(f.calls().contains(&"insert-track:2".into()));
    // 0 / 负值同样退回末尾（`IP_TRACKNUMBER` 是 1-based，0 不是合法锚点）。
    for anchor in [Some(0), Some(-4), None] {
        let mut track = host.create_audio_track("末尾", anchor, &|| true).unwrap();
        track.commit();
        drop(track);
    }
    assert_eq!(f.extra_tracks.borrow().len(), 5);
    assert!(f.calls().contains(&"insert-track:5".into()));
}
/// 测试用的绝对路径：三个平台写法不同，而被测的校验恰恰是 `Path::is_absolute()`。
/// 在 Linux 上 `E:\x.wav` 不是绝对路径，直接用 Windows 字面量会让用例在非 Windows
/// 平台上测到一个**正确**的拒绝，然后误报成回归。
fn abs_path(relative: &str) -> String {
    if cfg!(windows) {
        format!(r"E:\{relative}")
    } else {
        format!("/tmp/hifishifter-tests/{relative}")
    }
}

#[test]
fn host_media_multi_picker_preserves_complete_unicode_paths_and_cancel_is_empty() {
    let f = Fixture::new();
    f.enable_media();
    let host = std::sync::Arc::new(f.client());
    assert!(host.pick_audio_paths(true, &|| true).unwrap().is_empty());
    let paths = vec![abs_path("声音/第一段.wav"), abs_path("声音/第二段.wav")];
    *f.picker_paths.borrow_mut() = paths.clone();
    assert_eq!(host.pick_audio_paths(true, &|| true).unwrap(), paths);
}

/// 真实typed媒体对象所有权合同；不是假造ARA PCM或实际REAPER导入验收。
#[test]
fn host_media_import_preserves_utf8_and_transfers_source_ownership() {
    let f = Fixture::new();
    f.enable_media();
    let host = std::sync::Arc::new(f.client());
    let target = host.direct_track_target(&|| true).unwrap();
    let path = abs_path("素材/辅音和元音.wav");
    let guid = target.import_audio(&path, 3., &|| true).unwrap();
    assert_eq!(guid, "{11111111-1111-1111-1111-111111111111}");
    assert_eq!(f.media_path.borrow().as_str(), path.as_str());
    assert_eq!(f.values.borrow()["D_POSITION"], 3.);
    assert_eq!(f.values.borrow()["D_LENGTH"], 2.);
    assert_eq!(f.media_sources.borrow().len(), 1);
    assert!(!f.calls().contains(&"media-source-destroy".into()));
    unsafe { media_delete(f.track(), f.item()) };
    assert!(f.media_sources.borrow().is_empty());
}
#[test]
fn host_media_import_failure_removes_only_created_item_and_frees_unattached_source() {
    let f = Fixture::new();
    f.enable_media();
    f.fail_media_take.set(true);
    let host = std::sync::Arc::new(f.client());
    let target = host.direct_track_target(&|| true).unwrap();
    assert!(target
        .import_audio(&abs_path("clip.wav"), 0., &|| true)
        .unwrap_err()
        .contains("take creation"));
    assert!(f.calls().contains(&"media-item-delete".into()));
    assert_eq!(
        f.calls()
            .iter()
            .filter(|s| s.as_str() == "media-source-destroy")
            .count(),
        1
    );
    assert!(f.media_sources.borrow().is_empty());
}
unsafe extern "C" fn update_item(item: *mut c_void) {
    let f = fixture();
    assert!(item == f.item() || item == f.right_item());
    f.record("update-item");
}
unsafe extern "C" fn update_arrange() {
    fixture().record("update-arrange");
}

/// 已捕获地址被宿主重用后，GUID核对失败必须发生在任何setter之前。
#[test]
fn host_edit_reused_item_identity_never_reaches_setter() {
    let fixture = Fixture::new();
    fixture.enable_writer();
    let client = std::sync::Arc::new(fixture.client());
    let target = client.clip_target(|| true).unwrap();
    fixture.reset();
    fixture.bad_guid.set(true);
    assert!(target
        .set_item(c"D_POSITION", 3., &|| true)
        .unwrap_err()
        .contains("identity"));
    assert!(!fixture
        .calls()
        .iter()
        .any(|name| name.starts_with("write-")));
}

/// 第一笔写入可触发关闭/路由撤销；后续setter停止，Undo块仍在原project收尾。
#[test]
fn host_edit_reentry_revokes_later_writes_and_finishes_undo() {
    let fixture = Fixture::new();
    fixture.enable_writer();
    let client = std::sync::Arc::new(fixture.client());
    let live = std::rc::Rc::new(Cell::new(true));
    let target = client.clip_target(|| live.get()).unwrap();
    let block = target.begin_undo(&|| live.get()).unwrap();
    fixture.reset();
    let flag = live.clone();
    *fixture.hook.borrow_mut() = Some((
        "write-item:D_POSITION".into(),
        Box::new(move || flag.set(false)),
    ));
    assert!(target.set_item(c"D_POSITION", 3., &|| live.get()).is_err());
    assert!(target.set_take(c"D_PLAYRATE", 2., &|| live.get()).is_err());
    drop(block);
    let calls = fixture.calls();
    assert_eq!(
        calls
            .iter()
            .filter(|name| name.starts_with("write-"))
            .count(),
        1
    );
    assert!(calls.contains(&"undo-end".into()));
    assert!(!calls.contains(&"write-take:D_PLAYRATE".into()));
}
fn guid(
    object: *mut c_void,
    name: *const c_char,
    buffer: *mut c_char,
    write: bool,
    is_take: bool,
) -> bool {
    let f = fixture();
    let right = object
        == if is_take {
            f.right_take()
        } else {
            f.right_item()
        };
    let extra = if is_take {
        f.extra_take_index(object)
    } else {
        None
    };
    assert!(right || extra.is_some() || object == if is_take { f.take() } else { f.item() });
    assert!(!write, "GUID must remain read-only");
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"GUID");
    f.record(if is_take { "take_guid" } else { "item_guid" });
    let owned;
    let text = if f.bad_guid.get() {
        c"bad"
    } else if let Some(index) = extra {
        owned =
            std::ffi::CString::new(format!("{{55555555-5555-5555-5555-{:012}}}", index)).unwrap();
        owned.as_c_str()
    } else if right && is_take {
        c"{44444444-4444-4444-4444-444444444444}"
    } else if right {
        c"{33333333-3333-3333-3333-333333333333}"
    } else if is_take {
        c"{22222222-2222-2222-2222-222222222222}"
    } else {
        c"{11111111-1111-1111-1111-111111111111}"
    };
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer, text.to_bytes_with_nul().len());
    }
    true
}
unsafe extern "C" fn item_guid(o: *mut c_void, n: *const c_char, b: *mut c_char, w: bool) -> bool {
    guid(o, n, b, w, false)
}
unsafe extern "C" fn take_guid(o: *mut c_void, n: *const c_char, b: *mut c_char, w: bool) -> bool {
    guid(o, n, b, w, true)
}
unsafe extern "C" fn count(take: *mut c_void) -> i32 {
    let f = fixture();
    if take == f.right_take() {
        f.record("right-count");
        return 0;
    }
    assert!(
        take == f.take() || f.extra_take_index(take).is_some(),
        "stretch markers are read for an enumerated take"
    );
    f.record("count");
    f.count_override
        .get()
        .unwrap_or(f.markers.borrow().len() as i32)
}
unsafe extern "C" fn marker(take: *mut c_void, index: i32, pos: *mut f64, src: *mut f64) -> i32 {
    let f = fixture();
    assert!(take == f.take() || f.extra_take_index(take).is_some());
    f.record(format!("marker:{index}"));
    if f.bad_marker.get() {
        return -1;
    }
    let (p, s, _) = f.markers.borrow()[index as usize];
    unsafe {
        *pos = p;
        *src = s;
    }
    index
}
unsafe extern "C" fn slope(take: *mut c_void, index: i32) -> f64 {
    let f = fixture();
    assert!(take == f.take() || f.extra_take_index(take).is_some());
    f.record(format!("slope:{index}"));
    f.markers.borrow()[index as usize].2
}

#[test]
fn initialization_without_project_retains_interface_and_binds_only_the_later_direct_parent() {
    let f = Fixture::new();
    f.no_project.set(true);
    let client = f.client();
    assert_eq!(f.references(), 2);
    f.reset();
    assert!(client
        .sample(|| f.valid.get())
        .unwrap_err()
        .contains("not attached"));
    assert_eq!(
        f.calls(),
        ["parent:3"],
        "不能把null传给API而暗中使用当前活动project"
    );
    f.no_project.set(false);
    f.reset();
    assert_eq!(client.sample(|| f.valid.get()).unwrap(), (2.5, true));
    assert!(f.calls().contains(&"parent:3".into()));
    f.reset();
    client.sample(|| f.valid.get()).unwrap();
    assert!(
        !f.calls().contains(&"parent:3".into()),
        "绑定后不跟随活动tab重新选项目"
    );
    drop(client);
    assert_eq!(f.references(), 1);
}
#[test]
fn deferred_project_query_reentry_cannot_bind_or_call_the_next_api() {
    let f = Fixture::new();
    f.no_project.set(true);
    let client = f.client();
    f.no_project.set(false);
    f.reset();
    f.revoke_at.set(1);
    assert!(client.sample(|| f.valid.get()).is_err());
    assert_eq!(f.calls(), ["parent:3"]);
    assert_eq!(
        client.project.load(Ordering::Acquire),
        0,
        "外部调用撤销许可后不保存刚返回的parent"
    );
    drop(client);
    assert_eq!(f.references(), 1);
}
#[test]
fn task38a_native_geometry_preserves_direct_identity_new_fades_and_outside_raw_markers() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    let g = f.geometry(&client).unwrap();
    assert_eq!(
        (
            g.start_sec,
            g.duration_sec,
            g.source_start_sec,
            g.playback_rate
        ),
        (1., 4., 0., 0.5)
    );
    assert_eq!(
        (
            g.preserve_pitch,
            g.channel_mode,
            g.take_pitch,
            g.item_timebase,
            g.auto_stretch
        ),
        (true, 2, 3., 1, true)
    );
    assert_eq!(g.item_id, "{11111111-1111-1111-1111-111111111111}");
    assert_eq!(g.take_id, "{22222222-2222-2222-2222-222222222222}");
    assert_eq!(
        (
            g.fade_in_sec,
            g.fade_out_sec,
            g.auto_fade_in_sec,
            g.auto_fade_out_sec
        ),
        (0.2, 0.3, 0.4, 0.5)
    );
    assert_eq!(
        (
            g.fade_in_shape,
            g.fade_out_shape,
            g.fade_in_dir,
            g.fade_out_dir
        ),
        (1., 6., -0.1, 0.1)
    );
    assert_eq!(
        (
            g.fade_in_dir_new,
            g.fade_out_dir_new,
            g.fade_in_dir2_new,
            g.fade_out_dir2_new
        ),
        (-0.2, 0.2, -0.3, 0.3)
    );
    assert_eq!(
        g.markers
            .iter()
            .map(|m| (m.item_position_raw, m.source_position_raw, m.slope_raw))
            .collect::<Vec<_>>(),
        [(-1., -2., 2.5), (6., 3., -2.5)]
    );
    assert!(f.calls.borrow().contains(&"parent:2".into()));
    assert_eq!(f.refs.load(Ordering::Acquire), 2);
    drop(client);
    assert_eq!(f.refs.load(Ordering::Acquire), 1);
}

#[test]
fn task38a_missing_geometry_apis_and_take_never_disable_transport() {
    for name in [
        "ValidatePtr2",
        "GetMediaItemTake_Item",
        "GetMediaItemInfo_Value",
        "GetMediaItemTakeInfo_Value",
        "GetSetMediaItemInfo_String",
        "GetSetMediaItemTakeInfo_String",
        "GetTakeNumStretchMarkers",
        "GetTakeStretchMarker",
        "GetTakeStretchMarkerSlope",
        "GetProjectStateChangeCount",
    ] {
        let f = Fixture::new();
        f.missing.set(Some(name));
        let client = f.client();
        f.reset();
        assert!(f.geometry(&client).unwrap_err().contains("API unavailable"));
        assert_eq!(client.sample(|| f.valid.get()).unwrap(), (2.5, true));
    }
    let f = Fixture::new();
    f.no_take.set(true);
    let client = f.client();
    assert!(f.geometry(&client).unwrap_err().contains("parent take"));
    assert!(client.sample(|| f.valid.get()).is_ok());
    f.no_take.set(false);
    for kind in ["ReaProject*", "MediaItem_Take*", "MediaItem*"] {
        f.bad_type.set(Some(kind));
        assert!(f.geometry(&client).is_err());
    }
    drop(client);
    let f = Fixture::new();
    f.missing.set(Some("GetPlayStateEx"));
    let client = f.client();
    assert!(client.sample(|| f.valid.get()).is_err());
    assert!(f.geometry(&client).is_ok());
}

#[test]
fn task38a_bad_fields_guid_and_marker_count_are_bounded_and_explicit() {
    for (name, value) in [
        ("D_POSITION", f64::NAN),
        ("D_LENGTH", 0.),
        ("D_PLAYRATE", 0.),
        ("B_PPITCH", 2.),
        ("I_CHANMODE", 1.5),
        ("D_FADEINLEN", -1.),
    ] {
        let f = Fixture::new();
        f.values.borrow_mut().insert(name, value);
        let client = f.client();
        assert!(f.geometry(&client).is_err(), "{name}");
    }
    let f = Fixture::new();
    let client = f.client();
    f.bad_guid.set(true);
    assert!(f.geometry(&client).is_err());
    f.bad_guid.set(false);
    for count in [-1, MAX_MARKERS + 1, i32::MAX] {
        f.count_override.set(Some(count));
        f.reset();
        assert!(f.geometry(&client).is_err());
        assert!(!f.calls.borrow().iter().any(|s| s.starts_with("marker:")));
    }
    f.count_override.set(None);
    f.bad_marker.set(true);
    assert!(f.geometry(&client).is_err());
    f.bad_marker.set(false);
    f.markers.borrow_mut()[0].2 = f64::INFINITY;
    assert!(f.geometry(&client).is_err());
    *f.markers.borrow_mut() = vec![(f64::NAN, 0., 0.)];
    assert!(f.geometry(&client).is_err());
    *f.markers.borrow_mut() = vec![(0., f64::NAN, 0.)];
    assert!(f.geometry(&client).is_err());
    f.markers.borrow_mut().clear();
    assert!(f.geometry(&client).unwrap().markers.is_empty());
    *f.markers.borrow_mut() = vec![(0., 0., 0.); MAX_MARKERS as usize];
    assert_eq!(
        f.geometry(&client).unwrap().markers.len(),
        MAX_MARKERS as usize
    );
}

/// 负速率是宿主的倒放信号，不是非法值：几何保留**幅值**而不是丢掉整个 item。
///
/// 【为什么单独钉住】此前 `playback_rate <= 0.` 直接报错 ⇒ 该 item 从宿主清单里
/// 消失，比"识别成倒放"更糟（前端看不到它，也无法在它上面编辑）。方向由
/// `PCM_Source_GetSectionInfo` 的方向位单独表达，与这里的速率符号无关；与独立 App
/// 的 `reaper_import.rs`（`raw_play_rate.abs().max(0.01)`）同口径。
#[test]
fn a_negative_play_rate_is_normalized_instead_of_dropping_the_item() {
    let f = Fixture::new();
    f.values.borrow_mut().insert("D_PLAYRATE", -2.);
    let client = f.client();
    let geometry = f
        .geometry(&client)
        .expect("a negative rate must not drop the item");
    assert_eq!(geometry.playback_rate, 2.);
}

#[test]
fn task38a_each_raw_geometry_getter_reentry_stops_before_the_next_getter() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    f.geometry(&client).unwrap();
    let count = f.calls.borrow().len();
    for index in 1..=count {
        f.reset();
        f.revoke_at.set(index);
        assert!(f.geometry(&client).is_err(), "getter {index}");
        assert_eq!(f.calls.borrow().len(), index);
    }
    f.reset();
    f.revoke_at.set(1);
    assert!(client.sample(|| f.valid.get()).is_err());
    assert_eq!(&*f.calls.borrow(), &["validate:ReaProject*"]);
}

#[test]
fn task38a_project_change_integer_discards_whole_snapshot_without_retry() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    f.geometry(&client).unwrap();
    let baseline = f.calls.borrow().clone();
    let index = baseline.iter().position(|v| v == "D_LENGTH").unwrap() + 1;
    f.reset();
    f.change_at.set(index);
    assert!(f.geometry(&client).unwrap_err().contains("project changed"));
    assert_eq!(*f.calls.borrow(), baseline);
    f.reset();
    f.change.set(i32::MIN);
    assert!(
        f.geometry(&client).is_ok(),
        "integer is compared for equality, not monotonicity/sign"
    );
}

#[test]
fn task38a_native_take_invalidation_is_checked_before_another_take_getter() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    *f.hook.borrow_mut() = Some((
        "item_guid".into(),
        Box::new(|| {
            fixture().bad_type.set(Some("MediaItem_Take*"));
        }),
    ));
    assert!(f.geometry(&client).is_err());
    assert!(
        !f.calls.borrow().contains(&"take_guid".into()),
        "owning extension reference does not keep a deleted take alive"
    );
}

#[test]
fn task38a_query_initialization_reentry_balances_qi_and_stops_lookup() {
    let f = Fixture::new();
    let client = f.client();
    let count = f.calls.borrow().len();
    drop(client);
    for index in 1..=count {
        f.reset();
        f.revoke_at.set(index);
        assert!(unsafe { NativeHost::from_context(f.context(), || f.valid.get()) }.is_none());
        assert_eq!(f.calls.borrow().len(), index);
        assert_eq!(f.refs.load(Ordering::Acquire), 1);
    }
}

/// 转发层按"另一套"字节比较 IID 时，原生扩展仍要拿得到。
///
/// 【为什么值得一条用例】`platform_iid` 解决的是"正常宿主"；兜底重试解决的是
/// "宿主被包装过"（Wine / CrossOver / wrapper 进程）。两条路径都要有断言，
/// 否则兜底那段代码只是看起来存在。
#[test]
fn a_host_answering_in_the_alternate_layout_is_still_found() {
    let f = Fixture::new();
    f.force_alternate_iid_layout();
    let host = unsafe { NativeHost::from_context(f.context(), || true) };
    assert!(
        host.is_some(),
        "the alternate IID layout must be retried instead of giving up"
    );
}

#[test]
fn task38a_geometry_and_transport_reject_worker_threads_without_host_calls() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(client.geometry(|| true).is_err());
                assert!(client.sample(|| true).is_err());
            })
            .join()
            .unwrap();
    });
    assert!(f.calls.borrow().is_empty());
    f.valid.set(false);
    assert!(f.geometry(&client).is_err());
    assert!(f.calls.borrow().is_empty());
}

/// take FX 夹具：轨道链**为空**，take 链上有本插件与一个原生 FX。
fn take_fx_fixture() -> &'static Fixture {
    let fixture = Fixture::new();
    fixture.enable_media();
    fixture.fx_api.set(true);
    fixture.take_fx_api.set(true);
    *fixture.fx_take_chain.borrow_mut() = vec![FxEntry::HFS, FxEntry::OTHER];
    fixture
}

/// **轨道链的整链开关（`I_FXEN`）仍然会关门。**
///
/// 【为什么与 take 侧成对存在】take 侧修的是"读了一个**不存在**的读数"，不是"整链开关不
/// 重要"。轨道侧那个读数是真的：用户把整条轨道 FX 链关掉时插件必须停下，否则白烧 CPU
/// —— 而这条路径的判据与 take 侧共用同一份 `decide()`，所以它必须一直有覆盖。
#[test]
fn a_track_chain_toggle_still_closes_the_gate() {
    let f = Fixture::new();
    f.enable_media();
    f.fx_api.set(true);
    *f.fx_track_chain.borrow_mut() = vec![FxEntry::HFS];
    let host = std::sync::Arc::new(f.client());
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Active);
    f.fx_chain_on.set(false);
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Bypassed);
    f.fx_chain_on.set(true);
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Active);
}

/// **take FX 实例必须读 take 链，而不是轨道链。**
///
/// 【为什么这条是本次修复的主验收】改动前 `fx_bypass_state` 只认 `parent(1)` +
/// `TrackFX_*`：take 链上的实例永远匹配不到（`found == 0` ⇒ `Unknown`），于是
/// "旁通了还在渲染"在 take FX 下**必然复发**。这条用例同时钉住两个方向：
/// 用户旁通 take 链上的那个 FX ⇒ 必须报 `Bypassed`；恢复 ⇒ 必须报 `Active`。
#[test]
fn a_take_fx_instance_reads_the_take_chain_not_the_track_chain() {
    let f = take_fx_fixture();
    let host = std::sync::Arc::new(f.client());
    assert_eq!(
        host.resolve_placement(&[], &|| true),
        HostPlacement::TakeFx {
            track: f.track() as usize,
            item: f.item() as usize,
            take: f.take() as usize,
        }
    );
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Active);
    // 用户旁通 take 链上的本插件。
    f.fx_take_chain.borrow_mut()[0].enabled = false;
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Bypassed);
    // 恢复。
    f.fx_take_chain.borrow_mut()[0].enabled = true;
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Active);
    // 【绝不按轨道的形状去读 take 的"整链总开关"】take 没有这个参数；宿主对未知参数
    // 回 0.0（夹具照抄这条行为）。旧实现把它读成"链被关掉" ⇒ 判旁通 ⇒ 一点声音都没有，
    // 而这一整段用例从头到尾都能跑通 —— 判据本身不认识"没有这个读数"，那才是根因。
}

/// **轨道 FX 实例绝不读同轨另一条 take 链。**
///
/// 【为什么这是反方向的同一条边界】同一条轨上可以同时有轨道 FX 实例与 take FX 实例。
/// 若轨道实例越过自己的链去看 take 链，它会把"某个 take 上的实例被旁通"当成自己
/// 被旁通 —— 用户听不到声音，而日志里只有一条 `active -> bypassed`。
#[test]
fn a_track_fx_instance_never_reads_a_sibling_take_chain() {
    let f = Fixture::new();
    f.enable_media();
    f.fx_api.set(true);
    f.take_fx_api.set(true);
    // 轨道链上的本插件被旁通；take 链上的那个是启用的。
    *f.fx_track_chain.borrow_mut() = vec![FxEntry {
        enabled: false,
        ..FxEntry::HFS
    }];
    *f.fx_take_chain.borrow_mut() = vec![FxEntry::HFS];
    let host = std::sync::Arc::new(f.client());
    // region 落在 item 之外 ⇒ 没有"这组 region 全在 take 里"的证据 ⇒ 判轨道 FX。
    assert_eq!(
        host.resolve_placement(&[(0.0, 0.5)], &|| true),
        HostPlacement::TrackFx {
            track: f.track() as usize
        }
    );
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Bypassed);
    // 反过来：轨道链启用、take 链旁通，轨道实例仍报活跃（没有被 take 链带偏）。
    f.fx_track_chain.borrow_mut()[0].enabled = true;
    f.fx_take_chain.borrow_mut()[0].enabled = false;
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Active);
}

/// 两条链都有本插件、且 region 全在 take 的 item 内 ⇒ 判 take FX（歧义裁决的正例）。
#[test]
fn both_chains_are_disambiguated_by_the_bound_item_span() {
    let f = take_fx_fixture();
    *f.fx_track_chain.borrow_mut() = vec![FxEntry::HFS];
    let host = std::sync::Arc::new(f.client());
    // item 的区间是 [1, 5)（夹具的 D_POSITION / D_LENGTH）。
    assert_eq!(
        host.resolve_placement(&[(1.0, 4.0), (4.0, 5.0)], &|| true),
        HostPlacement::TakeFx {
            track: f.track() as usize,
            item: f.item() as usize,
            take: f.take() as usize,
        }
    );
    // 有一条 region 落在 item 之外 ⇒ 退回轨道 FX。
    assert_eq!(
        host.resolve_placement(&[(1.0, 4.0), (5.5, 9.0)], &|| true),
        HostPlacement::TrackFx {
            track: f.track() as usize
        }
    );
}

/// 宿主不提供 `TakeFX_*` 一族时，take 实例必须报 `Unknown`，**不得**回退去读轨道链。
#[test]
fn a_take_fx_instance_without_the_take_api_stays_unknown() {
    let f = Fixture::new();
    f.enable_media();
    f.fx_api.set(true);
    // take 侧 API 缺席（老宿主形态），轨道链上有一个**已旁通**的本插件。
    *f.fx_track_chain.borrow_mut() = vec![FxEntry {
        enabled: false,
        ..FxEntry::HFS
    }];
    let host = std::sync::Arc::new(f.client());
    assert_eq!(host.fx_bypass_state(&|| true), FxBypass::Bypassed);
    // 同一条链缺席时的手动确认：探不到 take 链 ⇒ 只可能落在轨道或 Unbound。
    assert!(host.probe_placement(&|| true).take.is_none());
}

/// 放置位置在重绑后必须**重新求值**，不能把上一次的结论带到下一次。
#[test]
fn the_placement_is_recomputed_after_a_rebind() {
    let f = take_fx_fixture();
    let host = std::sync::Arc::new(f.client());
    assert!(host.resolve_placement(&[], &|| true).is_take_fx());
    assert!(host.placement_resolved());
    // 重绑：适配器丢掉旧结论，宿主把同一个处理器搬到另一条链上。
    host.forget_placement();
    f.forget_and_rearm_as_track_fx();
    assert_eq!(host.placement(), HostPlacement::Unbound);
    assert!(!host.placement_resolved());
    assert_eq!(
        host.resolve_placement(&[], &|| true),
        HostPlacement::TrackFx {
            track: f.track() as usize
        }
    );
}

/// take FX 下"目标 item 就是本实例挂着的那个 item"必须被认出来。
///
/// 【为什么这是一个断言而不是注释】它是"方向/包络不重写 item chunk"的**唯一**判据：
/// 认错了，倒放就会去撞那次必然被拒的 chunk 写回（然后才退到动作）；认反了，轨道 FX
/// 实例会把别人的 item 当成自己的、绕开精确的 chunk 路径。
#[test]
fn a_take_fx_item_is_recognised_as_the_one_hosting_this_instance() {
    let f = take_fx_fixture();
    f.enable_item_state(&f.single_take_chunk());
    let host = std::sync::Arc::new(f.client());
    assert_eq!(host.placement().item(), None, "还没求值前不许瞎认");
    let target = host.clip_target(|| true).expect("bound take");
    assert!(!target.hosts_this_instance(), "Unbound 时按历史路径走");
    assert!(host.resolve_placement(&[], &|| true).is_take_fx());
    assert!(
        target.hosts_this_instance(),
        "take FX 下目标 item 就是本实例挂着的那个"
    );
    // 轨道 FX 的实例不拥有任何 item。
    f.forget_and_rearm_as_track_fx();
    host.forget_placement();
    assert!(host.resolve_placement(&[], &|| true).track().is_some());
    let target = host.clip_target(|| true).expect("bound take");
    assert!(!target.hosts_this_instance());
}

/// take FX 的清单只有**自己那条 take 的 item**，且写回目标绑在那条 take 上。
///
/// 【为什么非 active take 是关键】take FX 恰好最适合挂在多 take item 的某一条上，而
/// `GetActiveTake` 恒指向**宿主选中**的那条。绑定一旦取错，写回（音高/方向/淡变）
/// 全部落到别人的 take。
#[test]
fn a_take_fx_inventory_lists_only_the_bound_take() {
    let f = take_fx_fixture();
    f.enable_takes(2);
    f.parent_take.set(f.take_at(2) as usize);
    let host = std::sync::Arc::new(f.client());
    assert!(host.resolve_placement(&[], &|| true).is_take_fx());
    let tracks = host.ui_scope(&|| true, &[]).unwrap();
    assert_eq!(tracks.len(), 1, "take FX 只属于一条轨道");
    assert_eq!(tracks[0].items.len(), 1, "take FX 只列自己那个 item");
    let item = &tracks[0].items[0];
    assert_eq!(item.takes.len(), 3, "全部 take 仍然供界面展示");
    // 绑定的是第 3 条 take（非 active），而几何与写回目标都跟着它走。
    assert_eq!(
        item.bound_take.as_deref(),
        Some("{55555555-5555-5555-5555-000000000002}")
    );
    assert_eq!(
        item.geometry.take_id,
        "{55555555-5555-5555-5555-000000000002}"
    );
    assert_eq!(
        item.takes.iter().filter(|take| take.active).count(),
        1,
        "active 标记仍然且只有一个"
    );
}

/// take FX 下导入 / 建轨 / 剪贴板这三个入口必须关掉，其余能力不受影响。
#[test]
fn a_take_fx_instance_never_offers_track_level_entries() {
    let f = Fixture::new();
    f.enable_media();
    f.enable_split();
    f.enable_item_state(&f.single_take_chunk());
    let host = std::sync::Arc::new(f.client());
    let track_fx = host.capabilities_for(HostPlacement::TrackFx { track: 1 });
    assert!(track_fx.media_import, "轨道 FX 下导入仍然可用");
    assert!(track_fx.track_create, "轨道 FX 下建轨仍然可用");
    assert!(track_fx.item_clipboard, "轨道 FX 下剪贴板仍然可用");
    let take_fx = host.capabilities_for(HostPlacement::TakeFx {
        track: 1,
        item: 2,
        take: 3,
    });
    assert!(!take_fx.media_import, "take FX 下导入的目标对象不成立");
    assert!(!take_fx.track_create, "take FX 下建轨的目标对象不成立");
    assert!(!take_fx.item_clipboard, "复制 item 会连带复制出第二个实例");
    // 写能力本身不受影响：take FX 照样能拖片段、切分、写音高包络。
    assert!(take_fx.clip_write);
    assert!(take_fx.split);
    assert!(take_fx.take_pitch_bridge);
    assert!(take_fx.clip_reverse);
}

/// 有包络对象时，音高写回**只**碰包络，绝不重写 item chunk。
///
/// 【为什么这条是 take FX 的核心】真机实测：挂着本插件（ARA）的 take FX 的那个 item，
/// `SetItemStateChunk` 连"原样写回"都拒绝。包络对象那条路不受这条限制 —— 而且它不重建
/// item，轨道 FX 下也顺带避免了"写一次包络、把同一条 item 上的别的 take FX 重建一遍"。
#[test]
fn a_take_pitch_write_uses_the_envelope_object_when_it_exists() {
    let f = Fixture::new();
    f.enable_item_state(&f.single_take_chunk());
    f.inventory_enabled.set(true);
    f.take_env_enabled.set(true);
    f.take_env_chunk.replace(Some(
        "<PITCHENV\nACT 1 -1\nVIS 1 1 1\nLANEHEIGHT 0 0\nARM 0\nDEFSHAPE 0 -1 -1\nPT 0 0 0\n>\n"
            .to_string(),
    ));
    let host = std::sync::Arc::new(f.client());
    let target = host.clip_target(|| true).expect("bound take");
    let points = vec![
        hifishifter_kernel::pitch_bridge::EnvelopePoint::new(0.0, 3.0),
        hifishifter_kernel::pitch_bridge::EnvelopePoint::new(1.5, -2.0),
    ];
    target
        .write_take_pitch_points(&points, &|| true)
        .expect("write through the envelope");
    assert!(
        !f.calls().iter().any(|call| call == "set-item-state"),
        "不得重写 item chunk"
    );
    assert!(f.calls().iter().any(|call| call == "set-envelope-state"));
    // 读回也走包络对象，且点集逐字一致。
    let read = target.take_pitch_points(&|| true).expect("read");
    assert_eq!(read.points, points);
    assert!(read.present);
}

/// 没有包络时**先**借宿主动作建一条，而不是先去撞 item chunk。
///
/// 【为什么顺序本身是断言】真机上那条"先写 chunk"的路在 take FX 下必然失败：每次转换都要
/// 读一份几百 KB 的 item chunk、再吃一次宿主拒绝，然后才建包络 —— 用户看到的是几秒卡顿
/// 加一条吓人的日志。所以这里不仅要求结果对，还要求**根本没去碰** item chunk。
#[test]
fn a_missing_envelope_is_created_through_the_host_action_first() {
    let f = Fixture::new();
    f.enable_item_state(&f.single_take_chunk());
    f.inventory_enabled.set(true);
    f.take_env_enabled.set(true);
    // 宿主此刻确实会拒绝 chunk 写回（take FX 下的真机行为）—— 但这条路径不该走到那里。
    f.item_state_write_refused.set(true);
    let host = std::sync::Arc::new(f.client());
    let target = host.clip_target(|| true).expect("bound take");
    let points = vec![hifishifter_kernel::pitch_bridge::EnvelopePoint::new(
        0.0, 5.0,
    )];
    target
        .write_take_pitch_points(&points, &|| true)
        .expect("write through a created envelope");
    assert!(
        f.calls().iter().any(|call| call == "run-command-ex:41612"),
        "必须借宿主自己的动作建包络"
    );
    assert!(
        !f.calls().iter().any(|call| call == "get-item-state"),
        "建包络不该先读一份 item chunk"
    );
    let stored = f.take_env_chunk.borrow().clone().expect("envelope created");
    assert!(stored.contains("PT 0.00000000000000000 5.00000000000000000 0"));
    // 选择位必须恢复到进入时的样子（动作只该影响目标 item，不该改用户的选区）。
    assert!(!f.item_ui_selected.get(), "选择位要还原");
}

/// **按名字解析动作号，不认死号码**：夹具把命令号改成 99999，实现仍要跑对。
///
/// 【为什么这条是回归】真机上第一次尝试"借宿主动作"时，枚举函数的出参被当成字符缓冲区
/// 读，名字永远匹配不上 —— 用户看到的是"宿主里找不到该动作，请手动打开一次音高包络"。
/// 那时日志里其实连动作表都没读到。判据必须是"名字 → 命令号"，而不是记忆里的数字。
#[test]
fn the_action_is_resolved_by_name_not_by_a_remembered_number() {
    let f = Fixture::new();
    f.enable_item_state(&f.single_take_chunk());
    f.inventory_enabled.set(true);
    f.take_env_enabled.set(true);
    f.pitch_action_id.set(99_999);
    let host = std::sync::Arc::new(f.client());
    let target = host.clip_target(|| true).expect("bound take");
    let points = vec![hifishifter_kernel::pitch_bridge::EnvelopePoint::new(
        0.0, 1.0,
    )];
    target
        .write_take_pitch_points(&points, &|| true)
        .expect("write through the enumerated action");
    assert!(
        f.calls().iter().any(|call| call == "run-command-ex:99999"),
        "用的是动作表里的命令号：{:?}",
        f.calls()
    );
    assert!(f.take_env_chunk.borrow().is_some());
}

/// 空点集 = 删掉整条包络，而且**不碰** item chunk。
#[test]
fn an_empty_point_set_clears_the_envelope_without_touching_the_item() {
    let f = Fixture::new();
    f.enable_item_state(&f.single_take_chunk());
    f.inventory_enabled.set(true);
    f.take_env_enabled.set(true);
    f.take_env_chunk.replace(Some(
        "<PITCHENV\nACT 1 -1\nVIS 1 1 1\nLANEHEIGHT 0 0\nARM 0\nDEFSHAPE 0 -1 -1\nPT 0 0 0\n>\n"
            .to_string(),
    ));
    let host = std::sync::Arc::new(f.client());
    let target = host.clip_target(|| true).expect("bound take");
    target
        .write_take_pitch_points(&[], &|| true)
        .expect("clear");
    assert!(
        f.take_env_chunk.borrow().is_none(),
        "只有头的块 = 宿主删掉整条包络"
    );
    assert!(!f.calls().iter().any(|call| call == "set-item-state"));
}

/// **超过第一次读取缓冲区的 item chunk 必须整份读回、整份写回。**
///
/// 【为什么这条是"take 音高转换永远失败"的回归】`GetItemStateChunk` 能写多少由调用方给的
/// 缓冲区决定：放不下时宿主返回 `false`（真机也把前缀写满后照常返回 `true`）。旧实现
/// 第一次就只要 64 KiB，且把"读到 NUL 为止"当成读完 —— 于是一份带几千个 `PITCHENV` 点
/// （长片段）或带着 take FX 插件状态的 chunk 永远读不全，每一次转换都在解析阶段失败，
/// 而用户看到的只是"点了没反应"。判据必须是结构（能不能解析成一整份 `<ITEM>`），
/// 不是字节数猜测。
#[test]
fn a_chunk_larger_than_the_first_read_is_read_whole() {
    let f = Fixture::new();
    f.enable_media();
    let mut chunk = String::from(
        "<ITEM\nPOSITION 0\nLENGTH 1\nSEL 1\n\
         IGUID {11111111-1111-1111-1111-111111111111}\n\
         GUID {22222222-2222-2222-2222-222222222222}\n\
         <SOURCE WAVE\nFILE \"ara://source\"\n>\n<PITCHENV\nACT 1 -1\n",
    );
    for index in 0..6000 {
        chunk.push_str(&format!("PT {:.17} {:.17} 0\n", index as f64 * 0.01, 0.0));
    }
    chunk.push_str(">\n>\n");
    assert!(chunk.len() > 64 * 1024, "chunk is {} bytes", chunk.len());
    f.enable_item_state(&chunk);
    let host = std::sync::Arc::new(f.client());
    let target = host.clip_target(|| true).expect("bound take");
    // 读：旧实现在第一次（64 KiB）就停手，把半份 chunk 当整份交出去。
    let read = target.item_state_chunk(&|| true).expect("whole chunk");
    assert_eq!(read.len(), chunk.len());
    // 写：整块替换后仍然是一份完整 chunk，新点集在里面。
    let points = vec![
        hifishifter_kernel::pitch_bridge::EnvelopePoint::new(0.0, 3.0),
        hifishifter_kernel::pitch_bridge::EnvelopePoint::new(1.0, -2.0),
    ];
    target
        .write_take_pitch_points(&points, &|| true)
        .expect("write");
    let written = target.item_state_chunk(&|| true).expect("reread");
    let parsed =
        super::take_pitch::read_take_pitch(&written, "{22222222-2222-2222-2222-222222222222}")
            .expect("parse");
    assert_eq!(parsed.points, points);
}

/// **多 take item 分割后，右半段的写回目标仍是"同一条 take"。**
///
/// 【为什么这是本轮必须钉住的一条】`SplitMediaItem` 之后旧代码取 `GetActiveTake(right)`：
/// 那是宿主**选中**的 take，而 take FX 恰好可以挂在非 active 的那条上。一旦取错，
/// 右半段的音高/方向/淡变全部落到别人的 take 上 —— 用户看到的是"分割一次，另一条 take
/// 的音高变了"。
#[test]
fn a_multi_take_split_keeps_the_bound_take() {
    let f = Fixture::new();
    f.enable_split();
    f.enable_takes(1);
    // 实例挂在第 2 条 take 上（不是 active 的第 1 条）。
    f.parent_take.set(f.take_at(1) as usize);
    let host = std::sync::Arc::new(f.client());
    let target = host.clip_target(|| true).expect("bound take");
    let (left, right) = target.split_at(2.0, &|| true).expect("split");
    assert_eq!(left.item_id, "{11111111-1111-1111-1111-111111111111}");
    // 右半段的身份来自"同一序号"，不是 active take。
    assert_eq!(right.take_id, "{44444444-4444-4444-4444-444444444444}");
    assert_eq!(right.item_id, "{33333333-3333-3333-3333-333333333333}");
    // 两段都真的写回宿主（update-item 各一次）。
    assert_eq!(
        f.calls()
            .iter()
            .filter(|call| call.as_str() == "update-item")
            .count(),
        2
    );
}
