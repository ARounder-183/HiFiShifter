//! 原生QI/vtable/raw函数回归；fixture不是实际REAPER parent/API返回的验收证据。
use super::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

thread_local! {static ACTIVE:Cell<usize>=const {Cell::new(0)};}
#[repr(C)]
pub(crate) struct Fixture {
    table: *const HostVtbl,
    refs: AtomicU32,
    take_token: u8,
    item_token: u8,
    pub valid: Cell<bool>,
    connection_host: Cell<bool>,
    calls: RefCell<Vec<String>>,
    revoke_at: Cell<usize>,
    change_at: Cell<usize>,
    missing: Cell<Option<&'static str>>,
    no_take: Cell<bool>,
    no_project:Cell<bool>,
    bad_type: Cell<Option<&'static str>>,
    values: RefCell<BTreeMap<&'static str, f64>>,
    markers: RefCell<Vec<(f64, f64, f64)>>,
    count_override: Cell<Option<i32>>,
    bad_marker: Cell<bool>,
    bad_guid: Cell<bool>,
    change: Cell<i32>,
    pub hook: RefCell<Option<(String, Box<dyn Fn()>)>>,
}
impl Fixture {
    pub fn new() -> Box<Self> {
        let value = Box::new(Self {
            table: &TABLE,
            refs: AtomicU32::new(1),
            take_token: 1,
            item_token: 2,
            valid: Cell::new(true),
            connection_host: Cell::new(false),
            calls: RefCell::new(Vec::new()),
            revoke_at: Cell::new(0),
            change_at: Cell::new(0),
            missing: Cell::new(None),
            no_take: Cell::new(false),
            no_project:Cell::new(false),
            bad_type: Cell::new(None),
            values: RefCell::new(BTreeMap::from([
                ("D_POSITION", 1.),
                ("D_LENGTH", 4.),
                ("D_STARTOFFS", 0.),
                ("D_PLAYRATE", 0.5),
                ("B_PPITCH", 1.),
                ("I_CHANMODE", 2.),
                ("D_PITCH", 3.),
                ("C_BEATATTACHMODE", 1.),
                ("C_AUTOSTRETCH", 1.),
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
            hook: RefCell::new(None),
        });
        ACTIVE.set((&*value as *const Self) as usize);
        value
    }
    pub fn context(&self) -> *mut c_void {
        (self as *const Self as *mut Self).cast()
    }
    fn project(&self) -> *mut c_void {
        self.context()
    }
    fn take(&self) -> *mut c_void {
        (&self.take_token as *const u8 as *mut u8).cast()
    }
    fn item(&self) -> *mut c_void {
        (&self.item_token as *const u8 as *mut u8).cast()
    }
    pub fn client(&self) -> ReaperHost {
        unsafe { ReaperHost::from_context(self.context(), || self.valid.get()) }.unwrap()
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
    pub fn references(&self) -> u32 {
        self.refs.load(Ordering::Acquire)
    }
    pub fn enable_connection_host(&self) {
        self.connection_host.set(true);
    }
    pub fn set_value(&self, name: &'static str, value: f64) {
        self.values.borrow_mut().insert(name, value);
        self.change.set(self.change.get().wrapping_add(1));
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
        client: &ReaperHost,
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
    let host_iid = uid_guid([0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5]);
    if requested != uid_guid(IID) && !(f.connection_host.get() && requested == host_iid) {
        unsafe {
            *out = std::ptr::null_mut();
        }
        return crate::vst3::K_NO_INTERFACE;
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
        2 if !f.no_take.get() => f.take(),
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
        _ => std::ptr::null(),
    };
    p as *mut c_void
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
        "MediaItem_Take*" => object == f.take(),
        "MediaItem*" => object == f.item(),
        _ => false,
    }
}
unsafe extern "C" fn item(take: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert_eq!(take, f.take());
    f.record("item");
    f.item()
}
fn value(object: *mut c_void, name: *const c_char, is_take: bool) -> f64 {
    let f = fixture();
    assert_eq!(object, if is_take { f.take() } else { f.item() });
    let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_str().unwrap();
    f.record(name);
    f.values.borrow()[name]
}
unsafe extern "C" fn item_value(object: *mut c_void, name: *const c_char) -> f64 {
    value(object, name, false)
}
unsafe extern "C" fn take_value(object: *mut c_void, name: *const c_char) -> f64 {
    value(object, name, true)
}
fn guid(
    object: *mut c_void,
    name: *const c_char,
    buffer: *mut c_char,
    write: bool,
    is_take: bool,
) -> bool {
    let f = fixture();
    assert_eq!(object, if is_take { f.take() } else { f.item() });
    assert!(!write, "GUID must remain read-only");
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"GUID");
    f.record(if is_take { "take_guid" } else { "item_guid" });
    let text = if f.bad_guid.get() {
        c"bad"
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
    assert_eq!(take, f.take());
    f.record("count");
    f.count_override
        .get()
        .unwrap_or(f.markers.borrow().len() as i32)
}
unsafe extern "C" fn marker(take: *mut c_void, index: i32, pos: *mut f64, src: *mut f64) -> i32 {
    let f = fixture();
    assert_eq!(take, f.take());
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
    assert_eq!(take, f.take());
    f.record(format!("slope:{index}"));
    f.markers.borrow()[index as usize].2
}

#[test]
fn initialization_without_project_retains_interface_and_binds_only_the_later_direct_parent() {
    let f=Fixture::new();f.no_project.set(true);let client=f.client();assert_eq!(f.references(),2);
    f.reset();assert!(client.sample(||f.valid.get()).unwrap_err().contains("not attached"));
    assert_eq!(f.calls(),["parent:3"],"不能把null传给API而暗中使用当前活动project");
    f.no_project.set(false);f.reset();assert_eq!(client.sample(||f.valid.get()).unwrap(),(2.5,true));
    assert!(f.calls().contains(&"parent:3".into()));
    f.reset();client.sample(||f.valid.get()).unwrap();assert!(!f.calls().contains(&"parent:3".into()),"绑定后不跟随活动tab重新选项目");
    drop(client);assert_eq!(f.references(),1);
}
#[test]
fn deferred_project_query_reentry_cannot_bind_or_call_the_next_api() {
    let f=Fixture::new();f.no_project.set(true);let client=f.client();f.no_project.set(false);f.reset();f.revoke_at.set(1);
    assert!(client.sample(||f.valid.get()).is_err());assert_eq!(f.calls(),["parent:3"]);
    assert_eq!(client.project.load(Ordering::Acquire),0,"外部调用撤销许可后不保存刚返回的parent");
    drop(client);assert_eq!(f.references(),1);
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
        assert!(unsafe { ReaperHost::from_context(f.context(), || f.valid.get()) }.is_none());
        assert_eq!(f.calls.borrow().len(), index);
        assert_eq!(f.refs.load(Ordering::Acquire), 1);
    }
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
