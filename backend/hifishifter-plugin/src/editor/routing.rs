//! 经真实VST3 connection消息关联实例；只接受本进程登记的活令牌，不猜最后一轨。
use crate::editor::connection::UnknownVtbl;
use crate::render::extension::ExtensionOwner;
use crate::vst3::{platform_iid, TResult, K_RESULT_OK};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
fn routes() -> &'static Mutex<HashMap<String, Weak<ExtensionOwner>>> {
    static ROUTES: OnceLock<Mutex<HashMap<String, Weak<ExtensionOwner>>>> = OnceLock::new();
    ROUTES.get_or_init(Default::default)
}
/// 进程内**全部仍存活**的编辑器链路（心跳遍历用）。
///
/// 【为什么需要一张表】宿主侧周期工作（时钟采样、宿主 Undo 收口、`setDirty` 冲刷）
/// 必须跑在宿主主线程上，而它要的是一组 `EditorLink` —— 在独立进程形态（情形 B）里，
/// 调用方手里没有客户端状态（那是 GTK 线程的 `Rc<RefCell<…>>`，不能跨线程）。
/// 这张表只存 `Weak`：登记与回收都由 `EditorLink` 自己负责，不留悬空引用。
fn links() -> &'static Mutex<Vec<Weak<EditorLink>>> {
    static LINKS: OnceLock<Mutex<Vec<Weak<EditorLink>>>> = OnceLock::new();
    LINKS.get_or_init(Default::default)
}

/// 仍在存活的全部**组件 owner**（心跳用）。
///
/// 【为什么还要这一条】链路（`EditorLink`）要等编辑器视图建立、并且前端把
/// 租约 `bind` 上来之后才拿得到 owner；而宿主侧周期工作只需要 owner。
/// 只依赖链路，就会出现「心跳在跑、却一个 owner 都没遍历到」——采样频率掉到
/// 只有命令驱动的那几次，宿主读数因此反复过期。
pub(crate) fn live_owners() -> Vec<Arc<ExtensionOwner>> {
    let mut all = routes().lock().unwrap_or_else(|e| e.into_inner());
    all.retain(|_, owner| owner.strong_count() > 0);
    let mut owners: Vec<Arc<ExtensionOwner>> = Vec::new();
    for owner in all.values().filter_map(Weak::upgrade) {
        // 【为什么剔除已关闭的实例】它们仍可能被某个 Arc 短暂持有（宿主正在销毁），
        // 但已经不能服务任何命令。把它们算进「唯一活实例」会让兜底绑定瞄准一具尸体，
        // 也会让心跳去采样一个已关闭的时钟。
        if owner.is_closed() {
            continue;
        }
        if owners
            .iter()
            .any(|known| std::ptr::eq(Arc::as_ptr(known), Arc::as_ptr(&owner)))
        {
            continue;
        }
        owners.push(owner);
    }
    owners
}

/// 仍在存活的全部链路；顺带回收已销毁的 `Weak`（表因此不会无限增长）。
pub(crate) fn live_links() -> Vec<Arc<EditorLink>> {
    let mut all = links().lock().unwrap_or_else(|e| e.into_inner());
    all.retain(|link| link.strong_count() > 0);
    all.iter().filter_map(Weak::upgrade).collect()
}

pub(crate) struct RouteLease {
    token: String,
}
impl RouteLease {
    /// 登记组件自己的owner；租约析构即撤销，额外Arc不能让已销毁组件再次被连接。
    pub fn new(owner: &Arc<ExtensionOwner>) -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(1);
        let input = format!(
            "{}-{}-{:?}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
        );
        let token = blake3::hash(input.as_bytes()).to_hex().to_string();
        routes()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(token.clone(), Arc::downgrade(owner));
        Self { token }
    }
    pub fn token(&self) -> &str {
        &self.token
    }
}
impl Drop for RouteLease {
    fn drop(&mut self) {
        routes()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.token);
    }
}
/// 编辑器路由是**怎么**建立的。
///
/// 【为什么需要区分】两者对用户都「能用」，但诊断含义完全相反：握手成功说明宿主的
/// VST3 实现完整；兜底成功说明宿主跳过了 IConnectionPoint（或拒绝了我们发的消息）。
/// 把它们混成一个「已绑定」布尔，报障时就再也分不清「宿主变了」与「我们退化了」。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RouteOrigin {
    /// 宿主完成了 VST3 的 `IConnectionPoint` 握手。
    HostHandshake,
    /// 宿主没有完成握手；按「进程内唯一活实例」兜底绑定（见 [`bind_sole_owner`]）。
    SoleOwnerFallback,
}

/// 路由状态（诊断包、`plugin_health` 与前端共用的四态）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RouteState {
    /// 视图刚建立，握手结果还没出来 —— 这不是错误。
    WaitingHost,
    /// 由宿主握手绑定。
    Bound,
    /// 由兜底绑定。
    Fallback,
    /// 明确失败（原因见 [`EditorLink::route_failure`]）。
    Unbound,
}

impl RouteState {
    /// 稳定的线格式（跨语言、跨版本都只读这一份）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RouteState::WaitingHost => "waiting-host",
            RouteState::Bound => "bound",
            RouteState::Fallback => "fallback",
            RouteState::Unbound => "unbound",
        }
    }
}

#[derive(Default)]
struct Binding {
    token: Option<String>,
    generation: u64,
    /// 只在 `token` 存在时有意义。
    origin: Option<RouteOrigin>,
    /// 兜底失败的原因；只写一次（`ui_tick` 每 20 ms 跑一次）。
    failure: Option<String>,
}
const IID_COMPONENT_HANDLER2: [u32; 4] = [0xF040B4B3, 0xA36045EC, 0xABCDC045, 0xB4D5A2CC];
#[repr(C)]
struct Handler2Vtbl {
    base: UnknownVtbl,
    set_dirty: unsafe extern "system" fn(*mut c_void, u8) -> TResult,
}
#[derive(Default)]
pub(crate) struct EditorLink {
    binding: Mutex<Binding>,
    handler2: Mutex<usize>,
    dirty: std::sync::atomic::AtomicBool,
}
fn release_handler(pointer: usize) {
    if pointer != 0 {
        unsafe {
            let pointer = pointer as *mut c_void;
            ((**pointer.cast::<*const UnknownVtbl>()).release)(pointer);
        }
    }
}
impl EditorLink {
    /// 登记到心跳表（`Arc` 建成后调用一次）。重复登记同一链路不会重复计数。
    pub(crate) fn register_heartbeat(self: &Arc<Self>) {
        let mut all = links().lock().unwrap_or_else(|e| e.into_inner());
        if all
            .iter()
            .any(|link| std::ptr::eq(link.as_ptr(), Arc::as_ptr(self)))
        {
            return;
        }
        all.push(Arc::downgrade(self));
    }
}
impl Drop for EditorLink {
    fn drop(&mut self) {
        release_handler(*self.handler2.get_mut().unwrap_or_else(|e| e.into_inner()));
    }
}
impl EditorLink {
    /// 保存宿主IComponentHandler2的独立COM引用；不能把裸IComponentHandler指针跨线程调用。
    pub(crate) fn set_component_handler(&self, handler: *mut c_void) -> Result<(), String> {
        let next = if handler.is_null() {
            0
        } else {
            let mut out = std::ptr::null_mut();
            // 【为什么是 platform_iid】宿主按它自己那套字节比较 IID（Windows：COM/GUID；
            // Linux / macOS：每字大端）。写死 COM 布局时，Linux 上这一查恒失败 ⇒
            // `setDirty` 永不调用 ⇒ 用户改完工程、关窗口时宿主不提示保存。
            let iid = platform_iid(IID_COMPONENT_HANDLER2);
            let result = unsafe {
                ((**handler.cast::<*const UnknownVtbl>()).query)(handler, iid.as_ptr(), &mut out)
            };
            if result != K_RESULT_OK || out.is_null() {
                0
            } else {
                out as usize
            }
        };
        let old = {
            let mut current = self.handler2.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::replace(&mut *current, next)
        };
        crate::log_line(&format!(
            "[undo] IComponentHandler2 available={}",
            next != 0
        ));
        release_handler(old);
        Ok(())
    }
    /// actor/UI写入只置位；真实handler回调由插件UI timer调用。
    pub(crate) fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
    pub(crate) fn flush_dirty(&self) {
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return;
        }
        let pointer = *self.handler2.lock().unwrap_or_else(|e| e.into_inner());
        if pointer == 0 {
            return;
        }
        let pointer = pointer as *mut c_void;
        let vtbl = unsafe { &*(*(pointer as *const *const Handler2Vtbl)) };
        let result = unsafe { (vtbl.set_dirty)(pointer, 1) };
        crate::log_line(&format!(
            "[undo] IComponentHandler2::setDirty result={result}"
        ));
    }
    /// PID与令牌都匹配真实本地组件才绑定；宿主代理传递消息但不能替代归属证据。
    pub fn bind(&self, pid: i64, token: &str) -> Result<(), String> {
        if pid != std::process::id() as i64 || token.len() != 64 {
            return Err("editor route is not local".into());
        }
        let owner = routes()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(token)
            .and_then(Weak::upgrade);
        if owner.is_none() {
            return Err("editor route expired or unknown".into());
        }
        let mut binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        binding.generation = binding
            .generation
            .checked_add(1)
            .ok_or("editor route generation exhausted")?;
        binding.token = Some(token.to_owned());
        binding.origin = Some(RouteOrigin::HostHandshake);
        binding.failure = None;
        Ok(())
    }
    pub fn clear(&self) {
        let mut binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        binding.token = None;
        binding.origin = None;
        binding.generation = binding.generation.saturating_add(1);
    }
    /// 每次请求重新验证租约，杜绝组件已销毁但editor旧weak仍可升级的情况。
    pub fn owner(&self) -> Result<Arc<ExtensionOwner>, String> {
        let token = self
            .binding
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .token
            .clone()
            .ok_or("FX editor not connected to its processor yet")?;
        routes()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&token)
            .and_then(Weak::upgrade)
            .filter(|owner| !owner.is_closed())
            .ok_or_else(|| crate::editor::errors::FX_PROCESSOR_CLOSED.error())
    }
    /// 排队和执行各核对一次真实文档与绑定代次；clear/rebind不能为旧请求借新的组件入口。
    pub(super) fn authorize(
        &self,
        document: &Arc<crate::render::document::DocumentSession>,
    ) -> Result<u64, String> {
        let binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        let token = binding
            .token
            .as_ref()
            .ok_or("FX editor not connected to its processor yet")?;
        let owner = routes()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(token)
            .and_then(Weak::upgrade)
            .ok_or(crate::editor::errors::FX_PROCESSOR_CLOSED.error())?;
        if !Arc::ptr_eq(&owner.editor_document()?, document) {
            return Err(crate::editor::errors::ROUTE_OTHER_DOCUMENT.error());
        }
        Ok(binding.generation)
    }
    /// 本绑定的稳定标识（路由 token + 重绑代数）。
    ///
    /// 用于把缓存键绑到"这份文档的这条连接"上 —— 而不是某个会被回收复用的
    /// 指针地址（`Arc::as_ptr` 在文档重开后可能命中陈旧缓存）。
    pub(super) fn route_identity(&self) -> Option<(String, u64)> {
        let binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        Some((binding.token.as_ref()?.clone(), binding.generation))
    }
}
impl EditorLink {
    /// 已经绑定（任一来源）。
    pub(crate) fn is_bound(&self) -> bool {
        self.binding
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .token
            .is_some()
    }

    /// 四态路由状态。
    pub(crate) fn route_state(&self) -> RouteState {
        let binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        match (&binding.token, binding.origin) {
            (Some(_), Some(RouteOrigin::SoleOwnerFallback)) => RouteState::Fallback,
            (Some(_), _) => RouteState::Bound,
            (None, _) if binding.failure.is_some() => RouteState::Unbound,
            (None, _) => RouteState::WaitingHost,
        }
    }

    /// 兜底失败的原因（`unbound` 时才有）。
    pub(crate) fn route_failure(&self) -> Option<String> {
        self.binding
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .failure
            .clone()
    }

    /// 记录一次兜底失败。
    ///
    /// 【为什么只留第一条】调用方是 20 ms 的视图定时器；后到的失败原因几乎总是
    /// 「还是那一个」，覆盖掉最初的现场只会让诊断变模糊。
    pub(crate) fn note_route_failure(&self, reason: String) {
        let mut binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        if binding.failure.is_none() && binding.token.is_none() {
            binding.failure = Some(reason);
        }
    }

    /// 由「进程内唯一活实例」兜底绑定；只由 [`bind_sole_owner`] 调用。
    fn bind_fallback(&self, token: &str) {
        let mut binding = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        binding.generation = binding.generation.saturating_add(1);
        binding.token = Some(token.to_owned());
        binding.origin = Some(RouteOrigin::SoleOwnerFallback);
        binding.failure = None;
    }
}

/// 兜底绑定失败的结构化原因。
///
/// 【为什么是枚举而不是字符串】上层要据此决定「要不要继续等」（`Ambiguous` 是终局，
/// 再等也不会变），而字符串子串匹配是本仓明令禁止的错误分类方式。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum FallbackError {
    /// 进程里没有任何活着的插件实例（视图比组件活得久）。
    NoOwner,
    /// 有多个活实例：映射不唯一，宁可拒绝也不猜。
    Ambiguous(usize),
}

impl std::fmt::Display for FallbackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FallbackError::NoOwner => write!(formatter, "no live plug-in instance"),
            FallbackError::Ambiguous(count) => write!(
                formatter,
                "{count} live plug-in instances; refusing to guess which one owns this editor"
            ),
        }
    }
}

/// 宿主没有完成 `IConnectionPoint` 握手时的**唯一**兜底绑定。
///
/// 【为什么不是「猜最后一个实例」】判据是「进程内恰好只有一个活 owner」：此时映射是
/// **唯一**的，不是猜。多于一个就拒绝并给出可读原因 —— 宁可让用户看到一句明确的话，
/// 也不要把两个实例的编辑串在一起（那会让 A 轨的曲线写进 B 轨）。
///
/// 【为什么可以安全地绑定到「唯一」】`EditorLink::bind` 之后，`owner()` 会再校验
/// owner 未关闭，`authorize` 会再校验文档身份；多实例时我们根本不绑定，
/// 行为与没有这条兜底时完全一致。
pub(crate) fn bind_sole_owner(link: &EditorLink) -> Result<(), FallbackError> {
    if link.is_bound() {
        return Ok(());
    }
    let owners = live_owners();
    let token = sole_token(&owners)?;
    link.bind_fallback(&token);
    Ok(())
}

/// 测试入口：用**给定**的候选集合执行兜底（生产路径读进程级的活实例表）。
#[cfg(test)]
fn bind_sole_owner_with(
    link: &EditorLink,
    candidates: &[Arc<ExtensionOwner>],
) -> Result<(), FallbackError> {
    let token = sole_token(candidates)?;
    link.bind_fallback(&token);
    Ok(())
}

/// 从候选实例里选出唯一的那个并取它的租约令牌（纯函数：不读全局表之外的任何状态）。
///
/// 【为什么单独抽出来】实时判断依赖进程级的 owner 表，而那张表在测试里是共享的、
/// 无法安全地构造「恰好一个」。把「几个候选才叫唯一」抽成纯函数，就能确定性地钉住
/// 0 / 1 / N 三种结论。
fn sole_token(candidates: &[Arc<ExtensionOwner>]) -> Result<String, FallbackError> {
    match candidates {
        [] => Err(FallbackError::NoOwner),
        [only] => token_of(only).ok_or(FallbackError::NoOwner),
        many => Err(FallbackError::Ambiguous(many.len())),
    }
}

/// 反查某个实例的租约令牌。
fn token_of(owner: &Arc<ExtensionOwner>) -> Option<String> {
    let all = routes().lock().unwrap_or_else(|e| e.into_inner());
    all.iter().find_map(|(token, weak)| {
        weak.upgrade()
            .filter(|candidate| Arc::ptr_eq(candidate, owner))
            .map(|_| token.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    /// **不变量：兜底只在映射唯一时成立。**
    ///
    /// 【为什么钉死】这条兜底的全部安全性都建立在「只有一个候选」上。一旦它退化成
    /// 「取最后一个」，多实例宿主里两条轨的编辑就会串在一起 —— 那是最难归因的一类
    /// 数据损坏，而它的起因只是一个看起来无害的便利分支。
    #[test]
    fn the_fallback_binds_only_when_exactly_one_instance_is_alive() {
        assert_eq!(sole_token(&[]), Err(FallbackError::NoOwner));

        let a = Arc::new(ExtensionOwner::default());
        let b = Arc::new(ExtensionOwner::default());
        assert_eq!(
            sole_token(&[Arc::clone(&a), Arc::clone(&b)]),
            Err(FallbackError::Ambiguous(2))
        );

        let lease = RouteLease::new(&a);
        assert_eq!(sole_token(&[Arc::clone(&a)]), Ok(lease.token().to_owned()));

        // 绑定之后 owner() 立刻可用，且状态如实写成 fallback。
        let link = EditorLink::default();
        assert!(!link.is_bound());
        assert_eq!(link.route_state(), RouteState::WaitingHost);
        bind_sole_owner_with(&link, &[Arc::clone(&a)]).unwrap();
        assert!(link.is_bound());
        assert_eq!(link.route_state(), RouteState::Fallback);
        assert!(Arc::ptr_eq(&link.owner().unwrap(), &a));
    }

    /// 兜底失败只记第一条，且不会把已经绑好的链路打回未绑定。
    #[test]
    fn a_failed_fallback_keeps_the_first_reason_and_never_unbinds() {
        let owner = Arc::new(ExtensionOwner::default());
        let lease = RouteLease::new(&owner);
        let link = EditorLink::default();
        link.note_route_failure("first".into());
        link.note_route_failure("second".into());
        assert_eq!(link.route_state(), RouteState::Unbound);
        assert_eq!(link.route_failure().as_deref(), Some("first"));
        link.bind(std::process::id() as i64, lease.token()).unwrap();
        assert_eq!(link.route_state(), RouteState::Bound);
        assert_eq!(link.route_failure(), None);
    }

    #[test]
    fn actual_route_tokens_never_select_another_live_instance() {
        let a = Arc::new(ExtensionOwner::default());
        let b = Arc::new(ExtensionOwner::default());
        let ra = RouteLease::new(&a);
        let rb = RouteLease::new(&b);
        let la = EditorLink::default();
        let lb = EditorLink::default();
        la.bind(std::process::id() as i64, ra.token()).unwrap();
        lb.bind(std::process::id() as i64, rb.token()).unwrap();
        assert!(Arc::ptr_eq(&la.owner().unwrap(), &a));
        assert!(Arc::ptr_eq(&lb.owner().unwrap(), &b));
        assert!(la.bind(-1, rb.token()).is_err());
        assert!(Arc::ptr_eq(&la.owner().unwrap(), &a));
        drop(ra); // a仍有Arc，租约撤销必须独立起效。
        assert!(la.owner().is_err());
        assert!(Arc::ptr_eq(&lb.owner().unwrap(), &b));
        lb.clear();
        assert!(lb.owner().is_err());
    }

    #[repr(C)]
    struct FakeHandler {
        vtbl: *const Handler2Vtbl,
        refs: AtomicU32,
        dirty: AtomicU32,
    }
    unsafe extern "system" fn fake_query(
        this: *mut c_void,
        _iid: *const u8,
        out: *mut *mut c_void,
    ) -> TResult {
        unsafe {
            *out = this;
            (*this.cast::<FakeHandler>())
                .refs
                .fetch_add(1, Ordering::AcqRel);
        };
        K_RESULT_OK
    }
    unsafe extern "system" fn fake_add(this: *mut c_void) -> u32 {
        unsafe {
            (*this.cast::<FakeHandler>())
                .refs
                .fetch_add(1, Ordering::AcqRel)
                + 1
        }
    }
    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        unsafe {
            (*this.cast::<FakeHandler>())
                .refs
                .fetch_sub(1, Ordering::AcqRel)
                - 1
        }
    }
    unsafe extern "system" fn fake_dirty(this: *mut c_void, value: u8) -> TResult {
        unsafe {
            if value != 0 {
                (*this.cast::<FakeHandler>())
                    .dirty
                    .fetch_add(1, Ordering::AcqRel);
            }
        }
        K_RESULT_OK
    }
    static FAKE_HANDLER_VTBL: Handler2Vtbl = Handler2Vtbl {
        base: UnknownVtbl {
            query: fake_query,
            add: fake_add,
            release: fake_release,
        },
        set_dirty: fake_dirty,
    };

    /// handler查询/引用/dirty调用必须都发生在显式UI flush阶段，清理不泄漏宿主引用。
    #[test]
    fn component_handler_dirty_bridge_retains_and_flushes_on_ui_boundary() {
        let handler = Box::new(FakeHandler {
            vtbl: &FAKE_HANDLER_VTBL,
            refs: AtomicU32::new(1),
            dirty: AtomicU32::new(0),
        });
        let pointer = (&*handler as *const FakeHandler) as *mut c_void;
        let link = EditorLink::default();
        link.set_component_handler(pointer).unwrap();
        link.mark_dirty();
        assert_eq!(handler.dirty.load(Ordering::Acquire), 0);
        link.flush_dirty();
        assert_eq!(handler.dirty.load(Ordering::Acquire), 1);
        link.flush_dirty();
        assert_eq!(handler.dirty.load(Ordering::Acquire), 1);
        let _ = link.set_component_handler(std::ptr::null_mut());
        assert!(handler.refs.load(Ordering::Acquire) >= 1);
        drop(link);
        drop(handler);
    }
}
