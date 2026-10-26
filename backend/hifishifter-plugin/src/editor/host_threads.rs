//! 宿主线程的**身份表**：谁在哪条线程上，以及怎么把请求送到那里。
//!
//! 【为什么需要第二个角色】REAPER 的原生扩展 API 只认它自己的 model/UI 线程
//! （`host/reaper` 的五处守卫 + `host::marshal`）。而 ARA 的播放控制请求还有
//! **第二道同类守卫**，钉在 ARA 的 model 线程上：
//!
//! ```text
//! [cmd] stop_audio failed: invalid state: host playback request off model thread
//! ```
//!
//! 那道守卫在 vendored 的 `ara2-bridge-plugin` 里（三重校验、tree 一脏就拒绝编译），
//! 改不得。唯一正确解是从**我们这一侧**把调用投递回它要求的线程。
//!
//! # 为什么不能复用 `host::marshal::scoped`
//!
//! `scoped` 借用调用方的栈帧，只能在「投递方原地等」的前提下成立。而 ARA 的模型线程
//! **不保证**在我们愿意等的那段时间里回到我们的代码 —— 用户按播放时宿主可能正闲着。
//! 于是「等 250 ms 拿不到就作废」这个形状在这里必然失败，现场日志正是它的指纹：
//!
//! ```text
//! [cmd] play_original failed (client): host thread did not pick the request up within 250 ms
//! [cmd] stop_audio    failed (client): host thread did not pick the request up within 250 ms (3 still queued)
//! ```
//!
//! 括号里的数字一直在涨：任务排进了队列，而**泵不在那条线程上**。现在三件事一起改：
//!
//! 1. 泵装到模型线程上（`ara::model` 的四个回调入口，见那里的说明）；
//! 2. 投递的任务**拥有自己的数据**，因此可以留在队列里等下一次回调（而不是作废）；
//! 3. 等不到时回一个「已排队」的结论，而不是一个错误。
//!
//! # 只允许播放请求
//!
//! 这个队列的目标线程是宿主的模型线程，而模型回调栈里我们**正在**被宿主调用。
//! 让任意闭包排进去，等于允许在那一刻重入模型 —— 那是未定义行为。因此对外的入口
//! 只接受 [`PlaybackRequest`]：三个 ARA 明确规定的「请求」，宿主实现本就应当异步处理。

use ara2_bridge::plugin::PlaybackRequestHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::ThreadId;

/// 一次宿主播放请求。
///
/// 【为什么是枚举而不是闭包】见模块头最后一段：类型把它限死在三个纯宿主请求上，
/// 重入风险从「靠注释约束」变成「编译期不可表达」。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PlaybackRequest {
    /// 请求宿主开始播放。
    Start,
    /// 停止并把播放头放到给定秒 —— `stop_audio` 的完整语义。
    ///
    /// 【为什么合成一个请求】拆成 Stop + SetPosition 会是两次投递、两张票据；
    /// 若两次都排进队列，后一张票会把前一张判成「已被取代」，于是停止被跳过、
    /// 只做了定位 —— 用户按停止，播放却还在继续。
    StopAtPosition(f64),
    /// 请求宿主把播放头移到给定秒。
    SetPosition(f64),
}

impl PlaybackRequest {
    /// 在**模型线程**上执行一次。
    fn execute(self, handle: &PlaybackRequestHandle) -> Result<(), String> {
        match self {
            PlaybackRequest::Start => handle.start().map_err(|error| error.to_string()),
            PlaybackRequest::SetPosition(seconds) => handle
                .set_position(seconds)
                .map_err(|error| error.to_string()),
            PlaybackRequest::StopAtPosition(seconds) => {
                handle.stop().map_err(|error| error.to_string())?;
                handle
                    .set_position(seconds)
                    .map_err(|error| error.to_string())
            }
        }
    }
}

/// 一次模型线程投递的结论。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModelThreadError {
    /// 已排队，但这次没等到执行。
    ///
    /// 任务**不会**被丢弃：它会在宿主下一次模型回调时执行（除非被更新的传输请求取代）。
    /// 调用方据此回一个「已请求宿主」而不是一条红色错误。
    Queued,
    /// 队列满 / 目标线程未知。
    Failed(String),
}

impl ModelThreadError {
    pub(crate) fn message(self) -> String {
        match self {
            ModelThreadError::Queued => {
                "the ARA host has not called back on its model thread yet; the request stays queued"
                    .to_owned()
            }
            ModelThreadError::Failed(reason) => reason,
        }
    }
}

/// 哪一个传输请求是「最新的」。
///
/// 【为什么需要它】请求可以留在队列里等下一次回调。若用户先按播放、随即按停止，
/// 两个请求都会在那一刻执行 —— 后到的那个才代表用户此刻的意图。让旧请求在执行时
/// 发现自己已被取代，就不会出现「按了停止，几秒后却开始播放」。
static LATEST_REQUEST: AtomicU64 = AtomicU64::new(0);

/// 投递方最多等这么久。
///
/// 【为什么不是 250 ms】那个值是为宿主 UI 线程的「一次重排窗口」定的。模型线程的
/// 回调间隔取决于宿主自己在做什么（一次编辑事务里可能几毫秒，空闲时可能几秒）。
/// 两秒是「用户还觉得这是一次同步操作」的上限；超过它就不是慢，而是宿主此刻没有
/// 回调可接 —— 那时把请求留成「已排队」比报错更诚实。
pub(crate) const MODEL_REQUEST_WAIT: std::time::Duration = std::time::Duration::from_millis(2000);

/// 把一次播放请求送到 ARA 的 model 线程。
///
/// 返回 `Ok(Ok(()))` = 宿主接受了请求；`Ok(Err(reason))` = 宿主自己拒绝了它
/// （租约已撤销 / 线程不对 / 位置非法）；`Err(Queued)` = 已排队、稍后执行。
pub(crate) fn on_ara_model(
    target: ThreadId,
    handle: PlaybackRequestHandle,
    request: PlaybackRequest,
) -> Result<Result<(), String>, ModelThreadError> {
    let ticket = LATEST_REQUEST.fetch_add(1, Ordering::AcqRel) + 1;
    deliver(target, move || {
        // 被更新的传输请求取代 ⇒ 这次什么也不做。
        if LATEST_REQUEST.load(Ordering::Acquire) != ticket {
            return Err("superseded by a newer transport request".to_owned());
        }
        request.execute(&handle)
    })
}

/// 在目标线程上执行 `job`：已在该线程就地执行，否则排队并**有界**等待。
///
/// 【为什么任务必须 `'static`】等不到时它要留在队列里，而调用方马上就会离开栈帧。
fn deliver<R: Send + 'static>(
    target: ThreadId,
    job: impl FnOnce() -> R + Send + 'static,
) -> Result<R, ModelThreadError> {
    if std::thread::current().id() == target {
        return Ok(job());
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let posted = crate::host::marshal::post_owned(
        target,
        Box::new(move || {
            // 接收端已经走了（调用方超时）也无所谓：任务本来就必须执行完。
            let _ = sender.send(job());
        }),
    );
    if !posted {
        return Err(ModelThreadError::Failed(
            "the ARA model thread queue is full; the host is not draining it".to_owned(),
        ));
    }
    match receiver.recv_timeout(MODEL_REQUEST_WAIT) {
        Ok(value) => Ok(value),
        Err(_) => Err(ModelThreadError::Queued),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// 队列与「最新请求」都是进程级状态；两条用例并行会互相看见对方的任务。
    fn serialize() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// 已经结束的线程：它的 id 仍然有效，但不会再有泵来取任务。
    fn a_thread_that_never_pumps() -> ThreadId {
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let _ = sender.send(std::thread::current().id());
        })
        .join()
        .unwrap();
        receiver.recv().unwrap()
    }

    /// 起一个扮演模型线程的线程，并让它在 `stop` 之前一直泵队列。
    fn a_pumping_thread() -> (ThreadId, std::sync::Arc<std::sync::atomic::AtomicBool>) {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker_stop = stop.clone();
        std::thread::spawn(move || {
            let _ = sender.send(std::thread::current().id());
            while !worker_stop.load(Ordering::Acquire) {
                crate::host::marshal::pump();
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        (receiver.recv().unwrap(), stop)
    }

    /// **不变量：目标线程不会回到我们的代码时，结论是「已排队」。**
    ///
    /// 【为什么钉死】这里是本次报障的现场：任务排进了一条没有泵的线程，调用方每次
    /// 都拿到一个错误，而用户看到的是「播放不能用」。降级成「已排队」之后，同一个
    /// 请求会在宿主下一次模型回调时真的执行。
    #[test]
    fn a_request_for_a_thread_that_never_pumps_stays_queued() {
        let _guard = serialize();
        let target = a_thread_that_never_pumps();
        let outcome: Result<u32, ModelThreadError> = deliver(target, || 7_u32);
        assert_eq!(outcome, Err(ModelThreadError::Queued));
    }

    /// 目标线程就是当前线程时**就地**执行，不排队、不等待。
    #[test]
    fn a_request_on_the_current_thread_runs_inline() {
        let _guard = serialize();
        let outcome = deliver(std::thread::current().id(), || 11_u32);
        assert_eq!(outcome, Ok(11));
    }

    /// 目标线程在泵队列时，投递**当场**完成 —— 这是宿主正常回调时的形状。
    #[test]
    fn a_request_is_picked_up_when_the_target_thread_pumps() {
        let _guard = serialize();
        let (target, stop) = a_pumping_thread();
        let outcome = deliver(target, || 13_u32);
        stop.store(true, Ordering::Release);
        assert_eq!(outcome, Ok(13));
    }
}
