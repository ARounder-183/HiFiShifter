//! 每个renderer的有界后台准备邮箱：最多一个运行和一个最新任务，宿主回调只排队。

use std::sync::{Arc,Mutex,Condvar};
use std::sync::atomic::{AtomicBool,Ordering};
type Job=Box<dyn FnOnce(Arc<AtomicBool>)->Result<(),String>+Send>;
#[derive(Default)]
struct Mailbox {closed:bool,pending:Option<Job>,active:Option<Arc<AtomicBool>>,error:Option<String>}
struct State {mailbox:Mutex<Mailbox>,wake:Condvar}
pub(crate) struct PreparationQueue {state:Arc<State>,worker:Mutex<Option<std::thread::JoinHandle<()>>>}
impl PreparationQueue {
    /// worker不持renderer/document强引用；job只在运行时借出它们。
    pub fn new()->Result<Self,String> {
        let state=Arc::new(State {mailbox:Mutex::new(Default::default()),wake:Condvar::new()});
        let run=state.clone();
        let worker=std::thread::Builder::new().name("hfs-host-prepare".into()).spawn(move || {
            loop {
                let (job,cancel)={
                    let mut mailbox=run.mailbox.lock().unwrap();
                    while !mailbox.closed && mailbox.pending.is_none() {mailbox=run.wake.wait(mailbox).unwrap();}
                    if mailbox.closed {break;}
                    let cancel=Arc::new(AtomicBool::new(false));mailbox.active=Some(cancel.clone());
                    (mailbox.pending.take().unwrap(),cancel)
                };
                let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||job(cancel.clone())))
                    .unwrap_or_else(|_|Err("host preparation panicked".into()));
                let mut mailbox=run.mailbox.lock().unwrap();mailbox.active=None;
                if !cancel.load(Ordering::Acquire) {mailbox.error=result.err();}
                run.wake.notify_all();
            }
        }).map_err(|e|format!("host preparation worker: {e}"))?;
        Ok(Self {state,worker:Mutex::new(Some(worker))})
    }
    /// 只保存最新待办，旧任务的取消标记立即置位；不在宿主线程执行任务或等待计算。
    pub fn request(&self,job:Job)->Result<(),String> {
        let mut mailbox=self.state.mailbox.lock().unwrap();
        if mailbox.closed {return Err("host preparation closed".into());}
        if let Some(cancel)=&mailbox.active {cancel.store(true,Ordering::Release);}
        // worker与离线UI等待者共享Condvar但等待条件不同，必须唤醒全部以免只唤醒UI却遗留pending。
        mailbox.pending=Some(job);self.state.wake.notify_all();Ok(())
    }
    /// 状态查询只读小邮箱，不能把失败当作已经准备成功。
    pub fn state(&self)->(bool,Option<String>) {
        let mailbox=self.state.mailbox.lock().unwrap();
        (mailbox.pending.is_some()||mailbox.active.is_some(),mailbox.error.clone())
    }
    /// 仅VST3离线setup的UI线程使用；Condvar等待后台工作，process/setProcessing不得调用。
    pub fn wait_idle_until(&self,deadline:std::time::Instant)->Result<(),String> {
        let mut mailbox=self.state.mailbox.lock().unwrap();
        loop {
            if mailbox.closed {return Err("host preparation closed".into());}
            if mailbox.pending.is_none()&&mailbox.active.is_none() {return mailbox.error.clone().map_or(Ok(()),Err);}
            let remaining=deadline.checked_duration_since(std::time::Instant::now()).ok_or("offline preparation timed out")?;
            let (next,_)=self.state.wake.wait_timeout(mailbox,remaining).unwrap();mailbox=next;
        }
    }
    /// 源/模型撤销取消当前与待办，但不终止worker，下一次授权可再次排队。
    pub fn cancel(&self) {
        let mut mailbox=self.state.mailbox.lock().unwrap();mailbox.pending=None;
        if let Some(cancel)=&mailbox.active {cancel.store(true,Ordering::Release);}
    }
    /// 关闭取消/join；不持邮箱锁join，也不能在worker自身join自身。
    pub fn close(&self) {
        {let mut mailbox=self.state.mailbox.lock().unwrap();mailbox.closed=true;mailbox.pending=None;
            if let Some(cancel)=&mailbox.active {cancel.store(true,Ordering::Release);}self.state.wake.notify_all();}
        if let Some(worker)=self.worker.lock().unwrap().take() {
            if worker.thread().id()!=std::thread::current().id() {let _=worker.join();}
        }
    }
}
impl Drop for PreparationQueue {fn drop(&mut self){self.close();}}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;
    #[test]
    fn offline_wait_reports_timeout_failure_and_close_without_running_work_on_caller() {
        let queue=PreparationQueue::new().unwrap();let (entered,ready)=mpsc::channel();let (release,gate)=mpsc::channel();
        queue.request(Box::new(move |_| {entered.send(()).unwrap();gate.recv_timeout(Duration::from_secs(3)).unwrap();Err("fixture infer failed".into())})).unwrap();
        ready.recv_timeout(Duration::from_secs(3)).unwrap();
        assert!(queue.wait_idle_until(std::time::Instant::now()+Duration::from_millis(10)).unwrap_err().contains("timed out"));
        release.send(()).unwrap();assert_eq!(queue.wait_idle_until(std::time::Instant::now()+Duration::from_secs(3)).unwrap_err(),"fixture infer failed");
        queue.close();assert!(queue.wait_idle_until(std::time::Instant::now()+Duration::from_secs(1)).unwrap_err().contains("closed"));
    }

    /// 如果任务在调用者执行、队列不合并或不能取消旧任务，这个真实邮箱测试失败。
    #[test]
    fn request_is_background_only_and_keeps_only_the_latest_pending_job() {
        let queue=PreparationQueue::new().unwrap();
        let caller=std::thread::current().id();
        let (entered,started)=mpsc::channel();let (release,gate)=mpsc::channel();
        let (output,received)=mpsc::channel();
        let first=output.clone();
        queue.request(Box::new(move |cancel| {
            assert_ne!(std::thread::current().id(),caller);
            entered.send(()).unwrap();gate.recv_timeout(Duration::from_secs(3)).unwrap();
            assert!(cancel.load(Ordering::Acquire));first.send(1).unwrap();Ok(())
        })).unwrap();
        started.recv_timeout(Duration::from_secs(3)).unwrap();
        let second=output.clone();queue.request(Box::new(move |_|{second.send(2).unwrap();Ok(())})).unwrap();
        queue.request(Box::new(move |_|{output.send(3).unwrap();Ok(())})).unwrap();
        release.send(()).unwrap();
        assert_eq!(received.recv_timeout(Duration::from_secs(3)).unwrap(),1);
        assert_eq!(received.recv_timeout(Duration::from_secs(3)).unwrap(),3);
        queue.close();assert!(received.try_recv().is_err());
        assert!(queue.request(Box::new(|_|Ok(()))).is_err());
    }

    /// 失败可见，后续成功才清除错误；shutdown取消当前任务并等待它退出。
    #[test]
    fn failure_is_observable_and_shutdown_cancels_the_active_job() {
        let queue=PreparationQueue::new().unwrap();
        queue.request(Box::new(|_|Err("fixture failure".into()))).unwrap();
        let deadline=std::time::Instant::now()+Duration::from_secs(3);
        while queue.state().0 {assert!(std::time::Instant::now()<deadline);std::thread::yield_now();}
        assert_eq!(queue.state().1.as_deref(),Some("fixture failure"));
        let (ready,started)=mpsc::channel();
        queue.request(Box::new(move |cancel|{
            ready.send(()).unwrap();let deadline=std::time::Instant::now()+Duration::from_secs(3);
            while !cancel.load(Ordering::Acquire) {assert!(std::time::Instant::now()<deadline);std::thread::yield_now();}
            Ok(())
        })).unwrap();
        started.recv_timeout(Duration::from_secs(3)).unwrap();queue.close();
        assert!(!queue.state().0);
    }
}
