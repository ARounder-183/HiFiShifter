//! Windows命名管道生命周期；限时请求且析构同步结束工作线程。
use super::*;
use std::fs::File;
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::Storage::FileSystem::*;
use windows_sys::Win32::System::Pipes::*;

pub struct Server {
    record: InstanceRecord,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    path: PathBuf,
}
impl Server {
    /// 新建只接受本机请求的实例，真实监听成功后才发布记录。
    pub fn start(
        name: String,
        handler: impl Fn(Request) -> Response + Send + 'static,
    ) -> Result<Self, String> {
        Self::start_at(name, instance_dir()?, handler)
    }
    /// 指定发现目录，便于隔离部署而不触碰用户的其他实例。
    pub fn start_at(
        name: String,
        root: PathBuf,
        handler: impl Fn(Request) -> Response + Send + 'static,
    ) -> Result<Self, String> {
        let instance_id = uuid::Uuid::new_v4().to_string();
        let pid = std::process::id();
        let record = InstanceRecord {
            pipe_name: format!(r"\\.\pipe\HiFiShifter-ARA-{pid}-{instance_id}"),
            instance_id,
            pid,
            token: uuid::Uuid::new_v4().to_string(),
            name,
            heartbeat_ms: now_ms(),
            protocol: PROTOCOL,
        };
        let file = create_pipe(&record.pipe_name)?;
        std::fs::create_dir_all(&root)
            .map_err(|e| format!("instance directory {}: {e}", root.display()))?;
        let path = root.join(format!("{}.json", record.instance_id));
        std::fs::write(
            &path,
            serde_json::to_vec(&record).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("instance record {}: {e}", path.display()))?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let mut heartbeat = record.clone();
        let heartbeat_path = path.clone();
        let worker = std::thread::Builder::new()
            .name("hfs-ara-ipc".into())
            .spawn(move || {
                let mut last_heartbeat = Instant::now();
                while !flag.load(Ordering::Acquire) {
                    if last_heartbeat.elapsed() > Duration::from_secs(1) {
                        heartbeat.heartbeat_ms = now_ms();
                        if let Ok(bytes) = serde_json::to_vec(&heartbeat) {
                            let _ = std::fs::write(&heartbeat_path, bytes);
                        }
                        last_heartbeat = Instant::now();
                    }
                    // SAFETY: File owns a nonblocking server pipe; no overlapped structure is needed.
                    let connected =
                        unsafe { ConnectNamedPipe(file.as_raw_handle(), std::ptr::null_mut()) };
                    if connected == 0 && unsafe { GetLastError() } != ERROR_PIPE_CONNECTED {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    let mut io = PipeIo::new(&file, Some(&flag));
                    log::info!("[ara-ipc] client connected");
                    io.deadline = Instant::now() + Duration::from_secs(5);
                    let response = match read_frame::<Envelope>(&mut io) {
                        Ok(envelope)
                            if envelope.protocol == PROTOCOL
                                && envelope.token == heartbeat.token =>
                        {
                            log::info!("[ara-ipc] request authenticated");
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                handler(envelope.request)
                            }))
                            .unwrap_or_else(|_| Response {
                                error: Some("plugin request panicked".into()),
                                ..Default::default()
                            })
                        }
                        Ok(_) => Response {
                            error: Some("ARA authentication failed".into()),
                            ..Default::default()
                        },
                        Err(error) => {
                            log::warn!("[ara-ipc] request read failed: {error}");
                            unsafe {
                                DisconnectNamedPipe(file.as_raw_handle());
                            }
                            continue;
                        }
                    };
                    io.deadline = Instant::now() + Duration::from_secs(120);
                    if write_frame(&mut io, &response).is_ok() {
                        log::info!("[ara-ipc] response written ok={}", response.ok);
                        // 客户端读完后确认，避免Disconnect丢弃仍在管道缓冲里的尾部数据。
                        let mut ack = [0];
                        let _ = io.read_exact(&mut ack);
                    }
                    // SAFETY: server handle is retained for the next single-client connection.
                    unsafe {
                        DisconnectNamedPipe(file.as_raw_handle());
                    }
                }
            })
            .map_err(|e| {
                let _ = std::fs::remove_file(&path);
                e.to_string()
            })?;
        Ok(Self {
            record,
            stop,
            worker: Some(worker),
            path,
        })
    }
    pub fn record(&self) -> &InstanceRecord {
        &self.record
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

fn create_pipe(name: &str) -> Result<File, String> {
    use windows_sys::Win32::Security::{
        Authorization::ConvertStringSecurityDescriptorToSecurityDescriptorW, SECURITY_ATTRIBUTES,
    };
    let name = name.encode_utf16().chain([0]).collect::<Vec<_>>();
    // Only the current owner and SYSTEM can open the pipe. Remote clients are rejected separately.
    // Low integrity GUI can write this application-owned pipe, but only the same user is in the DACL.
    let acl = format!(
        "D:P(A;;GA;;;SY)(A;;GA;;;{})S:(ML;;NW;;;LW)",
        current_user_sid()?
    )
    .encode_utf16()
    .chain([0])
    .collect::<Vec<_>>();
    let mut descriptor = std::ptr::null_mut();
    // SAFETY: null-terminated SDDL and live output storage.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            acl.as_ptr(),
            1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(format!("SDDL: {}", std::io::Error::last_os_error()));
    }
    let security = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    // SAFETY: all pointers remain live for creation; returned handle is owned exactly once.
    let handle = unsafe {
        CreateNamedPipeW(
            name.as_ptr(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1,
            65536,
            65536,
            0,
            &security,
        )
    };
    let creation_error = std::io::Error::last_os_error();
    unsafe {
        LocalFree(descriptor);
    }
    if handle == INVALID_HANDLE_VALUE {
        return Err(format!("CreateNamedPipeW: {creation_error}"));
    }
    Ok(unsafe { File::from_raw_handle(handle) })
}

/// 使用实际用户SID而非owner group，避免提升权限的宿主与普通GUI的owner身份不同。
fn current_user_sid() -> Result<String, String> {
    use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;
    use windows_sys::Win32::Security::*;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    let mut token = std::ptr::null_mut();
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let result = (|| {
        let mut length = 0;
        unsafe {
            GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut length);
        }
        // Vec<usize> guarantees TOKEN_USER alignment and storage for the SID payload.
        let mut storage = vec![0_usize; (length as usize).div_ceil(std::mem::size_of::<usize>())];
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                storage.as_mut_ptr().cast(),
                length,
                &mut length,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let user = unsafe { &*storage.as_ptr().cast::<TOKEN_USER>() };
        let mut text = std::ptr::null_mut();
        if unsafe { ConvertSidToStringSidW(user.User.Sid, &mut text) } == 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let mut size = 0;
        while unsafe { *text.add(size) } != 0 {
            size += 1;
        }
        let sid = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(text, size) });
        unsafe {
            LocalFree(text.cast());
        }
        Ok(sid)
    })();
    unsafe {
        CloseHandle(token);
    }
    result
}

struct PipeIo<'a> {
    file: &'a File,
    deadline: Instant,
    stop: Option<&'a AtomicBool>,
}
impl<'a> PipeIo<'a> {
    fn new(file: &'a File, stop: Option<&'a AtomicBool>) -> Self {
        Self {
            file,
            deadline: Instant::now() + Duration::from_secs(120),
            stop,
        }
    }
    fn wait(&self) -> std::io::Result<()> {
        if self.stop.is_some_and(|s| s.load(Ordering::Acquire)) || Instant::now() >= self.deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "ARA request timed out or stopped",
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
        Ok(())
    }
}
impl Read for PipeIo<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let size = bytes.len().min(16384);
        loop {
            match (&*self.file).read(&mut bytes[..size]) {
                Ok(0) => self.wait()?,
                Ok(n) => return Ok(n),
                Err(e)
                    if e.raw_os_error() == Some(ERROR_NO_DATA as i32)
                        || e.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    self.wait()?
                }
                Err(e) => return Err(e),
            }
        }
    }
}
impl Write for PipeIo<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let size = bytes.len().min(16384);
        loop {
            match (&*self.file).write(&bytes[..size]) {
                Ok(0) => self.wait()?,
                Ok(n) => return Ok(n),
                Err(e)
                    if e.raw_os_error() == Some(ERROR_NO_DATA as i32)
                        || e.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    self.wait()?
                }
                Err(e) => return Err(e),
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn exchange(record: &InstanceRecord, request: &Request) -> Result<Response, String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let name = record
        .pipe_name
        .encode_utf16()
        .chain([0])
        .collect::<Vec<_>>();
    let file = loop {
        // Pipe open is not a filesystem create; use the Win32 pipe access contract explicitly.
        let handle = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        if handle != INVALID_HANDLE_VALUE {
            break unsafe { File::from_raw_handle(handle) };
        }
        let error = std::io::Error::last_os_error();
        match error {
            e if Instant::now() < deadline
                && [ERROR_PIPE_BUSY as i32, ERROR_FILE_NOT_FOUND as i32]
                    .contains(&e.raw_os_error().unwrap_or(0)) =>
            {
                std::thread::sleep(Duration::from_millis(10))
            }
            e => return Err(format!("open ARA pipe: {e}")),
        }
    };
    let mode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
    // SAFETY: client pipe handle and mode storage are live.
    if unsafe {
        SetNamedPipeHandleState(
            file.as_raw_handle(),
            &mode,
            std::ptr::null(),
            std::ptr::null(),
        )
    } == 0
    {
        return Err(format!(
            "set client pipe mode: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut io = PipeIo::new(&file, None);
    write_frame(
        &mut io,
        &Envelope {
            protocol: PROTOCOL,
            token: record.token.clone(),
            request: request.clone(),
        },
    )?;
    let response = read_frame(&mut io)?;
    io.write_all(&[1]).map_err(|e| e.to_string())?;
    Ok(response)
}
