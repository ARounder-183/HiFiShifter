//! 对照锁定SDK IBStream的组件state读写，短读写必须循环，不允许损坏state伪成功。
use std::ffi::c_void;

#[repr(C)]
struct StreamVtbl {
    query: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    read: unsafe extern "system" fn(*mut c_void, *mut c_void, i32, *mut i32) -> i32,
    write: unsafe extern "system" fn(*mut c_void, *mut c_void, i32, *mut i32) -> i32,
    seek: unsafe extern "system" fn(*mut c_void, i64, i32, *mut i64) -> i32,
    tell: unsafe extern "system" fn(*mut c_void, *mut i64) -> i32,
}

/// 宿主拥有有效IBStream，生命周期覆盖调用；空旧state合法。
pub(crate) unsafe fn read_state(stream: *mut c_void) -> Result<Vec<u8>, String> {
    if stream.is_null() {
        return Err("null state stream".into());
    }
    let vtbl = unsafe { &**stream.cast::<*const StreamVtbl>() };
    let mut prefix = [0; 4];
    let mut done = 0;
    while done < prefix.len() {
        let mut actual = 0;
        let result = unsafe {
            (vtbl.read)(
                stream,
                prefix[done..].as_mut_ptr().cast(),
                (4 - done) as i32,
                &mut actual,
            )
        };
        if actual == 0 && done == 0 {
            return Ok(Vec::new());
        }
        if result < 0 || actual <= 0 || actual as usize > 4 - done {
            return Err("truncated state prefix".into());
        }
        done += actual as usize;
    }
    let length = u32::from_le_bytes(prefix) as usize;
    if length > hifishifter_ara_ipc::MAX_FRAME {
        return Err("state too large".into());
    }
    let mut bytes = vec![0; length];
    let mut done = 0;
    while done < length {
        let mut actual = 0;
        let result = unsafe {
            (vtbl.read)(
                stream,
                bytes[done..].as_mut_ptr().cast(),
                (length - done) as i32,
                &mut actual,
            )
        };
        if result < 0 || actual <= 0 || actual as usize > length - done {
            return Err("truncated state content".into());
        }
        done += actual as usize;
    }
    Ok(bytes)
}
/// 宿主拥有有效可写IBStream；全部字节完成前不报告成功。
pub(crate) unsafe fn write_state(stream: *mut c_void, bytes: &[u8]) -> Result<(), String> {
    if stream.is_null() || bytes.len() > hifishifter_ara_ipc::MAX_FRAME {
        return Err("invalid state stream/length".into());
    }
    let vtbl = unsafe { &**stream.cast::<*const StreamVtbl>() };
    let prefix = (bytes.len() as u32).to_le_bytes();
    for segment in [prefix.as_slice(), bytes] {
        let mut done = 0;
        while done < segment.len() {
            let mut actual = 0;
            let result = unsafe {
                (vtbl.write)(
                    stream,
                    segment[done..].as_ptr().cast_mut().cast(),
                    (segment.len() - done) as i32,
                    &mut actual,
                )
            };
            if result < 0 || actual <= 0 || actual as usize > segment.len() - done {
                return Err("incomplete state write".into());
            }
            done += actual as usize;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[repr(C)]
    struct Stream {
        vtbl: *const StreamVtbl,
        bytes: Vec<u8>,
        position: usize,
    }
    unsafe extern "system" fn query(_: *mut c_void, _: *const u8, _: *mut *mut c_void) -> i32 {
        1
    }
    unsafe extern "system" fn refs(_: *mut c_void) -> u32 {
        1
    }
    unsafe extern "system" fn read(
        ptr: *mut c_void,
        buffer: *mut c_void,
        count: i32,
        actual: *mut i32,
    ) -> i32 {
        let stream = unsafe { &mut *ptr.cast::<Stream>() };
        let size = (count as usize)
            .min(3)
            .min(stream.bytes.len() - stream.position);
        unsafe {
            std::ptr::copy_nonoverlapping(
                stream.bytes.as_ptr().add(stream.position),
                buffer.cast(),
                size,
            );
            *actual = size as i32;
        }
        stream.position += size;
        if size == 0 {
            1
        } else {
            0
        }
    }
    unsafe extern "system" fn write(
        ptr: *mut c_void,
        buffer: *mut c_void,
        count: i32,
        actual: *mut i32,
    ) -> i32 {
        let stream = unsafe { &mut *ptr.cast::<Stream>() };
        let size = (count as usize).min(2);
        stream
            .bytes
            .extend_from_slice(unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), size) });
        unsafe {
            *actual = size as i32;
        }
        0
    }
    unsafe extern "system" fn seek(_: *mut c_void, _: i64, _: i32, _: *mut i64) -> i32 {
        1
    }
    unsafe extern "system" fn tell(_: *mut c_void, _: *mut i64) -> i32 {
        1
    }
    static VTABLE: StreamVtbl = StreamVtbl {
        query,
        add_ref: refs,
        release: refs,
        read,
        write,
        seek,
        tell,
    };

    #[test]
    fn real_stream_callbacks_round_trip_partial_reads_and_writes() {
        let mut stream = Stream {
            vtbl: &VTABLE,
            bytes: Vec::new(),
            position: 0,
        };
        let ptr = (&raw mut stream).cast();
        unsafe {
            write_state(ptr, b"state curves").unwrap();
        }
        assert_eq!(&stream.bytes[..4], &[12, 0, 0, 0]);
        assert_eq!(unsafe { read_state(ptr).unwrap() }, b"state curves");
        stream.bytes = vec![7, 0, 0, 0, b'{'];
        stream.position = 0;
        assert!(unsafe { read_state(ptr) }.is_err());
    }
}
