use ara2_bridge_core::SizedInput;
#[cfg(unix)]
use ara2_bridge_core::SizedRecord;
use ara2_bridge_sys::ARADocumentProperties;
#[cfg(unix)]
use std::ffi::c_char;
#[cfg(unix)]
use std::mem::offset_of;
use std::mem::size_of;

fn main() {
    let case = std::env::args().nth(1).unwrap_or_default();
    let result = match case.as_str() {
        "malformed" => caller_valid_malformed(),
        "null-adjacent" => unsafe_case(null_adjacent),
        "unreadable" => unsafe_case(unreadable),
        "guard-page" => unsafe_case(guard_page),
        _ => Err("expected malformed, null-adjacent, unreadable, or guard-page"),
    };
    if let Err(message) = result {
        eprintln!("{message}");
        std::process::exit(64);
    }
}

fn caller_valid_malformed() -> Result<(), &'static str> {
    let mut storage = vec![0_u8; size_of::<usize>()];
    // SAFETY: the local storage contains one complete, writable native ARASize.
    unsafe { storage.as_mut_ptr().cast::<usize>().write_unaligned(1) };
    // SAFETY: local storage remains readable for its advertised single byte and the leading size.
    let result = unsafe { SizedInput::<ARADocumentProperties>::from_ptr(storage.as_ptr().cast()) };
    if result.is_err() {
        Ok(())
    } else {
        Err("caller-valid malformed record was accepted")
    }
}

fn unsafe_case(operation: unsafe fn()) -> Result<(), &'static str> {
    if std::env::var_os("ARA2_BRIDGE_ALLOW_INVALID_POINTER_CASE").is_none() {
        return Err("invalid pointer cases require the sanitizer harness opt-in");
    }
    // SAFETY: the explicit environment opt-in is set only by an isolated sanitizer subprocess.
    unsafe { operation() };
    Err("invalid pointer operation unexpectedly returned")
}

unsafe fn null_adjacent() {
    let pointer = 1_usize as *const ARADocumentProperties;
    // SAFETY: deliberately violates the documented caller contract in an isolated sanitizer child.
    let _ = unsafe { SizedInput::<ARADocumentProperties>::from_ptr(pointer) };
}

#[cfg(unix)]
unsafe fn unreadable() {
    // SAFETY: requests one anonymous private page; the result is checked before use.
    let page = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            page_size(),
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(page, libc::MAP_FAILED);
    // SAFETY: deliberately reads a PROT_NONE page in the isolated sanitizer child.
    let _ = unsafe { SizedInput::<ARADocumentProperties>::from_ptr(page.cast()) };
}

#[cfg(not(unix))]
unsafe fn unreadable() {
    unsafe { null_adjacent() };
}

#[cfg(unix)]
unsafe fn guard_page() {
    let length = page_size();
    // SAFETY: requests two writable anonymous private pages; the result is checked before use.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length * 2,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(mapping, libc::MAP_FAILED);
    // SAFETY: the second mapped page is page-aligned and within the two-page allocation.
    let guard = unsafe { mapping.cast::<u8>().add(length) };
    // SAFETY: changes exactly the second page to inaccessible and checks success.
    assert_eq!(
        unsafe { libc::mprotect(guard.cast(), length, libc::PROT_NONE) },
        0
    );
    // SAFETY: the first page remains writable through its final native ARASize.
    let record = unsafe { guard.sub(size_of::<usize>()) };
    // SAFETY: the final bytes of the first page contain a complete writable native ARASize.
    unsafe {
        record.cast::<usize>().write_unaligned(
            <ARADocumentProperties as SizedRecord>::FIELD_EXTENTS
                .last()
                .copied()
                .unwrap(),
        );
    }
    // SAFETY: construction reads only the complete leading size in the first page.
    let input = unsafe {
        SizedInput::<ARADocumentProperties>::from_ptr(record.cast::<ARADocumentProperties>())
    }
    .unwrap();
    // SAFETY: deliberately violates the advertised-extent readability contract by placing the
    // represented `name` field in the guard page; the sanitizer child must classify the failure.
    let _ = unsafe {
        input.copy_field::<*const c_char>(
            offset_of!(ARADocumentProperties, name),
            ara2_bridge_sys::layout::ARADOCUMENT_PROPERTIES_NAME,
        )
    };
}

#[cfg(not(unix))]
unsafe fn guard_page() {
    unsafe { null_adjacent() };
}

#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: `_SC_PAGESIZE` has no pointer arguments or side effects.
    usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).expect("positive page size")
}
