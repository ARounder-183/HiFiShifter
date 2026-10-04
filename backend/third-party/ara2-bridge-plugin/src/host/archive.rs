//! Call-scoped host archive readers and writers.

use ara2_bridge_core::{ApiGeneration, AraError, SizedInput};
use ara2_bridge_sys::*;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem::offset_of;

type GetSize =
    unsafe extern "C" fn(ARAArchivingControllerHostRef, ARAArchiveReaderHostRef) -> usize;
type Read = unsafe extern "C" fn(
    ARAArchivingControllerHostRef,
    ARAArchiveReaderHostRef,
    usize,
    usize,
    *mut u8,
) -> ARABool;
type Write = unsafe extern "C" fn(
    ARAArchivingControllerHostRef,
    ARAArchiveWriterHostRef,
    usize,
    usize,
    *const u8,
) -> ARABool;
type Progress = unsafe extern "C" fn(ARAArchivingControllerHostRef, f32);
type ArchiveId =
    unsafe extern "C" fn(ARAArchivingControllerHostRef, ARAArchiveReaderHostRef) -> ARAPersistentID;

/// Required archive transport supplied by the host.
pub struct ArchiveAccess<'host> {
    host_ref: ARAArchivingControllerHostRef,
    get_size: GetSize,
    read: Read,
    write: Write,
    archive_progress: Progress,
    unarchive_progress: Progress,
    archive_id: Option<ArchiveId>,
    _lifetime: PhantomData<&'host ()>,
}

impl<'host> ArchiveAccess<'host> {
    pub(crate) unsafe fn from_raw(
        host_ref: ARAArchivingControllerHostRef,
        interface: *const ARAArchivingControllerInterface,
        generation: ApiGeneration,
    ) -> Result<Self, AraError> {
        if host_ref.is_null() || interface.is_null() {
            return Err(AraError::Abi("required archiving service is null"));
        }
        // SAFETY: the caller supplies readable interface storage for the returned lifetime.
        let input = unsafe { SizedInput::from_ptr(interface) }?;
        macro_rules! required {
            ($field:ident, $type:ty, $extent:ident, $error:literal) => {{
                // SAFETY: generated offset/type/extent identify the named field.
                unsafe {
                    input.copy_field::<Option<$type>>(
                        offset_of!(ARAArchivingControllerInterface, $field),
                        ara2_bridge_sys::layout::$extent,
                    )
                }?
                .ok_or(AraError::Abi($error))?
            }};
        }
        let archive_id = if input.contains_extent(
            ara2_bridge_sys::layout::ARAARCHIVING_CONTROLLER_INTERFACE_GET_DOCUMENT_ARCHIVE_ID,
        ) {
            // SAFETY: the represented prefix contains this generated callback field.
            unsafe {
                input.copy_field::<Option<ArchiveId>>(
                    offset_of!(ARAArchivingControllerInterface, getDocumentArchiveID),
                    ara2_bridge_sys::layout::ARAARCHIVING_CONTROLLER_INTERFACE_GET_DOCUMENT_ARCHIVE_ID,
                )
            }?
        } else {
            None
        };
        if generation >= ApiGeneration::V2Final && archive_id.is_none() {
            return Err(AraError::Abi(
                "ARA 2 host must provide the document archive ID callback",
            ));
        }
        Ok(Self {
            host_ref,
            get_size: required!(
                getArchiveSize,
                GetSize,
                ARAARCHIVING_CONTROLLER_INTERFACE_GET_ARCHIVE_SIZE,
                "archive size callback is null"
            ),
            read: required!(
                readBytesFromArchive,
                Read,
                ARAARCHIVING_CONTROLLER_INTERFACE_READ_BYTES_FROM_ARCHIVE,
                "archive read callback is null"
            ),
            write: required!(
                writeBytesToArchive,
                Write,
                ARAARCHIVING_CONTROLLER_INTERFACE_WRITE_BYTES_TO_ARCHIVE,
                "archive write callback is null"
            ),
            archive_progress: required!(
                notifyDocumentArchivingProgress,
                Progress,
                ARAARCHIVING_CONTROLLER_INTERFACE_NOTIFY_DOCUMENT_ARCHIVING_PROGRESS,
                "archive progress callback is null"
            ),
            unarchive_progress: required!(
                notifyDocumentUnarchivingProgress,
                Progress,
                ARAARCHIVING_CONTROLLER_INTERFACE_NOTIFY_DOCUMENT_UNARCHIVING_PROGRESS,
                "unarchive progress callback is null"
            ),
            archive_id,
            _lifetime: PhantomData,
        })
    }

    /// Creates a non-storable reader client for one active restore callback.
    pub fn with_reader<R>(
        &self,
        reader: ARAArchiveReaderHostRef,
        operation: impl FnOnce(HostArchiveReader<'_>) -> R,
    ) -> Result<R, AraError> {
        if reader.is_null() {
            return Err(AraError::InvalidArgument("archive reader is null"));
        }
        Ok(operation(HostArchiveReader {
            access: self,
            reader,
        }))
    }

    /// Creates a non-storable writer client for one active store callback.
    pub fn with_writer<R>(
        &self,
        writer: ARAArchiveWriterHostRef,
        operation: impl FnOnce(HostArchiveWriter<'_>) -> R,
    ) -> Result<R, AraError> {
        if writer.is_null() {
            return Err(AraError::InvalidArgument("archive writer is null"));
        }
        Ok(operation(HostArchiveWriter {
            access: self,
            writer,
        }))
    }
}

/// Reader valid only during its enclosing host restore callback.
pub struct HostArchiveReader<'call> {
    access: &'call ArchiveAccess<'call>,
    reader: ARAArchiveReaderHostRef,
}

impl HostArchiveReader<'_> {
    /// Returns the archive byte length.
    pub fn len(&self) -> usize {
        // SAFETY: the reader and validated host service are live for this callback scope.
        unsafe { (self.access.get_size)(self.access.host_ref, self.reader) }
    }

    /// Returns whether the archive contains no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads bytes at an absolute archive position.
    pub fn read_at(&self, position: usize, output: &mut [u8]) -> Result<(), AraError> {
        position
            .checked_add(output.len())
            .filter(|end| *end <= self.len())
            .ok_or(AraError::InvalidArgument("archive read is out of bounds"))?;
        // SAFETY: the validated service receives a live reader and writable output buffer.
        let result = unsafe {
            (self.access.read)(
                self.access.host_ref,
                self.reader,
                position,
                output.len(),
                output.as_mut_ptr(),
            )
        };
        if result != kARAFalse {
            Ok(())
        } else {
            Err(AraError::Peer("host archive read failed"))
        }
    }

    /// Copies the ARA 2 document archive identifier, if represented.
    pub fn document_archive_id(&self) -> Result<Option<String>, AraError> {
        let Some(callback) = self.access.archive_id else {
            return Ok(None);
        };
        // SAFETY: validated callback and live reader; ARA requires a persistent UTF-8 C string.
        let pointer = unsafe { callback(self.access.host_ref, self.reader) };
        if pointer.is_null() {
            return Err(AraError::Peer("host returned null document archive ID"));
        }
        // SAFETY: the host callback contract supplies a NUL-terminated string for the call.
        let id = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .map_err(|_| AraError::Abi("document archive ID is not UTF-8"))?;
        Ok(Some(id.to_owned()))
    }

    /// Reports monotonic unarchiving progress in `0..=1`.
    pub fn report_progress(&self, progress: f32) -> Result<(), AraError> {
        validate_progress(progress)?;
        // SAFETY: the validated service and host ref remain live for the call.
        unsafe { (self.access.unarchive_progress)(self.access.host_ref, progress) };
        Ok(())
    }
}

/// Writer valid only during its enclosing host store callback.
pub struct HostArchiveWriter<'call> {
    access: &'call ArchiveAccess<'call>,
    writer: ARAArchiveWriterHostRef,
}

impl HostArchiveWriter<'_> {
    /// Writes bytes at an absolute archive position.
    pub fn write_at(&self, position: usize, bytes: &[u8]) -> Result<(), AraError> {
        position
            .checked_add(bytes.len())
            .ok_or(AraError::InvalidArgument("archive write range overflow"))?;
        // SAFETY: the validated service receives a live writer and readable input bytes.
        let result = unsafe {
            (self.access.write)(
                self.access.host_ref,
                self.writer,
                position,
                bytes.len(),
                bytes.as_ptr(),
            )
        };
        if result != kARAFalse {
            Ok(())
        } else {
            Err(AraError::Peer("host archive write failed"))
        }
    }

    /// Reports monotonic archiving progress in `0..=1`.
    pub fn report_progress(&self, progress: f32) -> Result<(), AraError> {
        validate_progress(progress)?;
        // SAFETY: the validated service and host ref remain live for the call.
        unsafe { (self.access.archive_progress)(self.access.host_ref, progress) };
        Ok(())
    }
}

fn validate_progress(progress: f32) -> Result<(), AraError> {
    if progress.is_finite() && (0.0..=1.0).contains(&progress) {
        Ok(())
    } else {
        Err(AraError::InvalidArgument(
            "archive progress must be finite and in 0..=1",
        ))
    }
}
