//! Optional content and analysis capabilities.

use crate::{ContentObject, ContentReaderSnapshot};
use ara2_bridge_core::{AraError, ContentGrade, ContentTimeRange, RawHandle};

/// Supplies immutable typed-content snapshots for live model objects.
pub trait ContentProvider {
    /// Returns whether content of a raw ARA type is currently available.
    fn is_content_available(&self, _object: ContentObject, _content_type: i32) -> bool {
        false
    }

    /// Returns the grade for currently available content.
    fn content_grade(&self, _object: ContentObject, _content_type: i32) -> ContentGrade {
        ContentGrade::INITIAL
    }

    /// Builds an immutable reader snapshot for the selected object, kind, and optional range.
    fn create_content_reader(
        &mut self,
        _object: ContentObject,
        _content_type: i32,
        _range: Option<ContentTimeRange>,
    ) -> Result<Option<ContentReaderSnapshot>, AraError> {
        Ok(None)
    }
}

/// Starts and cancels asynchronous analysis for audio sources.
pub trait AnalysisProvider {
    /// Returns whether this source still has incomplete analysis for a content type.
    fn is_analysis_incomplete(&self, _audio_source: RawHandle, _content_type: i32) -> bool {
        false
    }

    /// Starts analysis for the requested content types.
    fn request_analysis(
        &mut self,
        _audio_source: RawHandle,
        _content_types: &[i32],
    ) -> Result<(), AraError> {
        Err(AraError::Unsupported("analysis is not implemented"))
    }

    /// Synchronously cancels work before sample access is revoked or the source is destroyed.
    fn cancel_analysis(&mut self, _audio_source: RawHandle) {}
}
