//! Typed content analysis and reader scenarios.

use super::catalog::{with_document, ScenarioReport};
use crate::test_audio_source_properties;
use ara2_bridge_core::{ApiGeneration, AraError, ContentGrade, Notes};

pub(super) fn content_reading() -> Result<ScenarioReport, AraError> {
    with_document(
        "content-reading",
        ApiGeneration::V23Final,
        |host, trace, session| {
            let source = {
                let mut edit = session.edit()?;
                let source = edit.create_audio_source(test_audio_source_properties()?)?;
                edit.finish()?;
                source
            };
            if !session.audio_source_content_available::<Notes>(source)? {
                return Err(AraError::Peer("note content is unavailable"));
            }
            if session.audio_source_content_grade::<Notes>(source)? != ContentGrade::APPROVED {
                return Err(AraError::Peer("note content grade is not approved"));
            }
            session.request_audio_source_content_analysis::<Notes>(source)?;
            session.notify_model_updates()?;
            let mut reader = session
                .audio_source_content_reader::<Notes>(source, None)?
                .ok_or(AraError::Peer("note reader was not created"))?;
            let event = reader.event(0)?;
            let event_valid = reader.len() == 1 && event.frequency() == Some(440.0);
            drop(reader);
            let progress_valid = host.trace().count("analysis_progress") == 3;
            let request_valid = trace.count("request_analysis") == 1;
            if !(event_valid && progress_valid && request_valid) {
                return Err(AraError::Peer("typed content scenario assertions failed"));
            }
            Ok((7, 5, 5))
        },
    )
}
