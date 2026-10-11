//! Processing-algorithm and audio-file-chunk scenarios.

use super::catalog::{with_document, ScenarioReport};
use crate::test_audio_source_properties;
use ara2_bridge_core::{read_ixml, ApiGeneration, AraChunkSet, AraError};
use std::io::Cursor;
use std::sync::atomic::AtomicU8;

static CHUNK_WRITER: AtomicU8 = AtomicU8::new(0);
const CHUNK_WAVE: &[u8] = include_bytes!("../../fixtures/scenarios/chunk-wave.wav");
const CHUNK_AIFF: &[u8] = include_bytes!("../../fixtures/scenarios/chunk-aiff.aiff");

pub(super) fn processing_algorithms() -> Result<ScenarioReport, AraError> {
    with_document(
        "processing-algorithms",
        ApiGeneration::V23Final,
        |_, _, session| {
            let source = {
                let mut edit = session.edit()?;
                let source = edit.create_audio_source(test_audio_source_properties()?)?;
                edit.finish()?;
                source
            };
            let algorithms = session.processing_algorithms()?;
            let catalog_valid =
                algorithms.len() == 2 && algorithms[1].persistent_id() == "test.polyphonic";
            let mut edit = session.edit()?;
            edit.request_processing_algorithm(source, 1)?;
            edit.finish()?;
            let selected = session.processing_algorithm_for_audio_source(source)? == 1;
            if !(catalog_valid && selected) {
                return Err(AraError::Peer("processing algorithm scenario failed"));
            }
            Ok((5, 3, 3))
        },
    )
}

pub(super) fn audio_file_chunk_load() -> Result<ScenarioReport, AraError> {
    let mut wave = Cursor::new(CHUNK_WAVE);
    let wave_xml = read_ixml(&mut wave)
        .map_err(|_| AraError::Peer("WAVE fixture iXML could not be read"))?
        .ok_or(AraError::Peer("WAVE fixture has no iXML"))?;
    let mut aiff = Cursor::new(CHUNK_AIFF);
    let aiff_xml = read_ixml(&mut aiff)
        .map_err(|_| AraError::Peer("AIFF fixture iXML could not be read"))?
        .ok_or(AraError::Peer("AIFF fixture has no iXML"))?;
    let wave_set = AraChunkSet::parse(&wave_xml)
        .map_err(|_| AraError::Peer("WAVE ARA chunk could not be parsed"))?;
    let aiff_set = AraChunkSet::parse(&aiff_xml)
        .map_err(|_| AraError::Peer("AIFF ARA chunk could not be parsed"))?;
    if wave_set.archive_ids().count() != 1 || aiff_set.archive_ids().count() != 1 {
        return Err(AraError::Peer("chunk fixtures did not restore one source"));
    }
    Ok(ScenarioReport::success("audio-file-chunk-load", 4, 4, 2))
}

pub(super) fn audio_file_chunk_save() -> Result<ScenarioReport, AraError> {
    with_document(
        "audio-file-chunk-save",
        ApiGeneration::V23Final,
        |host, trace, session| {
            let source = {
                let mut edit = session.edit()?;
                let source = edit.create_audio_source(test_audio_source_properties()?)?;
                edit.finish()?;
                source
            };
            let stored = session.store_audio_source_to_audio_file_chunk(&CHUNK_WRITER, source)?;
            let bytes = host
                .written_archive()
                .ok_or(AraError::Peer("chunk storage produced no archive bytes"))?;
            if bytes != b"ARA"
                || stored.document_archive_id() != "org.ara2-bridge.test.archive"
                || !stored.open_automatically()
                || trace.count("store_audio_file_chunk") != 1
            {
                return Err(AraError::Peer("audio-file chunk storage scenario failed"));
            }
            Ok((4, 4, 3))
        },
    )
}
