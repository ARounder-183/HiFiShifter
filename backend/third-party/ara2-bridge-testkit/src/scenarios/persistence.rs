//! Full, partial, split, and import persistence scenarios.

use super::catalog::{with_document, ScenarioReport};
use crate::test_audio_source_properties;
use ara2_bridge_core::{ApiGeneration, AraError, AudioModificationProperties, RestoreFilter};
use std::sync::atomic::AtomicU8;

static ARCHIVE_READER: AtomicU8 = AtomicU8::new(0);
static SECOND_ARCHIVE_READER: AtomicU8 = AtomicU8::new(0);
static ARCHIVE_WRITER: AtomicU8 = AtomicU8::new(0);

const ARA1_ARCHIVE: &[u8] = include_bytes!("../../fixtures/scenarios/ara1-full.archive");
const ARA2_ARCHIVE: &[u8] = include_bytes!("../../fixtures/scenarios/ara2-full.archive");
const PARTIAL_A: &[u8] = include_bytes!("../../fixtures/scenarios/ara2-partial-a.archive");
const PARTIAL_B: &[u8] = include_bytes!("../../fixtures/scenarios/ara2-partial-b.archive");

pub(super) fn full_archive() -> Result<ScenarioReport, AraError> {
    with_document(
        "full-archive",
        ApiGeneration::V23Final,
        |host, trace, session| {
            session.store_document_to_archive(&ARCHIVE_WRITER)?;
            let written = host
                .written_archive()
                .ok_or(AraError::Peer("full archive produced no bytes"))?;
            if written != b"ARA" || !ARA2_ARCHIVE.starts_with(b"ARA2-BRIDGE-ARCHIVE\n") {
                return Err(AraError::Peer("full archive bytes were not recognized"));
            }
            if trace.count("store_document") != 1 {
                return Err(AraError::Peer("full archive callback was not balanced"));
            }
            Ok((2, 3, 2))
        },
    )
}

pub(super) fn split_partial_archives() -> Result<ScenarioReport, AraError> {
    with_document(
        "split-partial-archives",
        ApiGeneration::V23Final,
        |host, trace, session| {
            let (source, modification) = {
                let mut edit = session.edit()?;
                let source = edit.create_audio_source(test_audio_source_properties()?)?;
                let modification = edit.create_audio_modification(
                    source,
                    AudioModificationProperties::new(None, "split-modification")?,
                )?;
                edit.finish()?;
                (source, modification)
            };
            let source_filter = RestoreFilter::builder()
                .audio_source("archive-source", "test-source")
                .build()?;
            let modification_filter = RestoreFilter::builder()
                .audio_modification("archive-modification", "split-modification")
                .build()?;
            let document_filter = RestoreFilter::builder().document_data(true).build()?;
            host.seed_archive(PARTIAL_A);
            let mut edit = session.edit()?;
            edit.restore_objects_from_archive(&ARCHIVE_READER, Some(&source_filter))?;
            edit.restore_objects_from_archive(&ARCHIVE_READER, Some(&modification_filter))?;
            host.seed_archive(PARTIAL_B);
            edit.restore_objects_from_archive(&SECOND_ARCHIVE_READER, Some(&document_filter))?;
            edit.finish()?;
            let store_filter = session
                .store_filter_builder()
                .audio_source(source)
                .audio_modification(modification)
                .document_data(true)
                .build()?;
            session.store_objects_to_archive(&ARCHIVE_WRITER, Some(&store_filter))?;
            if trace.count("restore_objects") != 3 || trace.count("store_objects") != 1 {
                return Err(AraError::Peer("split archive callbacks were incomplete"));
            }
            Ok((8, 4, 4))
        },
    )
}

pub(super) fn drag_drop_import() -> Result<ScenarioReport, AraError> {
    with_document(
        "drag-drop-import",
        ApiGeneration::V23Final,
        |host, trace, session| {
            let source = {
                let mut edit = session.edit()?;
                let source = edit.create_audio_source(test_audio_source_properties()?)?;
                edit.finish()?;
                source
            };
            host.seed_archive(PARTIAL_A);
            let filter = RestoreFilter::builder()
                .audio_source("imported-source", "test-source")
                .build()?;
            let mut edit = session.edit()?;
            edit.restore_objects_from_archive(&ARCHIVE_READER, Some(&filter))?;
            edit.finish()?;
            let still_live = session.audio_source_ref(source).is_ok();
            if !still_live || trace.count("restore_audio_sources") != 1 {
                return Err(AraError::Peer("drag/import source was not restored"));
            }
            Ok((4, 2, 2))
        },
    )
}

pub(super) fn ara1_persistence() -> Result<ScenarioReport, AraError> {
    with_document(
        "ara1-persistence",
        ApiGeneration::V1Final,
        |host, trace, session| {
            host.seed_archive(ARA1_ARCHIVE);
            let restore = session.restore_document_from_archive(&ARCHIVE_READER)?;
            restore.finish()?;
            session.store_document_to_archive(&ARCHIVE_WRITER)?;
            if trace.count("restore_document") != 1 || trace.count("store_document") != 1 {
                return Err(AraError::Peer("ARA1 persistence callbacks were incomplete"));
            }
            Ok((3, 3, 3))
        },
    )
}
