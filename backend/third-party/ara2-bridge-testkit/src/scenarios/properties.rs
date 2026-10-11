//! Property, content-update, graph-cloning, dirtiness, and poisoning scenarios.

use super::basic_document_smoke;
use super::catalog::{with_document, ScenarioReport};
use crate::{test_audio_source_properties, TestHost, TestPluginTrace};
use ara2_bridge_core::{
    ApiGeneration, AraError, AudioModificationProperties, ContentUpdateScopes, DocumentProperties,
};
use ara2_bridge_host::DocumentSession;

pub(super) fn property_updates() -> Result<ScenarioReport, AraError> {
    with_document(
        "property-updates",
        ApiGeneration::V23Final,
        |_, trace, session| {
            let mut edit = session.edit()?;
            edit.update_document_properties(DocumentProperties::new(Some("Updated document"))?)?;
            let source = edit.create_audio_source(test_audio_source_properties()?)?;
            edit.update_audio_source(source, test_audio_source_properties()?)?;
            edit.finish()?;
            let assertions = usize::from(trace.count("update_document") == 1)
                + usize::from(trace.count("update_audio_source") == 1);
            if assertions != 2 {
                return Err(AraError::Peer("property updates were not delegated"));
            }
            Ok((4, assertions, 2))
        },
    )
}

pub(super) fn content_updates() -> Result<ScenarioReport, AraError> {
    with_document(
        "content-updates",
        ApiGeneration::V23Final,
        |_, trace, session| {
            let mut edit = session.edit()?;
            let source = edit.create_audio_source(test_audio_source_properties()?)?;
            edit.update_audio_source_content(source, None, ContentUpdateScopes::empty())?;
            edit.finish()?;
            if trace.count("update_audio_source_content") != 1 {
                return Err(AraError::Peer("content update was not delegated"));
            }
            Ok((3, 1, 1))
        },
    )
}

pub(super) fn modification_cloning() -> Result<ScenarioReport, AraError> {
    with_document(
        "audio-modification-cloning",
        ApiGeneration::V23Final,
        |_, trace, session| {
            let mut edit = session.edit()?;
            let source = edit.create_audio_source(test_audio_source_properties()?)?;
            let original = edit.create_audio_modification(
                source,
                AudioModificationProperties::new(Some("Original"), "clone-original")?,
            )?;
            let clone = edit.clone_audio_modification(
                original,
                AudioModificationProperties::new(Some("Clone"), "clone-copy")?,
            )?;
            edit.finish()?;
            let distinct = session.audio_modification_ref(original)?
                != session.audio_modification_ref(clone)?;
            if !distinct || trace.count("clone_audio_modification") != 1 {
                return Err(AraError::Peer("audio-modification clone was not distinct"));
            }
            Ok((4, 2, 1))
        },
    )
}

pub(super) fn basic_document() -> Result<ScenarioReport, AraError> {
    let report = basic_document_smoke()?;
    let assertions = usize::from(report.sample_access_exercised())
        + usize::from(report.extension_assignment_exercised())
        + usize::from(report.controller_first_close_exercised())
        + usize::from(report.companion_first_close_exercised());
    if assertions != 4 {
        return Err(AraError::Peer("basic document smoke assertions failed"));
    }
    Ok(ScenarioReport::success("basic-document", 8, assertions, 7))
}

pub(super) fn ara23_dirtiness() -> Result<ScenarioReport, AraError> {
    with_document(
        "ara23-dirtiness",
        ApiGeneration::V23Final,
        |host, trace, session| {
            let mut edit = session.edit()?;
            let _ = edit.create_audio_source(test_audio_source_properties()?)?;
            edit.finish()?;
            trace.mark_document_dirty()?;
            session.notify_model_updates()?;
            if host.trace().count("audio_source_changed") != 1
                || host.trace().count("document_data_changed") != 1
            {
                return Err(AraError::Peer("ARA 2.3 dirtiness was not delivered"));
            }
            Ok((4, 2, 2))
        },
    )
}

pub(super) fn poisoning() -> Result<ScenarioReport, AraError> {
    let trace = TestPluginTrace::new();
    trace.reject_next_audio_source_after_host_callback();
    let factory = crate::build_test_factory(trace.clone())?;
    let host = TestHost::new(ApiGeneration::V23Final)?;
    let loaded = host.load_factory(&factory)?;
    let mut session = DocumentSession::new(
        &loaded,
        host.services(),
        DocumentProperties::new(Some("poisoning"))?,
    )?;
    let mut edit = session.edit()?;
    let rejected = edit
        .create_audio_source(test_audio_source_properties()?)
        .is_err();
    edit.finish()?;
    let poisoned = session.is_poisoned();
    let close_failed = session.close().is_err();
    if !(rejected && poisoned && close_failed) {
        return Err(AraError::Peer(
            "reentrant failure did not quarantine the session",
        ));
    }
    Ok(ScenarioReport::success("poisoning", 3, 3, 2))
}
