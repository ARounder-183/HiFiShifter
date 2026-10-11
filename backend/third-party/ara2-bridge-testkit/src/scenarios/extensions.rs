//! Editor roles, role combinations, and teardown-order scenarios.

use super::catalog::{with_document, ScenarioReport};
use super::rendering::create_graph;
use crate::build_test_extension;
use ara2_bridge_core::{ApiGeneration, AraError};
use ara2_bridge_host::{ExtensionRoles, RendererRole};

pub(super) fn editor_view() -> Result<ScenarioReport, AraError> {
    with_document("editor-view", ApiGeneration::V23Final, |_, _, session| {
        let graph = create_graph(session, 0)?;
        let roles = ExtensionRoles::EDITOR_RENDERER | ExtensionRoles::EDITOR_VIEW;
        let (binding, lease) = build_test_extension(
            ApiGeneration::V23Final,
            ExtensionRoles::all().bits(),
            roles.bits(),
        )?;
        // SAFETY: binding and lease retain callback backing through all view calls.
        let extension = unsafe { session.bind_extension(binding.as_raw(), roles, roles)? };
        let region =
            extension.assign_playback_region(session, RendererRole::Editor, graph.region)?;
        let sequence = extension.assign_region_sequence(session, graph.sequence)?;
        extension.notify_selection(session, &[graph.region], &[graph.sequence], None)?;
        extension.notify_hidden_region_sequences(session, &[graph.sequence])?;
        let notified =
            binding.view_selection().is_some() && binding.hidden_region_sequences().len() == 1;
        drop(region);
        drop(sequence);
        let removed = binding.assignment_counts() == (0, 0);
        drop(extension);
        drop(binding);
        lease.destroy();
        if !(notified && removed) {
            return Err(AraError::Peer("editor view scenario failed"));
        }
        Ok((8, 2, 6))
    })
}

pub(super) fn role_combinations() -> Result<ScenarioReport, AraError> {
    let combinations = [
        ExtensionRoles::PLAYBACK_RENDERER,
        ExtensionRoles::EDITOR_RENDERER,
        ExtensionRoles::EDITOR_VIEW,
        ExtensionRoles::all(),
    ];
    let mut assertions = 0;
    for roles in combinations {
        let (binding, lease) = build_test_extension(
            ApiGeneration::V23Final,
            ExtensionRoles::all().bits(),
            roles.bits(),
        )?;
        assertions += usize::from(binding.enabled_roles().bits() == roles.bits());
        drop(binding);
        lease.destroy();
    }
    if assertions != combinations.len() {
        return Err(AraError::Peer("extension role mapping mismatch"));
    }
    Ok(ScenarioReport::success(
        "extension-role-combinations",
        combinations.len(),
        assertions,
        combinations.len(),
    ))
}

pub(super) fn controller_first_teardown() -> Result<ScenarioReport, AraError> {
    let report = crate::scenarios::basic_document_smoke()?;
    if !report.controller_first_close_exercised() {
        return Err(AraError::Peer(
            "controller-first teardown was not exercised",
        ));
    }
    Ok(ScenarioReport::success(
        "controller-first-teardown",
        3,
        1,
        2,
    ))
}

pub(super) fn companion_first_teardown() -> Result<ScenarioReport, AraError> {
    let report = crate::scenarios::basic_document_smoke()?;
    if !report.companion_first_close_exercised() {
        return Err(AraError::Peer("companion-first teardown was not exercised"));
    }
    Ok(ScenarioReport::success("companion-first-teardown", 3, 1, 2))
}
