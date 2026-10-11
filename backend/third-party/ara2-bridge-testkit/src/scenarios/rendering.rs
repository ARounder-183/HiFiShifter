//! Playback and editor renderer assignment scenarios.

use super::catalog::{with_document, ScenarioReport};
use crate::{
    all_transformations, build_test_extension, test_audio_source_properties, TestPluginTrace,
};
use ara2_bridge_core::{
    ApiGeneration, AraError, AudioModificationProperties, MusicalContextProperties,
    PlaybackRegionProperties, RegionSequenceProperties,
};
use ara2_bridge_host::{
    DocumentSession, ExtensionRoles, PlaybackRegionHandle, RegionSequenceHandle, RendererRole,
};

pub(super) struct GraphHandles {
    pub sequence: RegionSequenceHandle,
    pub region: PlaybackRegionHandle,
}

pub(super) fn create_graph(
    session: &mut DocumentSession<'_, '_>,
    transformation_flags: i32,
) -> Result<GraphHandles, AraError> {
    let mut edit = session.edit()?;
    let context = edit.create_musical_context(MusicalContextProperties::new(
        Some("Scenario music"),
        0,
        None,
    )?)?;
    let sequence = edit.create_region_sequence(RegionSequenceProperties::new(
        Some("Scenario sequence"),
        0,
        edit.musical_context_ref(context)?,
        None,
    )?)?;
    let source = edit.create_audio_source(test_audio_source_properties()?)?;
    let modification = edit.create_audio_modification(
        source,
        AudioModificationProperties::new(Some("Scenario modification"), "scenario-modification")?,
    )?;
    let region = edit.create_playback_region(
        modification,
        PlaybackRegionProperties::for_ara2(
            transformation_flags,
            0.0,
            1.0,
            0.0,
            1.0,
            edit.region_sequence_ref(sequence)?,
            Some("Scenario region"),
            None,
        )?,
    )?;
    edit.finish()?;
    Ok(GraphHandles { sequence, region })
}

pub(super) fn playback_rendering() -> Result<ScenarioReport, AraError> {
    run_playback("playback-rendering", 0)
}

pub(super) fn playback_rendering_time_stretch() -> Result<ScenarioReport, AraError> {
    run_playback(
        "playback-rendering-time-stretch",
        i32::try_from(all_transformations().bits())
            .map_err(|_| AraError::InvalidArgument("transformation flags exceed ARA integer"))?,
    )
}

fn run_playback(name: &'static str, flags: i32) -> Result<ScenarioReport, AraError> {
    with_document(
        name,
        ApiGeneration::V23Final,
        |_, _: &TestPluginTrace, session| {
            let graph = create_graph(session, flags)?;
            let roles = ExtensionRoles::PLAYBACK_RENDERER;
            let (binding, lease) =
                build_test_extension(ApiGeneration::V23Final, roles.bits(), roles.bits())?;
            // SAFETY: binding and lease retain the complete native-facing extension backing.
            let extension = unsafe { session.bind_extension(binding.as_raw(), roles, roles)? };
            let assignment =
                extension.assign_playback_region(session, RendererRole::Playback, graph.region)?;
            let (head, tail) = session.playback_region_head_and_tail_time(graph.region)?;
            let assigned = binding.assignment_counts().0 == 1;
            drop(assignment);
            let removed = binding.assignment_counts().0 == 0;
            drop(extension);
            drop(binding);
            lease.destroy();
            if !(assigned && removed && head >= 0.0 && tail >= 0.0) {
                return Err(AraError::Peer("playback renderer scenario failed"));
            }
            Ok((7, 4, 4))
        },
    )
}
