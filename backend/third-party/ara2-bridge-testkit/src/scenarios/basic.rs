//! Basic capability-rich Rust host ↔ Rust plug-in smoke scenario.

use crate::{
    build_test_extension, build_test_factory, test_audio_source_properties, TestHost,
    TestPluginTrace,
};
use ara2_bridge_core::{
    ApiGeneration, AraError, AudioModificationProperties, DocumentProperties,
    MusicalContextProperties, Notes, PlaybackRegionProperties, RegionSequenceProperties,
};
use ara2_bridge_host::{DocumentSession, ExtensionRoles, RendererRole};

/// Observable results from [`basic_document_smoke`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasicDocumentReport {
    edit_cycles: usize,
    content_events_read: usize,
    analysis_progress_events: usize,
    sample_access_exercised: bool,
    extension_assignment_exercised: bool,
    controller_first_close_exercised: bool,
    companion_first_close_exercised: bool,
}

impl BasicDocumentReport {
    /// Returns the number of completed graph edit cycles.
    pub const fn edit_cycles(&self) -> usize {
        self.edit_cycles
    }
    /// Returns the number of typed plug-in content events read.
    pub const fn content_events_read(&self) -> usize {
        self.content_events_read
    }
    /// Returns the number of analysis progress callbacks received by the host.
    pub const fn analysis_progress_events(&self) -> usize {
        self.analysis_progress_events
    }
    /// Returns whether real host sample-reader callbacks were exercised.
    pub const fn sample_access_exercised(&self) -> bool {
        self.sample_access_exercised
    }
    /// Returns whether an extension renderer assignment was exercised.
    pub const fn extension_assignment_exercised(&self) -> bool {
        self.extension_assignment_exercised
    }
    /// Returns whether document/controller teardown preceded companion teardown.
    pub const fn controller_first_close_exercised(&self) -> bool {
        self.controller_first_close_exercised
    }
    /// Returns whether companion teardown preceded document/controller teardown.
    pub const fn companion_first_close_exercised(&self) -> bool {
        self.companion_first_close_exercised
    }
}

/// Runs the named factory, graph, analysis, sample, content, extension, and teardown smoke path.
pub fn basic_document_smoke() -> Result<BasicDocumentReport, AraError> {
    let generation = ApiGeneration::V23Final;
    let plugin_trace = TestPluginTrace::new();
    let factory = build_test_factory(plugin_trace.clone())?;
    let host = TestHost::new(generation)?;
    let loaded = host.load_factory(&factory)?;
    let mut session = DocumentSession::new(
        &loaded,
        host.services(),
        DocumentProperties::new(Some("Basic document"))?,
    )?;
    let (sequence, source, region) = {
        let mut edit = session.edit()?;
        let context =
            edit.create_musical_context(MusicalContextProperties::new(Some("Music"), 0, None)?)?;
        let sequence = edit.create_region_sequence(RegionSequenceProperties::new(
            Some("Sequence"),
            0,
            edit.musical_context_ref(context)?,
            None,
        )?)?;
        let source = edit.create_audio_source(test_audio_source_properties()?)?;
        let modification = edit.create_audio_modification(
            source,
            AudioModificationProperties::new(Some("Modification"), "basic-modification")?,
        )?;
        let region = edit.create_playback_region(
            modification,
            PlaybackRegionProperties::for_ara2(
                0,
                0.0,
                1.0,
                0.0,
                1.0,
                edit.region_sequence_ref(sequence)?,
                Some("Region"),
                None,
            )?,
        )?;
        edit.finish()?;
        (sequence, source, region)
    };

    session.set_audio_source_samples_access(source, true)?;
    let samples = host.read_source_samples(session.audio_source_ref(source)?, 0, 4)?;
    session.set_audio_source_samples_access(source, false)?;
    let sample_access_exercised = samples.len() == 2
        && samples.iter().all(|channel| channel.len() == 4)
        && host.trace().count("read_audio_samples") == 1;

    session.request_audio_source_content_analysis::<Notes>(source)?;
    session.notify_model_updates()?;
    let mut content_reader = session
        .audio_source_content_reader::<Notes>(source, None)?
        .ok_or(AraError::Peer("fixture did not return note content"))?;
    let content_events_read = content_reader.len();
    let _ = content_reader.event(0)?;
    drop(content_reader);

    {
        let mut edit = session.edit()?;
        edit.request_processing_algorithm(source, 1)?;
        edit.finish()?;
    }

    let roles = ExtensionRoles::all();
    let (binding, lease) = build_test_extension(generation, roles.bits(), roles.bits())?;
    // SAFETY: binding and lease retain the complete extension allocation through document close.
    let extension = unsafe { session.bind_extension(binding.as_raw(), roles, roles)? };
    let assignment = extension.assign_playback_region(&session, RendererRole::Playback, region)?;
    let sequence_assignment = extension.assign_region_sequence(&session, sequence)?;
    drop(assignment);
    drop(sequence_assignment);
    let extension_assignment_exercised = binding.assignment_counts() == (0, 0);

    session
        .close()
        .map_err(|_| AraError::Peer("controller-first document close failed"))?;
    drop(extension);
    drop(binding);
    lease.destroy();

    let mut companion_first = DocumentSession::new(
        &loaded,
        host.services(),
        DocumentProperties::new(Some("Companion-first close"))?,
    )?;
    let (binding, lease) = build_test_extension(generation, roles.bits(), roles.bits())?;
    // SAFETY: the host extension wrapper is explicitly dropped before companion backing below.
    let extension = unsafe { companion_first.bind_extension(binding.as_raw(), roles, roles)? };
    drop(extension);
    drop(binding);
    lease.destroy();
    companion_first
        .close()
        .map_err(|_| AraError::Peer("companion-first document close failed"))?;

    Ok(BasicDocumentReport {
        edit_cycles: 2,
        content_events_read,
        analysis_progress_events: host.trace().count("analysis_progress"),
        sample_access_exercised,
        extension_assignment_exercised,
        controller_first_close_exercised: true,
        companion_first_close_exercised: true,
    })
}
