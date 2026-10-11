//! Named upstream and bridge-specific release scenarios.

use crate::{build_test_factory, TestHost, TestPluginTrace};
use ara2_bridge_core::{ApiGeneration, AraError, DocumentProperties};
use ara2_bridge_host::DocumentSession;

/// Observable result of one public-API scenario execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScenarioReport {
    name: &'static str,
    operations: usize,
    assertions: usize,
    expected_calls: usize,
    skip_count: usize,
}

impl ScenarioReport {
    pub(super) const fn success(
        name: &'static str,
        operations: usize,
        assertions: usize,
        expected_calls: usize,
    ) -> Self {
        Self {
            name,
            operations,
            assertions,
            expected_calls,
            skip_count: 0,
        }
    }

    /// Returns the stable scenario name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the number of public operations exercised.
    pub const fn operations(&self) -> usize {
        self.operations
    }

    /// Returns the number of scenario postconditions checked.
    pub const fn assertions(&self) -> usize {
        self.assertions
    }

    /// Returns the number of expected foreign calls observed.
    pub const fn expected_calls(&self) -> usize {
        self.expected_calls
    }

    /// Returns the number of capability skips taken by the runner.
    pub const fn skip_count(&self) -> usize {
        self.skip_count
    }
}

/// Static definition of one named conformance scenario.
#[derive(Clone, Copy)]
pub struct ScenarioDefinition {
    /// Stable kebab-case scenario name.
    pub name: &'static str,
    /// API generation exercised by the scenario.
    pub generation: ApiGeneration,
    /// Capability prerequisites supplied by the rich release fixture.
    pub required_capabilities: &'static [&'static str],
    /// Public-API runner for the scenario.
    pub run: fn() -> Result<ScenarioReport, AraError>,
}

/// Returns every named upstream-parity and bridge-specific release scenario.
pub const fn upstream_scenarios() -> &'static [ScenarioDefinition] {
    SCENARIOS
}

pub(super) fn with_document(
    name: &'static str,
    generation: ApiGeneration,
    operation: impl FnOnce(
        &TestHost,
        &TestPluginTrace,
        &mut DocumentSession<'_, '_>,
    ) -> Result<(usize, usize, usize), AraError>,
) -> Result<ScenarioReport, AraError> {
    let trace = TestPluginTrace::new();
    let factory = build_test_factory(trace.clone())?;
    let host = TestHost::new(generation)?;
    let loaded = host.load_factory(&factory)?;
    let mut session = DocumentSession::new(
        &loaded,
        host.services(),
        DocumentProperties::new(Some(name))?,
    )?;
    let (operations, assertions, expected_calls) = operation(&host, &trace, &mut session)?;
    session
        .close()
        .map_err(|_| AraError::Peer("scenario document close failed"))?;
    Ok(ScenarioReport::success(
        name,
        operations,
        assertions,
        expected_calls,
    ))
}

const ALL_CAPABILITIES: &[&str] = &[
    "content",
    "analysis",
    "partial-persistence",
    "processing-algorithms",
    "audio-file-chunks",
    "all-extension-roles",
];

const SCENARIOS: &[ScenarioDefinition] = &[
    scenario(
        "property-updates",
        ApiGeneration::V23Final,
        super::properties::property_updates,
    ),
    scenario(
        "content-updates",
        ApiGeneration::V23Final,
        super::properties::content_updates,
    ),
    scenario(
        "content-reading",
        ApiGeneration::V23Final,
        super::content::content_reading,
    ),
    scenario(
        "audio-modification-cloning",
        ApiGeneration::V23Final,
        super::properties::modification_cloning,
    ),
    scenario(
        "full-archive",
        ApiGeneration::V23Final,
        super::persistence::full_archive,
    ),
    scenario(
        "split-partial-archives",
        ApiGeneration::V23Final,
        super::persistence::split_partial_archives,
    ),
    scenario(
        "drag-drop-import",
        ApiGeneration::V23Final,
        super::persistence::drag_drop_import,
    ),
    scenario(
        "playback-rendering",
        ApiGeneration::V23Final,
        super::rendering::playback_rendering,
    ),
    scenario(
        "playback-rendering-time-stretch",
        ApiGeneration::V23Final,
        super::rendering::playback_rendering_time_stretch,
    ),
    scenario(
        "editor-view",
        ApiGeneration::V23Final,
        super::extensions::editor_view,
    ),
    scenario(
        "processing-algorithms",
        ApiGeneration::V23Final,
        super::processing::processing_algorithms,
    ),
    scenario(
        "audio-file-chunk-load",
        ApiGeneration::V23Final,
        super::processing::audio_file_chunk_load,
    ),
    scenario(
        "audio-file-chunk-save",
        ApiGeneration::V23Final,
        super::processing::audio_file_chunk_save,
    ),
    scenario(
        "basic-document",
        ApiGeneration::V23Final,
        super::properties::basic_document,
    ),
    scenario(
        "ara1-persistence",
        ApiGeneration::V1Final,
        super::persistence::ara1_persistence,
    ),
    scenario(
        "ara23-dirtiness",
        ApiGeneration::V23Final,
        super::properties::ara23_dirtiness,
    ),
    scenario(
        "extension-role-combinations",
        ApiGeneration::V23Final,
        super::extensions::role_combinations,
    ),
    scenario(
        "poisoning",
        ApiGeneration::V23Final,
        super::properties::poisoning,
    ),
    scenario(
        "controller-first-teardown",
        ApiGeneration::V23Final,
        super::extensions::controller_first_teardown,
    ),
    scenario(
        "companion-first-teardown",
        ApiGeneration::V23Final,
        super::extensions::companion_first_teardown,
    ),
];

const fn scenario(
    name: &'static str,
    generation: ApiGeneration,
    run: fn() -> Result<ScenarioReport, AraError>,
) -> ScenarioDefinition {
    ScenarioDefinition {
        name,
        generation,
        required_capabilities: ALL_CAPABILITIES,
        run,
    }
}
