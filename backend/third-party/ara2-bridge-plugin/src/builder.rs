//! Validating builders for ARA factories and binary registries.

use crate::factory::ControllerCreator;
use crate::{Factory, Plugin, PluginEntry, PluginModel};
use ara2_bridge_core::{ApiGeneration, AraError, PlaybackTransformationFlags};
use std::collections::BTreeSet;
use std::ffi::CString;
use std::sync::Arc;

/// Capabilities advertised by an immutable ARA factory.
#[derive(Clone, Debug)]
pub struct FactoryCapabilities {
    pub(crate) analyzable_content_types: Vec<i32>,
    pub(crate) playback_transformations: PlaybackTransformationFlags,
    pub(crate) stores_audio_file_chunks: bool,
}

impl Default for FactoryCapabilities {
    fn default() -> Self {
        Self {
            analyzable_content_types: Vec::new(),
            playback_transformations: PlaybackTransformationFlags::empty(),
            stores_audio_file_chunks: false,
        }
    }
}

impl FactoryCapabilities {
    /// Advertises the content types accepted for analysis.
    pub fn with_analyzable_content_types(
        mut self,
        content_types: impl IntoIterator<Item = i32>,
    ) -> Self {
        self.analyzable_content_types = content_types.into_iter().collect();
        self
    }

    /// Advertises supported playback transformations.
    pub fn with_playback_transformations(mut self, flags: PlaybackTransformationFlags) -> Self {
        self.playback_transformations = flags;
        self
    }

    /// Advertises whether the plug-in persists data in audio-file chunks.
    pub fn with_audio_file_chunk_storage(mut self, supported: bool) -> Self {
        self.stores_audio_file_chunks = supported;
        self
    }
}

pub(crate) struct FactorySpec {
    pub(crate) factory_id: CString,
    pub(crate) archive_id: CString,
    pub(crate) compatible_ids: Vec<CString>,
    pub(crate) lowest: ApiGeneration,
    pub(crate) highest: ApiGeneration,
    pub(crate) capabilities: FactoryCapabilities,
    pub(crate) name: CString,
    pub(crate) manufacturer: CString,
    pub(crate) information_url: CString,
    pub(crate) version: CString,
    pub(crate) create_controller: Option<Arc<ControllerCreator>>,
}

/// Builder for one immutable ARA factory.
pub struct FactoryBuilder {
    factory_id: String,
    archive_id: String,
    compatible_ids: Vec<String>,
    lowest: ApiGeneration,
    highest: ApiGeneration,
    capabilities: FactoryCapabilities,
    name: String,
    manufacturer: String,
    information_url: String,
    version: String,
    create_controller: Option<Arc<ControllerCreator>>,
}

impl FactoryBuilder {
    /// Starts a factory using ARA 2.0 Final through 2.3 Final by default.
    pub fn new(factory_id: impl Into<String>, archive_id: impl Into<String>) -> Self {
        let factory_id = factory_id.into();
        Self {
            name: factory_id.clone(),
            factory_id,
            archive_id: archive_id.into(),
            compatible_ids: Vec::new(),
            lowest: ApiGeneration::V2Final,
            highest: ApiGeneration::V23Final,
            capabilities: FactoryCapabilities::default(),
            manufacturer: String::new(),
            information_url: String::new(),
            version: String::new(),
            create_controller: None,
        }
    }

    /// Sets the user-visible plug-in metadata retained by the factory.
    pub fn display(
        mut self,
        name: impl Into<String>,
        manufacturer: impl Into<String>,
        information_url: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        self.name = name.into();
        self.manufacturer = manufacturer.into();
        self.information_url = information_url.into();
        self.version = version.into();
        self
    }

    /// Sets the inclusive range of released API generations supported by the factory.
    pub fn generations(mut self, lowest: ApiGeneration, highest: ApiGeneration) -> Self {
        self.lowest = lowest;
        self.highest = highest;
        self
    }

    /// Adds a document archive identifier accepted during restore.
    pub fn compatible_archive_id(mut self, archive_id: impl Into<String>) -> Self {
        self.compatible_ids.push(archive_id.into());
        self
    }

    /// Sets the capability values advertised by the factory.
    pub fn capabilities(mut self, capabilities: FactoryCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Registers a fallible constructor for each document controller created by this factory.
    ///
    /// The constructor runs on the host's ARA model thread and must return a fresh plug-in value;
    /// document state and optional capability handlers are never shared between controllers.
    pub fn document_controller<P>(
        mut self,
        create: impl Fn() -> Result<Plugin<P>, AraError> + Send + Sync + 'static,
    ) -> Self
    where
        P: PluginModel + 'static,
    {
        self.create_controller = Some(Arc::new(move |generation, factory, host, properties| {
            let plugin = create()?;
            // SAFETY: the factory callback validates initialization and forwards ARA-owned input
            // pointers whose storage is required to remain live for the controller lifetime.
            unsafe { plugin.create_document_controller(generation, factory, host, properties) }
        }));
        self
    }

    /// Validates and builds stable ABI backing for this factory.
    pub fn build(self) -> Result<Factory, AraError> {
        validate_id(&self.factory_id, "factory ID")?;
        validate_id(&self.archive_id, "document archive ID")?;
        if self.lowest > self.highest {
            return Err(AraError::InvalidArgument(
                "lowest API generation exceeds highest generation",
            ));
        }
        if !self.lowest.supported_on_target() || !self.highest.supported_on_target() {
            return Err(AraError::Unsupported(
                "factory generation range is unavailable on this target",
            ));
        }
        validate_capabilities(&self.capabilities)?;

        let mut compatible = BTreeSet::new();
        let compatible_ids = self
            .compatible_ids
            .into_iter()
            .map(|id| {
                validate_id(&id, "compatible document archive ID")?;
                if id == self.archive_id || !compatible.insert(id.clone()) {
                    return Err(AraError::InvalidArgument(
                        "document archive IDs must be unique",
                    ));
                }
                c_string(id, "compatible document archive ID contains NUL")
            })
            .collect::<Result<Vec<_>, _>>()?;

        Factory::new(FactorySpec {
            factory_id: c_string(self.factory_id, "factory ID contains NUL")?,
            archive_id: c_string(self.archive_id, "document archive ID contains NUL")?,
            compatible_ids,
            lowest: self.lowest,
            highest: self.highest,
            capabilities: self.capabilities,
            name: display_string(self.name, "plug-in name")?,
            manufacturer: display_string(self.manufacturer, "manufacturer name")?,
            information_url: display_string(self.information_url, "information URL")?,
            version: display_string(self.version, "version")?,
            create_controller: self.create_controller,
        })
    }
}

fn validate_id(value: &str, kind: &'static str) -> Result<(), AraError> {
    if value.is_empty() || !value.is_ascii() {
        return Err(AraError::InvalidArgument(match kind {
            "factory ID" => "factory ID must be nonempty ASCII",
            "document archive ID" => "document archive ID must be nonempty ASCII",
            _ => "compatible document archive ID must be nonempty ASCII",
        }));
    }
    Ok(())
}

fn c_string(value: String, nul_error: &'static str) -> Result<CString, AraError> {
    CString::new(value).map_err(|_| AraError::InvalidArgument(nul_error))
}

fn display_string(value: String, kind: &'static str) -> Result<CString, AraError> {
    if value.is_empty() {
        return Err(AraError::InvalidArgument(match kind {
            "plug-in name" => "plug-in name must be nonempty",
            "manufacturer name" => "manufacturer name must be nonempty",
            "information URL" => "information URL must be nonempty",
            _ => "version must be nonempty",
        }));
    }
    c_string(value, "factory display string contains NUL")
}

fn validate_capabilities(capabilities: &FactoryCapabilities) -> Result<(), AraError> {
    let flags = capabilities.playback_transformations;
    if flags.contains(PlaybackTransformationFlags::REFLECT_TEMPO)
        && !flags.contains(PlaybackTransformationFlags::TIMESTRETCH)
    {
        return Err(AraError::InvalidArgument(
            "tempo reflection requires timestretch support",
        ));
    }
    let fades = flags & PlaybackTransformationFlags::CONTENT_FADES;
    if !fades.is_empty() && fades != PlaybackTransformationFlags::CONTENT_FADES {
        return Err(AraError::InvalidArgument(
            "content-based head and tail fades must be advertised together",
        ));
    }
    let mut content_types = BTreeSet::new();
    if capabilities
        .analyzable_content_types
        .iter()
        .any(|content_type| !content_types.insert(*content_type))
    {
        return Err(AraError::InvalidArgument(
            "analyzable content types must be unique",
        ));
    }
    Ok(())
}

/// Builder for all factories exported by one plug-in binary.
#[derive(Default)]
pub struct PluginRegistryBuilder {
    factories: Vec<Factory>,
}

impl PluginRegistryBuilder {
    /// Adds one already validated factory to the binary registry.
    pub fn factory(mut self, factory: Factory) -> Self {
        self.factories.push(factory);
        self
    }

    /// Rejects duplicate factory IDs and completes the registry.
    pub fn build(self) -> Result<PluginRegistry, AraError> {
        let mut ids = BTreeSet::new();
        if self
            .factories
            .iter()
            .any(|factory| !ids.insert(factory.id().to_owned()))
        {
            return Err(AraError::InvalidArgument("duplicate factory ID"));
        }
        Ok(PluginRegistry {
            factories: self.factories,
        })
    }
}

/// Binary-level collection of immutable factories and independent entries.
pub struct PluginRegistry {
    factories: Vec<Factory>,
}

impl PluginRegistry {
    /// Starts an empty plug-in binary registry.
    pub fn builder() -> PluginRegistryBuilder {
        PluginRegistryBuilder::default()
    }

    /// Finds an immutable factory by persistent ID.
    pub fn factory(&self, id: &str) -> Option<&Factory> {
        self.factories.iter().find(|factory| factory.id() == id)
    }

    /// Finds one factory's independent initialization entry by persistent ID.
    pub fn entry(&self, id: &str) -> Option<&PluginEntry> {
        self.factory(id).map(Factory::entry)
    }

    /// Iterates over all factories in registration order.
    pub fn factories(&self) -> impl ExactSizeIterator<Item = &Factory> {
        self.factories.iter()
    }
}
