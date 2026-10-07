//! Bounded ARA iXML parsing and canonical emission.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use quick_xml::escape::unescape;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashSet;

/// Typed ARA audio-file chunk errors.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ChunkError {
    /// A singleton schema element occurred more than once.
    #[error("duplicate XML element: {0}")]
    DuplicateElement(&'static str),
    /// Required schema data is absent.
    #[error("missing XML element: {0}")]
    MissingElement(&'static str),
    /// XML syntax or text is invalid.
    #[error("invalid ARA XML: {0}")]
    Invalid(&'static str),
    /// Input or decoded archive data exceeds configured limits.
    #[error("ARA XML limit exceeded: {0}")]
    Limit(&'static str),
}

/// Explicit parser allocation limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkLimits {
    /// Maximum XML input bytes.
    pub max_xml_bytes: usize,
    /// Maximum decoded archive bytes in one entry.
    pub max_archive_bytes: usize,
    /// Maximum number of audio-source records.
    pub max_entries: usize,
}

impl Default for ChunkLimits {
    fn default() -> Self {
        Self {
            max_xml_bytes: 16 * 1024 * 1024,
            max_archive_bytes: 64 * 1024 * 1024,
            max_entries: 65_536,
        }
    }
}

/// Optional display-only suggested plug-in metadata.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SuggestedPlugIn {
    /// Suggested plug-in display name.
    pub plug_in_name: Option<String>,
    /// Lowest compatible display version.
    pub lowest_supported_version: Option<String>,
    /// Manufacturer display name.
    pub manufacturer_name: Option<String>,
    /// Information URL.
    pub information_url: Option<String>,
}

/// One audio-source archive record from the ARA dictionary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioSourceArchive {
    document_archive_id: String,
    open_automatically: bool,
    create_distinct_audio_modification: bool,
    suggested_plug_in: Option<SuggestedPlugIn>,
    persistent_id: String,
    archive_data: Vec<u8>,
}

impl AudioSourceArchive {
    /// Returns the dictionary key.
    pub fn document_archive_id(&self) -> &str {
        &self.document_archive_id
    }
    /// Returns the automatic-open request.
    pub fn open_automatically(&self) -> bool {
        self.open_automatically
    }
    /// Returns the distinct-modification request.
    pub fn create_distinct_audio_modification(&self) -> bool {
        self.create_distinct_audio_modification
    }
    /// Returns suggested plug-in metadata.
    pub fn suggested_plug_in(&self) -> Option<&SuggestedPlugIn> {
        self.suggested_plug_in.as_ref()
    }
    /// Returns the current audio-source persistent ID.
    pub fn persistent_id(&self) -> &str {
        &self.persistent_id
    }
    /// Returns decoded plug-in archive bytes.
    pub fn archive_data(&self) -> &[u8] {
        &self.archive_data
    }
}

/// Ordered ARA audio-source archive dictionary.
#[derive(Clone, Debug, Default)]
pub struct AraChunkSet {
    entries: Vec<AudioSourceArchive>,
    preserved: Option<PreservedDocument>,
}

impl PartialEq for AraChunkSet {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Eq for AraChunkSet {}

impl AraChunkSet {
    /// Parses with default production limits.
    pub fn parse(input: &[u8]) -> Result<Self, ChunkError> {
        Self::parse_with_limits(input, ChunkLimits::default())
    }

    /// Parses with explicit allocation limits.
    pub fn parse_with_limits(input: &[u8], limits: ChunkLimits) -> Result<Self, ChunkError> {
        if input.len() > limits.max_xml_bytes {
            return Err(ChunkError::Limit("XML input"));
        }
        if contains_ascii_case_insensitive(input, b"<!DOCTYPE")
            || contains_ascii_case_insensitive(input, b"<!ENTITY")
        {
            return Err(ChunkError::Invalid(
                "document types and entities are forbidden",
            ));
        }

        let mut reader = Reader::from_reader(input);
        reader.config_mut().trim_text(true);
        let mut state = ParseState::new(limits);
        loop {
            match reader.read_event() {
                Ok(Event::Start(start)) => {
                    let name = local_name(start.name().as_ref())?;
                    state.start(&name, false)?;
                }
                Ok(Event::Empty(start)) => {
                    let name = local_name(start.name().as_ref())?;
                    state.start(&name, true)?;
                    state.end(&name)?;
                }
                Ok(Event::Text(text)) => {
                    if state.text_target.is_some() {
                        let decoded = text
                            .decode()
                            .map_err(|_| ChunkError::Invalid("text is not valid XML encoding"))?;
                        let text = unescape(&decoded)
                            .map_err(|_| ChunkError::Invalid("invalid XML entity"))?;
                        state.text.push_str(&text);
                    }
                }
                Ok(Event::CData(text)) => {
                    if state.text_target.is_some() {
                        let text = std::str::from_utf8(text.as_ref())
                            .map_err(|_| ChunkError::Invalid("CDATA is not UTF-8"))?;
                        state.text.push_str(text);
                    }
                }
                Ok(Event::End(end)) => {
                    let name = local_name(end.name().as_ref())?;
                    state.end(&name)?;
                }
                Ok(Event::DocType(_)) => {
                    return Err(ChunkError::Invalid("document types are forbidden"));
                }
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(_) => return Err(ChunkError::Invalid("malformed XML")),
            }
        }
        let mut set = state.finish()?;
        set.preserved = PreservedDocument::from_input(input);
        Ok(set)
    }

    /// Returns an entry by document archive ID.
    pub fn get(&self, archive_id: &str) -> Option<&AudioSourceArchive> {
        self.entries
            .iter()
            .find(|entry| entry.document_archive_id == archive_id)
    }

    /// Returns archive IDs in dictionary order.
    pub fn archive_ids(&self) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .map(|entry| entry.document_archive_id.as_str())
    }

    /// Emits canonical iXML containing the ordered ARA dictionary.
    pub fn emit(&self) -> Vec<u8> {
        if let Some(preserved) = &self.preserved {
            let dictionary = preserved
                .sources
                .as_ref()
                .and_then(|sources| sources.emit(&self.entries))
                .unwrap_or_else(|| self.emit_audio_sources().into_bytes());
            let mut output = Vec::with_capacity(
                preserved.prefix.len()
                    + preserved.ara_start.len()
                    + preserved.before_sources.len()
                    + dictionary.len()
                    + preserved.after_sources.len()
                    + preserved.ara_end.len()
                    + preserved.suffix.len(),
            );
            output.extend_from_slice(&preserved.prefix);
            output.extend_from_slice(&preserved.ara_start);
            output.extend_from_slice(&preserved.before_sources);
            output.extend_from_slice(&dictionary);
            output.extend_from_slice(&preserved.after_sources);
            output.extend_from_slice(&preserved.ara_end);
            output.extend_from_slice(&preserved.suffix);
            return output;
        }
        let dictionary = self.emit_audio_sources();
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<BWFXML><ARA>{dictionary}</ARA></BWFXML>\n"
        )
        .into_bytes()
    }

    fn emit_audio_sources(&self) -> String {
        let mut output = String::from("<audioSources>");
        for entry in &self.entries {
            output.push_str("<audioSource><documentArchiveID>");
            escape(&mut output, &entry.document_archive_id);
            output.push_str("</documentArchiveID><openAutomatically>");
            output.push_str(if entry.open_automatically {
                "true"
            } else {
                "false"
            });
            output.push_str("</openAutomatically><createDistinctAudioModification>");
            output.push_str(if entry.create_distinct_audio_modification {
                "true"
            } else {
                "false"
            });
            output.push_str("</createDistinctAudioModification>");
            if let Some(suggested) = &entry.suggested_plug_in {
                output.push_str("<suggestedPlugIn>");
                optional_element(&mut output, "plugInName", suggested.plug_in_name.as_deref());
                optional_element(
                    &mut output,
                    "lowestSupportedVersion",
                    suggested.lowest_supported_version.as_deref(),
                );
                optional_element(
                    &mut output,
                    "manufacturerName",
                    suggested.manufacturer_name.as_deref(),
                );
                optional_element(
                    &mut output,
                    "informationURL",
                    suggested.information_url.as_deref(),
                );
                output.push_str("</suggestedPlugIn>");
            }
            output.push_str("<persistentID>");
            escape(&mut output, &entry.persistent_id);
            output.push_str("</persistentID><archiveData>");
            output.push_str(&STANDARD.encode(&entry.archive_data));
            output.push_str("</archiveData></audioSource>");
        }
        output.push_str("</audioSources>");
        output
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreservedDocument {
    prefix: Vec<u8>,
    ara_start: Vec<u8>,
    before_sources: Vec<u8>,
    sources: Option<PreservedAudioSources>,
    after_sources: Vec<u8>,
    ara_end: Vec<u8>,
    suffix: Vec<u8>,
}

impl PreservedDocument {
    fn from_input(input: &[u8]) -> Option<Self> {
        let ara = find_element(input, b"ARA")?;
        let inner = &input[ara.start_end..ara.end_start];
        let (before_sources, sources, after_sources) =
            if let Some(sources) = find_element(inner, b"audioSources") {
                (
                    inner[..sources.start].to_vec(),
                    PreservedAudioSources::from_input(&inner[sources.start..sources.end_end]),
                    inner[sources.end_end..].to_vec(),
                )
            } else {
                (inner.to_vec(), None, Vec::new())
            };
        Some(Self {
            prefix: input[..ara.start].to_vec(),
            ara_start: input[ara.start..ara.start_end].to_vec(),
            before_sources,
            sources,
            after_sources,
            ara_end: input[ara.end_start..ara.end_end].to_vec(),
            suffix: input[ara.end_end..].to_vec(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreservedAudioSources {
    start: Vec<u8>,
    gaps: Vec<Vec<u8>>,
    entries: Vec<Vec<u8>>,
    end: Vec<u8>,
}

impl PreservedAudioSources {
    fn from_input(input: &[u8]) -> Option<Self> {
        let root = find_element(input, b"audioSources")?;
        let body = &input[root.start_end..root.end_start];
        let children = direct_elements(body)?;
        let sources: Vec<_> = children
            .into_iter()
            .filter(|child| child.local == "audioSource")
            .collect();
        let mut gaps = Vec::with_capacity(sources.len() + 1);
        let mut entries = Vec::with_capacity(sources.len());
        let mut cursor = 0;
        for source in sources {
            gaps.push(body[cursor..source.bounds.start].to_vec());
            entries.push(body[source.bounds.start..source.bounds.end_end].to_vec());
            cursor = source.bounds.end_end;
        }
        gaps.push(body[cursor..].to_vec());
        Some(Self {
            start: input[root.start..root.start_end].to_vec(),
            gaps,
            entries,
            end: input[root.end_start..root.end_end].to_vec(),
        })
    }

    fn emit(&self, entries: &[AudioSourceArchive]) -> Option<Vec<u8>> {
        if entries.len() != self.entries.len() || self.gaps.len() != entries.len() + 1 {
            return None;
        }
        let mut output = Vec::new();
        output.extend_from_slice(&self.start);
        for (index, entry) in entries.iter().enumerate() {
            output.extend_from_slice(&self.gaps[index]);
            output.extend_from_slice(&canonicalize_source(&self.entries[index], entry)?);
        }
        output.extend_from_slice(self.gaps.last()?);
        output.extend_from_slice(&self.end);
        Some(output)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DirectElement {
    local: String,
    bounds: ElementBounds,
}

fn direct_elements(input: &[u8]) -> Option<Vec<DirectElement>> {
    let mut reader = Reader::from_reader(input);
    let mut depth = 0usize;
    let mut open: Option<DirectElement> = None;
    let mut elements = Vec::new();
    loop {
        let event_start = usize::try_from(reader.buffer_position()).ok()?;
        let event = reader.read_event().ok()?;
        let event_end = usize::try_from(reader.buffer_position()).ok()?;
        match event {
            Event::Start(start) => {
                if depth == 0 {
                    open = Some(DirectElement {
                        local: local_name(start.name().as_ref()).ok()?,
                        bounds: ElementBounds {
                            start: event_start,
                            start_end: event_end,
                            end_start: event_end,
                            end_end: event_end,
                        },
                    });
                }
                depth = depth.checked_add(1)?;
            }
            Event::Empty(start) if depth == 0 => elements.push(DirectElement {
                local: local_name(start.name().as_ref()).ok()?,
                bounds: ElementBounds {
                    start: event_start,
                    start_end: event_end,
                    end_start: event_end,
                    end_end: event_end,
                },
            }),
            Event::End(_) => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let mut element = open.take()?;
                    element.bounds.end_start = event_start;
                    element.bounds.end_end = event_end;
                    elements.push(element);
                }
            }
            Event::Eof => return (depth == 0 && open.is_none()).then_some(elements),
            _ => {}
        }
    }
}

fn canonicalize_source(input: &[u8], entry: &AudioSourceArchive) -> Option<Vec<u8>> {
    let mut output = input.to_vec();
    replace_direct_text(&mut output, "documentArchiveID", &entry.document_archive_id)?;
    if has_direct_child(&output, "openAutomatically")? {
        replace_direct_text(
            &mut output,
            "openAutomatically",
            if entry.open_automatically {
                "true"
            } else {
                "false"
            },
        )?;
    } else {
        insert_direct_text_after(
            &mut output,
            "documentArchiveID",
            "openAutomatically",
            if entry.open_automatically {
                "true"
            } else {
                "false"
            },
        )?;
    }
    if has_direct_child(&output, "createDistinctAudioModification")? {
        replace_direct_text(
            &mut output,
            "createDistinctAudioModification",
            if entry.create_distinct_audio_modification {
                "true"
            } else {
                "false"
            },
        )?;
    } else {
        insert_direct_text_after(
            &mut output,
            "openAutomatically",
            "createDistinctAudioModification",
            if entry.create_distinct_audio_modification {
                "true"
            } else {
                "false"
            },
        )?;
    }
    replace_direct_text(&mut output, "persistentID", &entry.persistent_id)?;
    replace_direct_text(
        &mut output,
        "archiveData",
        &STANDARD.encode(&entry.archive_data),
    )?;
    Some(output)
}

fn source_root_and_children(input: &[u8]) -> Option<(ElementBounds, Vec<DirectElement>)> {
    let root = find_element(input, b"audioSource")?;
    let children = direct_elements(&input[root.start_end..root.end_start])?;
    Some((root, children))
}

fn has_direct_child(input: &[u8], local: &str) -> Option<bool> {
    let (_, children) = source_root_and_children(input)?;
    Some(children.iter().any(|child| child.local == local))
}

fn replace_direct_text(input: &mut Vec<u8>, local: &str, text: &str) -> Option<()> {
    let (root, children) = source_root_and_children(input)?;
    let child = children.into_iter().find(|child| child.local == local)?;
    let start = root.start_end.checked_add(child.bounds.start)?;
    let start_end = root.start_end.checked_add(child.bounds.start_end)?;
    let end_start = root.start_end.checked_add(child.bounds.end_start)?;
    let end_end = root.start_end.checked_add(child.bounds.end_end)?;
    let mut replacement = Vec::new();
    replacement.extend_from_slice(&input[start..start_end]);
    let mut escaped = String::new();
    escape(&mut escaped, text);
    replacement.extend_from_slice(escaped.as_bytes());
    replacement.extend_from_slice(&input[end_start..end_end]);
    input.splice(start..end_end, replacement);
    Some(())
}

fn insert_direct_text_after(
    input: &mut Vec<u8>,
    after_local: &str,
    new_local: &str,
    text: &str,
) -> Option<()> {
    let (root, children) = source_root_and_children(input)?;
    let child = children
        .into_iter()
        .find(|child| child.local == after_local)?;
    let insertion = root.start_end.checked_add(child.bounds.end_end)?;
    let qualified_parent = qualified_start_name(&input[root.start..root.start_end])?;
    let prefix_end = qualified_parent
        .iter()
        .rposition(|byte| *byte == b':')
        .map_or(0, |position| position + 1);
    let prefix = &qualified_parent[..prefix_end];
    let mut element = Vec::new();
    element.push(b'<');
    element.extend_from_slice(prefix);
    element.extend_from_slice(new_local.as_bytes());
    element.push(b'>');
    let mut escaped = String::new();
    escape(&mut escaped, text);
    element.extend_from_slice(escaped.as_bytes());
    element.extend_from_slice(b"</");
    element.extend_from_slice(prefix);
    element.extend_from_slice(new_local.as_bytes());
    element.push(b'>');
    input.splice(insertion..insertion, element);
    Some(())
}

fn qualified_start_name(start_tag: &[u8]) -> Option<&[u8]> {
    let name_start = start_tag.iter().position(|byte| *byte == b'<')? + 1;
    let name_end = start_tag[name_start..]
        .iter()
        .position(|byte| byte.is_ascii_whitespace() || matches!(*byte, b'>' | b'/'))?
        + name_start;
    Some(&start_tag[name_start..name_end])
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ElementBounds {
    start: usize,
    start_end: usize,
    end_start: usize,
    end_end: usize,
}

fn find_element(input: &[u8], wanted_local: &[u8]) -> Option<ElementBounds> {
    let mut cursor = 0;
    while cursor < input.len() {
        let relative = input[cursor..].iter().position(|byte| *byte == b'<')?;
        let start = cursor + relative;
        let name_start = start + 1;
        let first = *input.get(name_start)?;
        if matches!(first, b'/' | b'!' | b'?') {
            cursor = name_start + 1;
            continue;
        }
        let name_end = input[name_start..]
            .iter()
            .position(|byte| byte.is_ascii_whitespace() || matches!(*byte, b'>' | b'/'))?
            + name_start;
        let qualified = &input[name_start..name_end];
        let local = qualified
            .rsplit(|byte| *byte == b':')
            .next()
            .unwrap_or(qualified);
        let start_end = tag_end(input, name_end)?;
        if local == wanted_local {
            if input.get(start_end.saturating_sub(2)..start_end) == Some(b"/>") {
                return Some(ElementBounds {
                    start,
                    start_end,
                    end_start: start_end,
                    end_end: start_end,
                });
            }
            let mut closing = Vec::with_capacity(qualified.len() + 3);
            closing.extend_from_slice(b"</");
            closing.extend_from_slice(qualified);
            let closing_relative = find_bytes(&input[start_end..], &closing)?;
            let end_start = start_end + closing_relative;
            let end_end = tag_end(input, end_start + closing.len())?;
            return Some(ElementBounds {
                start,
                start_end,
                end_start,
                end_end,
            });
        }
        cursor = start_end;
    }
    None
}

fn tag_end(input: &[u8], start: usize) -> Option<usize> {
    let mut quote = None;
    for (offset, byte) in input.get(start..)?.iter().copied().enumerate() {
        match (quote, byte) {
            (None, b'\'' | b'"') => quote = Some(byte),
            (Some(open), close) if open == close => quote = None,
            (None, b'>') => return Some(start + offset + 1),
            _ => {}
        }
    }
    None
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextTarget {
    DocumentArchiveId,
    OpenAutomatically,
    CreateDistinct,
    PlugInName,
    LowestVersion,
    Manufacturer,
    InformationUrl,
    PersistentId,
    ArchiveData,
}

impl TextTarget {
    fn schema_name(self) -> &'static str {
        match self {
            Self::DocumentArchiveId => "documentArchiveID",
            Self::OpenAutomatically => "openAutomatically",
            Self::CreateDistinct => "createDistinctAudioModification",
            Self::PlugInName => "plugInName",
            Self::LowestVersion => "lowestSupportedVersion",
            Self::Manufacturer => "manufacturerName",
            Self::InformationUrl => "informationURL",
            Self::PersistentId => "persistentID",
            Self::ArchiveData => "archiveData",
        }
    }
}

#[derive(Default)]
struct EntryBuilder {
    seen: HashSet<&'static str>,
    document_archive_id: Option<String>,
    open_automatically: Option<bool>,
    create_distinct: Option<bool>,
    suggested_seen: bool,
    suggested: SuggestedPlugIn,
    persistent_id: Option<String>,
    archive_data: Option<Vec<u8>>,
}

struct ParseState {
    limits: ChunkLimits,
    stack: Vec<String>,
    ara_count: usize,
    audio_sources_count: usize,
    current: Option<EntryBuilder>,
    text_target: Option<TextTarget>,
    text: String,
    entries: Vec<AudioSourceArchive>,
}

impl ParseState {
    fn new(limits: ChunkLimits) -> Self {
        Self {
            limits,
            stack: Vec::new(),
            ara_count: 0,
            audio_sources_count: 0,
            current: None,
            text_target: None,
            text: String::new(),
            entries: Vec::new(),
        }
    }

    fn parent(&self) -> Option<&str> {
        self.stack.last().map(String::as_str)
    }

    fn start(&mut self, name: &str, empty: bool) -> Result<(), ChunkError> {
        let parent = self.parent().map(str::to_owned);
        if name == "ARA" {
            self.ara_count += 1;
            if self.ara_count > 1 {
                return Err(ChunkError::DuplicateElement("ARA"));
            }
        } else if name == "audioSources" && parent.as_deref() == Some("ARA") {
            self.audio_sources_count += 1;
            if self.audio_sources_count > 1 {
                return Err(ChunkError::DuplicateElement("audioSources"));
            }
        } else if name == "audioSource" && parent.as_deref() == Some("audioSources") {
            if self.current.is_some() {
                return Err(ChunkError::Invalid("nested audioSource"));
            }
            if self.entries.len() >= self.limits.max_entries {
                return Err(ChunkError::Limit("audio-source entry count"));
            }
            self.current = Some(EntryBuilder::default());
        } else if name == "suggestedPlugIn" && self.current.is_some() {
            let current = self.current.as_mut().expect("checked above");
            if current.suggested_seen {
                return Err(ChunkError::DuplicateElement("suggestedPlugIn"));
            }
            current.suggested_seen = true;
        } else if let Some(target) = target_for(name, parent.as_deref(), self.current.is_some()) {
            let current = self.current.as_mut().expect("target requires entry");
            if !current.seen.insert(target.schema_name()) {
                return Err(ChunkError::DuplicateElement(target.schema_name()));
            }
            self.text_target = Some(target);
            self.text.clear();
        }
        self.stack.push(name.to_owned());
        if empty && self.text_target.is_none() {
            self.text.clear();
        }
        Ok(())
    }

    fn end(&mut self, name: &str) -> Result<(), ChunkError> {
        if self.stack.last().map(String::as_str) != Some(name) {
            return Err(ChunkError::Invalid("mismatched XML element"));
        }
        if self
            .text_target
            .is_some_and(|target| target.schema_name() == name)
        {
            let target = self.text_target.take().expect("checked above");
            let text = std::mem::take(&mut self.text);
            self.assign(target, text)?;
        }
        if name == "audioSource" {
            let builder = self
                .current
                .take()
                .ok_or(ChunkError::Invalid("audioSource end without start"))?;
            self.entries.push(builder.finish(self.limits)?);
        }
        self.stack.pop();
        Ok(())
    }

    fn assign(&mut self, target: TextTarget, text: String) -> Result<(), ChunkError> {
        let current = self
            .current
            .as_mut()
            .ok_or(ChunkError::Invalid("content outside audioSource"))?;
        match target {
            TextTarget::DocumentArchiveId => current.document_archive_id = Some(text),
            TextTarget::OpenAutomatically => current.open_automatically = Some(parse_bool(&text)?),
            TextTarget::CreateDistinct => current.create_distinct = Some(parse_bool(&text)?),
            TextTarget::PlugInName => current.suggested.plug_in_name = optional_text(text),
            TextTarget::LowestVersion => {
                current.suggested.lowest_supported_version = optional_text(text)
            }
            TextTarget::Manufacturer => current.suggested.manufacturer_name = optional_text(text),
            TextTarget::InformationUrl => current.suggested.information_url = optional_text(text),
            TextTarget::PersistentId => current.persistent_id = Some(text),
            TextTarget::ArchiveData => {
                let compact: String = text
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .collect();
                let decoded = STANDARD
                    .decode(compact.as_bytes())
                    .map_err(|_| ChunkError::Invalid("archiveData is not MIME Base64"))?;
                if decoded.len() > self.limits.max_archive_bytes {
                    return Err(ChunkError::Limit("decoded archiveData"));
                }
                current.archive_data = Some(decoded);
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<AraChunkSet, ChunkError> {
        if !self.stack.is_empty() || self.current.is_some() {
            return Err(ChunkError::Invalid("unclosed XML element"));
        }
        let mut ids = HashSet::with_capacity(self.entries.len());
        for entry in &self.entries {
            if !ids.insert(entry.document_archive_id.as_str()) {
                return Err(ChunkError::Invalid("duplicate documentArchiveID"));
            }
        }
        Ok(AraChunkSet {
            entries: self.entries,
            preserved: None,
        })
    }
}

impl EntryBuilder {
    fn finish(self, _limits: ChunkLimits) -> Result<AudioSourceArchive, ChunkError> {
        let document_archive_id = self
            .document_archive_id
            .ok_or(ChunkError::MissingElement("documentArchiveID"))?;
        let persistent_id = self
            .persistent_id
            .ok_or(ChunkError::MissingElement("persistentID"))?;
        validate_id(&document_archive_id)?;
        validate_id(&persistent_id)?;
        let archive_data = self
            .archive_data
            .ok_or(ChunkError::MissingElement("archiveData"))?;
        let suggested = self.suggested;
        let suggested_plug_in = self.suggested_seen.then_some(suggested).filter(|value| {
            value.plug_in_name.is_some()
                || value.lowest_supported_version.is_some()
                || value.manufacturer_name.is_some()
                || value.information_url.is_some()
        });
        Ok(AudioSourceArchive {
            document_archive_id,
            open_automatically: self.open_automatically.unwrap_or(false),
            create_distinct_audio_modification: self.create_distinct.unwrap_or(false),
            suggested_plug_in,
            persistent_id,
            archive_data,
        })
    }
}

fn target_for(name: &str, parent: Option<&str>, has_entry: bool) -> Option<TextTarget> {
    if !has_entry {
        return None;
    }
    match (name, parent) {
        ("documentArchiveID", Some("audioSource")) => Some(TextTarget::DocumentArchiveId),
        ("openAutomatically", Some("audioSource")) => Some(TextTarget::OpenAutomatically),
        ("createDistinctAudioModification", Some("audioSource")) => {
            Some(TextTarget::CreateDistinct)
        }
        ("persistentID", Some("audioSource")) => Some(TextTarget::PersistentId),
        ("archiveData", Some("audioSource")) => Some(TextTarget::ArchiveData),
        ("plugInName", Some("suggestedPlugIn")) => Some(TextTarget::PlugInName),
        ("lowestSupportedVersion", Some("suggestedPlugIn")) => Some(TextTarget::LowestVersion),
        ("manufacturerName", Some("suggestedPlugIn")) => Some(TextTarget::Manufacturer),
        ("informationURL", Some("suggestedPlugIn")) => Some(TextTarget::InformationUrl),
        _ => None,
    }
}

fn local_name(name: &[u8]) -> Result<String, ChunkError> {
    let local = name.rsplit(|byte| *byte == b':').next().unwrap_or(name);
    std::str::from_utf8(local)
        .map(str::to_owned)
        .map_err(|_| ChunkError::Invalid("element name is not UTF-8"))
}

fn parse_bool(text: &str) -> Result<bool, ChunkError> {
    match text.trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ChunkError::Invalid("boolean must be true or false")),
    }
}

fn validate_id(id: &str) -> Result<(), ChunkError> {
    if id.is_empty() || !id.is_ascii() || id.contains('\0') {
        return Err(ChunkError::Invalid("persistent IDs must be nonempty ASCII"));
    }
    Ok(())
}

fn optional_text(text: String) -> Option<String> {
    (!text.is_empty()).then_some(text)
}

fn contains_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

fn escape(output: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\'' => output.push_str("&apos;"),
            _ => output.push(character),
        }
    }
}

fn optional_element(output: &mut String, name: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        output.push('<');
        output.push_str(name);
        output.push('>');
        escape(output, value);
        output.push_str("</");
        output.push_str(name);
        output.push('>');
    }
}
