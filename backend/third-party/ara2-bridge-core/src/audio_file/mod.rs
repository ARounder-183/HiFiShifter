//! ARA metadata stored in audio-file iXML chunks.

mod container;
mod path;
mod xml;

pub use container::{read_ixml, read_ixml_with_limit, rewrite_ixml, AudioFileError, AudioFileKind};
pub use path::{replace_ara_in_path, PathRewriteError};
pub use xml::{AraChunkSet, AudioSourceArchive, ChunkError, ChunkLimits, SuggestedPlugIn};
