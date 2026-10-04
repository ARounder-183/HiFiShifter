//! Focused semantic traits implemented by safe ARA plug-in delegates.

mod content;
mod model;
mod persistence;

pub use content::{AnalysisProvider, ContentProvider};
pub use model::{
    AudioModifications, AudioSources, DocumentLifecycle, MusicalContexts, PlaybackRegions,
    RegionSequences,
};
pub use persistence::{PartialPersistence, Persistence};
