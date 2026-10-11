//! Reusable public-API interoperability scenarios.

mod basic;
mod catalog;
mod content;
mod extensions;
mod persistence;
mod processing;
mod properties;
mod rendering;

pub use basic::{basic_document_smoke, BasicDocumentReport};
pub use catalog::{upstream_scenarios, ScenarioDefinition, ScenarioReport};
