//! Protocol boundary. The App implements `EditorBackend`; this crate never owns a project copy.
pub mod contract;
pub mod discovery;
pub mod server;
pub mod stdio;
pub mod voicebank;

pub use contract::*;
use std::{future::Future, pin::Pin};

pub trait EditorBackend: Send + Sync + 'static {
    fn execute(
        &self,
        request: EditorRequest,
    ) -> Pin<Box<dyn Future<Output = serde_json::Value> + Send>>;
}
