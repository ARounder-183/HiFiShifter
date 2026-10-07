//! Generated callback coverage and document-controller vtable construction.

pub(crate) mod callbacks;
pub(crate) mod generated_callbacks;
mod vtable;

pub use callbacks::{CallbackContract, Delegate};
pub use generated_callbacks::{PLUGIN_CONTRACT_TESTS, PLUGIN_DELEGATES};
pub use vtable::{document_controller_interface, ControllerCapabilities, ControllerInterface};
