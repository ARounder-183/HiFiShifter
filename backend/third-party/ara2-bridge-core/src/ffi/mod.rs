//! Narrow foreign-storage validation boundary.

mod scalar;
mod sized;
mod slice;
mod string;

pub use scalar::AraBool;
pub use sized::{SizedInput, SizedRecord};
pub use slice::ForeignSlice;
pub use string::ForeignStr;
