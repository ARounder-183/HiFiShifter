//! Runtime-owned graph records.

use ara2_bridge_core::{
    AudioModificationKind, AudioSourceKind, Handle, MusicalContextKind, RegionSequenceKind,
};

pub(crate) struct Node<T> {
    pub(crate) value: Option<T>,
    pub(crate) active: bool,
    pub(crate) children: usize,
}

impl<T> Node<T> {
    pub(crate) const fn provisional() -> Self {
        Self {
            value: None,
            active: true,
            children: 0,
        }
    }

    pub(crate) fn live_value(&mut self) -> &mut T {
        self.value
            .as_mut()
            .expect("only committed runtime records are publicly reachable")
    }
}

pub(crate) struct RegionSequenceNode<T> {
    pub(crate) node: Node<T>,
    pub(crate) context: Handle<MusicalContextKind>,
}

pub(crate) struct AudioModificationNode<T> {
    pub(crate) node: Node<T>,
    pub(crate) source: Handle<AudioSourceKind>,
}

pub(crate) struct PlaybackRegionNode<T> {
    pub(crate) node: Node<T>,
    pub(crate) modification: Handle<AudioModificationKind>,
    pub(crate) sequence: Handle<RegionSequenceKind>,
}
