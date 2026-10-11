//! Deterministic interface coverage joins used by conformance and release tooling.

use ara2_bridge_sys::compatibility::CompatibilityRecord;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// One safe implementation associated with a released ARA callback slot.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CoverageDelegate {
    /// Released interface containing the callback.
    pub surface: &'static str,
    /// Exact callback field name from the ARA headers.
    pub c_name: &'static str,
    /// Public bridge implementation responsible for the callback.
    pub implementation: &'static str,
}

/// Behavioral evidence classes required for released callback slots.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContractClass {
    /// Valid input reaches the expected safe implementation.
    Positive,
    /// Version prefixes and permitted absent callbacks are checked.
    PrefixAndAbsence,
    /// Malformed foreign values are rejected with the documented sentinel.
    MalformedInput,
    /// User or peer failures retain their typed/fallback behavior.
    UserOrPeerFailure,
    /// Panics or native exceptions cannot unwind through the ABI.
    PanicOrException,
    /// Lifecycle and thread restrictions are enforced.
    LifecycleAndThread,
    /// References and retained backing are released in both teardown orders.
    Teardown,
}

/// How behavioral evidence applies to a callback slot.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceScope {
    /// The named test invokes this exact callback slot.
    PerSlot,
    /// The named tests exercise a shared adapter used by every slot in the interface group.
    SharedMechanism,
}

/// One behavioral test classification associated with a released callback slot.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ContractTest {
    /// Released interface containing the callback.
    pub surface: &'static str,
    /// Exact callback field name from the ARA headers.
    pub c_name: &'static str,
    /// Test target containing the behavioral evidence.
    pub target: &'static str,
    /// Exact `repository/path.rs#test_function` evidence identifiers.
    pub test_ids: &'static [&'static str],
    /// Whether evidence is callback-specific or applies through a shared adapter.
    pub scope: EvidenceScope,
    /// Shared implementation path that makes interface-wide evidence applicable.
    pub mechanism: &'static str,
    /// Behavioral classes exercised by the target group.
    pub classes: &'static [ContractClass],
}

/// Complete joined evidence for one released callback slot.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CoverageEntry {
    /// Released interface containing the callback.
    pub surface: &'static str,
    /// Exact callback field name from the ARA headers.
    pub c_name: &'static str,
    /// Safe implementation responsible for the callback.
    pub implementation: &'static str,
    /// Test target containing behavioral evidence.
    pub test_target: &'static str,
    /// Exact test-function identifiers checked by release tooling.
    pub test_ids: &'static [&'static str],
    /// Whether the tests invoke this slot or its shared adapter.
    pub evidence_scope: EvidenceScope,
    /// Shared implementation path covered by the evidence.
    pub mechanism: &'static str,
    /// Behavioral evidence classes exercised for the interface group.
    pub classes: &'static [ContractClass],
}

/// Joined callback coverage report.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CoverageReport {
    entries: Vec<CoverageEntry>,
    #[serde(skip)]
    gaps: Vec<String>,
}

impl CoverageReport {
    /// Joins released callback slots to safe implementations and contract tests.
    pub fn build(
        records: &'static [CompatibilityRecord],
        delegates: &[CoverageDelegate],
        contracts: &[ContractTest],
    ) -> Self {
        let slots = records
            .iter()
            .flat_map(|record| {
                record
                    .callbacks
                    .iter()
                    .map(move |callback| (record.surface, *callback))
            })
            .collect::<BTreeSet<_>>();
        let mut gaps = Vec::new();
        let delegate_map = unique_delegates(delegates, &mut gaps);
        let contract_map = unique_contracts(contracts, &mut gaps);

        for (surface, c_name) in delegate_map.keys().chain(contract_map.keys()) {
            if !slots.contains(&(*surface, *c_name)) {
                gaps.push(format!(
                    "inventory contains unknown slot: {surface}.{c_name}"
                ));
            }
        }

        let mut entries = Vec::new();
        for (surface, c_name) in slots {
            let delegate = delegate_map.get(&(surface, c_name));
            let contract = contract_map.get(&(surface, c_name));
            if delegate.is_none() {
                gaps.push(format!("missing delegate: {surface}.{c_name}"));
            }
            if contract.is_none() {
                gaps.push(format!("missing contract test: {surface}.{c_name}"));
            }
            if let (Some(delegate), Some(contract)) = (delegate, contract) {
                if contract.classes.is_empty() {
                    gaps.push(format!("empty contract classification: {surface}.{c_name}"));
                }
                if contract.test_ids.is_empty() {
                    gaps.push(format!("missing test identifiers: {surface}.{c_name}"));
                }
                if contract.scope == EvidenceScope::SharedMechanism
                    && contract.mechanism.trim().is_empty()
                {
                    gaps.push(format!("missing shared mechanism: {surface}.{c_name}"));
                }
                entries.push(CoverageEntry {
                    surface,
                    c_name,
                    implementation: delegate.implementation,
                    test_target: contract.target,
                    test_ids: contract.test_ids,
                    evidence_scope: contract.scope,
                    mechanism: contract.mechanism,
                    classes: contract.classes,
                });
            }
        }
        gaps.sort();
        gaps.dedup();
        Self { entries, gaps }
    }

    /// Returns all semantic gaps in stable lexical order.
    pub fn semantic_gaps(&self) -> &[String] {
        &self.gaps
    }

    /// Returns the joined slot entries in stable interface/callback lexical order.
    pub fn entries(&self) -> &[CoverageEntry] {
        &self.entries
    }

    /// Renders the human-readable callback report deterministically.
    pub fn render_markdown(&self) -> String {
        let mut output = format!(
            "# ARA Interface Coverage\n\nReleased callback slots: {}\n\n",
            self.entries.len()
        );
        output.push_str(
            "| Interface | Callback | Safe implementation | Contract target | Classes |\n",
        );
        output.push_str("| --- | --- | --- | --- | --- |\n");
        for entry in &self.entries {
            let classes = entry
                .classes
                .iter()
                .map(|class| format!("{class:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let tests = entry.test_ids.join("<br>");
            output.push_str(&format!(
                "| `{}` | `{}` | `{}` | `{}`<br>{}<br>{:?}: `{}` | {} |\n",
                entry.surface,
                entry.c_name,
                entry.implementation,
                entry.test_target,
                tests,
                entry.evidence_scope,
                entry.mechanism,
                classes
            ));
        }
        output
    }

    /// Renders the machine-readable callback report deterministically.
    pub fn render_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self).map(|json| format!("{json}\n"))
    }
}

fn unique_delegates<'a>(
    delegates: &'a [CoverageDelegate],
    gaps: &mut Vec<String>,
) -> BTreeMap<(&'static str, &'static str), &'a CoverageDelegate> {
    let mut map = BTreeMap::new();
    for delegate in delegates {
        if map
            .insert((delegate.surface, delegate.c_name), delegate)
            .is_some()
        {
            gaps.push(format!(
                "duplicate delegate: {}.{}",
                delegate.surface, delegate.c_name
            ));
        }
    }
    map
}

fn unique_contracts<'a>(
    contracts: &'a [ContractTest],
    gaps: &mut Vec<String>,
) -> BTreeMap<(&'static str, &'static str), &'a ContractTest> {
    let mut map = BTreeMap::new();
    for contract in contracts {
        if map
            .insert((contract.surface, contract.c_name), contract)
            .is_some()
        {
            gaps.push(format!(
                "duplicate contract test: {}.{}",
                contract.surface, contract.c_name
            ));
        }
    }
    map
}

const COMPLETE_CLASSES: &[ContractClass] = &[
    ContractClass::Positive,
    ContractClass::PrefixAndAbsence,
    ContractClass::MalformedInput,
    ContractClass::UserOrPeerFailure,
    ContractClass::PanicOrException,
    ContractClass::LifecycleAndThread,
    ContractClass::Teardown,
];

const FACTORY_CLASSES: &[ContractClass] = &[
    ContractClass::Positive,
    ContractClass::PrefixAndAbsence,
    ContractClass::MalformedInput,
    ContractClass::PanicOrException,
    ContractClass::LifecycleAndThread,
    ContractClass::Teardown,
];

/// Returns the safe delegate inventory used by the coverage join.
pub fn all_delegates() -> Vec<CoverageDelegate> {
    slot_pairs()
        .into_iter()
        .filter_map(|(surface, c_name)| {
            delegate_implementation(surface, c_name).map(|implementation| CoverageDelegate {
                surface,
                c_name,
                implementation,
            })
        })
        .collect()
}

/// Returns the behavioral contract-test inventory used by the coverage join.
pub fn all_contract_tests() -> Vec<ContractTest> {
    slot_pairs()
        .into_iter()
        .filter_map(|(surface, c_name)| {
            contract_evidence(surface).map(|(target, test_ids, mechanism, classes)| ContractTest {
                surface,
                c_name,
                target,
                test_ids,
                scope: EvidenceScope::SharedMechanism,
                mechanism,
                classes,
            })
        })
        .collect()
}

fn slot_pairs() -> BTreeSet<(&'static str, &'static str)> {
    ara2_bridge_sys::compatibility::RECORDS
        .iter()
        .flat_map(|record| {
            record
                .callbacks
                .iter()
                .map(move |callback| (record.surface, *callback))
        })
        .collect()
}

fn delegate_implementation(surface: &str, c_name: &str) -> Option<&'static str> {
    match surface {
        "ARAFactory" => Some("ara2_bridge_plugin::Factory"),
        "ARADocumentControllerInterface" => ara2_bridge_plugin::PLUGIN_DELEGATES
            .iter()
            .any(|delegate| delegate.c_name == c_name)
            .then_some("ara2_bridge_plugin::ControllerInterface"),
        "ARAAudioAccessControllerInterface"
        | "ARAArchivingControllerInterface"
        | "ARAContentAccessControllerInterface"
        | "ARAModelUpdateControllerInterface"
        | "ARAPlaybackControllerInterface" => ara2_bridge_host::host_callback_manifest()
            .contains(&c_name)
            .then_some("ara2_bridge_host::HostServices"),
        "ARAPlugInExtensionInterface"
        | "ARAPlaybackRendererInterface"
        | "ARAEditorRendererInterface"
        | "ARAEditorViewInterface" => {
            Some("ara2_bridge_plugin::ExtensionBinding / ara2_bridge_host::ExtensionController")
        }
        _ => None,
    }
}

fn contract_evidence(
    surface: &str,
) -> Option<(
    &'static str,
    &'static [&'static str],
    &'static str,
    &'static [ContractClass],
)> {
    match surface {
        "ARAFactory" => Some((
            "ARA factory callbacks",
            &[
                "ara2-bridge-plugin/tests/factory.rs#each_factory_entry_has_independent_generation_state",
                "ara2-bridge-plugin/tests/factory.rs#raw_factory_pointer_and_callbacks_are_stable_across_balanced_reinitialization",
                "ara2-bridge-plugin/tests/factory.rs#invalid_ranges_capabilities_duplicates_and_assert_addresses_are_rejected",
            ],
            "ara2_bridge_plugin::factory::{initialize,uninitialize,create_document_controller}",
            FACTORY_CLASSES,
        )),
        "ARADocumentControllerInterface" => Some((
            "generated document-controller dispatch",
            &[
                "ara2-bridge-testkit/tests/plugin_contract.rs#raw_fixture_drives_all_54_document_controller_callbacks",
                "ara2-bridge-testkit/tests/plugin_contract.rs#every_callback_rejects_a_null_controller_with_its_abi_fallback",
                "ara2-bridge-host/tests/plugin_dispatch.rs#minimal_controller_loads_every_supported_generation_prefix",
                "ara2-bridge-plugin/src/ffi/callbacks.rs#generated_shells_call_named_delegate_and_contain_panics",
                "ara2-bridge-core/tests/dispatch.rs#expected_errors_map_to_method_sentinels_without_poisoning",
                "ara2-bridge-core/tests/lifecycle.rs#model_operations_reject_the_wrong_thread",
                "ara2-bridge-host/tests/plugin_dispatch.rs#factory_creates_identity_checked_controller_and_drop_balances_destruction",
            ],
            "ara2_bridge_plugin::ffi::callbacks::dispatch + ara2_bridge_host::plugin::generated_dispatch",
            COMPLETE_CLASSES,
        )),
        "ARAAudioAccessControllerInterface"
        | "ARAArchivingControllerInterface"
        | "ARAContentAccessControllerInterface"
        | "ARAModelUpdateControllerInterface"
        | "ARAPlaybackControllerInterface" => {
            Some((
                "shared host-service callback dispatch",
                &[
                    "ara2-bridge-host/tests/host_callbacks.rs#every_host_slot_has_a_dispatcher_and_contract_class",
                    "ara2-bridge-host/tests/host_callback_fallbacks.rs#every_host_callback_has_a_null_host_fallback",
                    "ara2-bridge-host/tests/services_builder.rs#required_services_are_stable_and_optional_services_are_absent",
                    "ara2-bridge-host/tests/services_builder.rs#legacy_archive_prefix_omits_the_ara2_archive_id_tail",
                    "ara2-bridge-host/tests/services_builder.rs#a_panicking_service_quarantines_only_its_document",
                    "ara2-bridge-host/tests/audio_access.rs#planar_reads_silence_out_of_range_portions_and_failures",
                    "ara2-bridge-core/tests/lifecycle.rs#sample_content_and_teardown_states_are_checked",
                ],
                "ara2_bridge_host::services::dispatch::with_state",
                COMPLETE_CLASSES,
            ))
        }
        "ARAPlugInExtensionInterface"
        | "ARAPlaybackRendererInterface"
        | "ARAEditorRendererInterface"
        | "ARAEditorViewInterface" => {
            Some((
                "shared extension-role callback dispatch",
                &[
                    "ara2-bridge-plugin/tests/extensions.rs#role_calls_validate_assignment_and_controller_lifetime",
                    "ara2-bridge-plugin/tests/extensions.rs#interface_storage_survives_either_owner_destruction_order",
                    "ara2-bridge-plugin/tests/extensions.rs#ara1_binding_exposes_legacy_extension_prefix",
                    "ara2-bridge-plugin/tests/extensions.rs#editor_view_callbacks_copy_selection_and_hidden_sequence_arrays",
                    "ara2-bridge-host/tests/extensions.rs#role_sets_are_validated_against_returned_interface_pairs",
                    "ara2-bridge-host/tests/extensions.rs#ara2_roles_assign_graph_objects_and_copy_view_state",
                    "ara2-bridge-testkit/tests/editor_renderer_concurrency.rs#concurrent_editor_renderer_updates_and_both_teardown_orders_are_leak_free",
                ],
                "ara2_bridge_plugin::extension::ExtensionBinding::with_state + ara2_bridge_host::extension::ExtensionController",
                COMPLETE_CLASSES,
            ))
        }
        _ => None,
    }
}
