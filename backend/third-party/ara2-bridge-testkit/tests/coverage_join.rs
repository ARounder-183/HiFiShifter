use ara2_bridge_testkit::coverage::{all_contract_tests, all_delegates, CoverageReport};

#[test]
fn every_public_slot_is_delegated_classified_and_reported() {
    let report = CoverageReport::build(
        ara2_bridge_sys::compatibility::RECORDS,
        &all_delegates(),
        &all_contract_tests(),
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("docs/conformance/interface-coverage.md");
    let freshness = std::fs::read_to_string(path)
        .map(|checked_in| checked_in.starts_with(&report.render_markdown()))
        .map_err(|error| error.to_string());
    assert!(
        report.semantic_gaps().is_empty() && freshness == Ok(true),
        "semantic gaps: {:#?}; report freshness: {freshness:?}",
        report.semantic_gaps()
    );
}

#[test]
fn missing_delegate_is_reported_by_interface_and_callback() {
    let mut delegates = all_delegates();
    let missing = delegates.remove(0);
    let report = CoverageReport::build(
        ara2_bridge_sys::compatibility::RECORDS,
        &delegates,
        &all_contract_tests(),
    );
    assert!(report.semantic_gaps().contains(&format!(
        "missing delegate: {}.{}",
        missing.surface, missing.c_name
    )));
}
