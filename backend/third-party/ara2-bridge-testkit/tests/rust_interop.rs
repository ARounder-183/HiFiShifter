use ara2_bridge_testkit::scenarios::basic_document_smoke;

#[test]
fn rust_host_and_plugin_complete_the_basic_document_smoke_scenario() {
    let report = basic_document_smoke().unwrap();

    assert_eq!(report.edit_cycles(), 2);
    assert_eq!(report.content_events_read(), 1);
    assert_eq!(report.analysis_progress_events(), 3);
    assert!(report.sample_access_exercised());
    assert!(report.extension_assignment_exercised());
    assert!(report.controller_first_close_exercised());
    assert!(report.companion_first_close_exercised());
}
