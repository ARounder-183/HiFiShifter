use ara2_bridge_testkit::scenarios::upstream_scenarios;

#[test]
fn concurrent_editor_renderer_updates_and_both_teardown_orders_are_leak_free() {
    let workers = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                let scenarios = upstream_scenarios();
                for name in [
                    "editor-view",
                    "controller-first-teardown",
                    "companion-first-teardown",
                ] {
                    let scenario = scenarios
                        .iter()
                        .find(|scenario| scenario.name == name)
                        .unwrap();
                    let report = (scenario.run)().unwrap();
                    assert_eq!(report.name(), name);
                    assert!(report.operations() > 0);
                    assert!(report.assertions() > 0);
                    assert!(report.expected_calls() > 0);
                    assert_eq!(report.skip_count(), 0);
                }
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        worker.join().unwrap();
    }
}
