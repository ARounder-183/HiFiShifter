use ara2_bridge_core::ApiGeneration;
use ara2_bridge_testkit::native::{
    run_cpp_host_rust_plugin, run_rust_host_cpp_plugin, NativeScenario, NativeScenarioConfig,
    NativeScenarioResult,
};

fn config(scenario: NativeScenario) -> NativeScenarioConfig {
    NativeScenarioConfig {
        generation: ApiGeneration::V23Final,
        scenario,
    }
}

fn assert_complete(scenario: NativeScenario, result: NativeScenarioResult) {
    assert_eq!(result.generation, ApiGeneration::V23Final);
    assert_eq!(result.scenario, scenario.name());
    assert!(result.callback_count >= 3, "{result:#?}");
    assert!(result.diagnostics.is_empty(), "{result:#?}");
    assert_eq!(result.live_objects, 0, "{result:#?}");
}

#[test]
fn rust_test_host_drives_celemony_cpp_test_plugin() {
    for &scenario in NativeScenario::buildable() {
        assert_complete(
            scenario,
            run_rust_host_cpp_plugin(config(scenario)).expect("C++ TestPlugIn pairing"),
        );
    }
}

#[test]
fn celemony_cpp_test_host_drives_rust_test_plugin() {
    for &scenario in NativeScenario::buildable() {
        assert_complete(
            scenario,
            run_cpp_host_rust_plugin(config(scenario)).expect("C++ TestHost pairing"),
        );
    }
}
