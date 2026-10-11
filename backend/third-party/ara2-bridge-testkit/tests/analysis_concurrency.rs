use ara2_bridge_core::{ApiGeneration, DocumentProperties, Notes};
use ara2_bridge_host::DocumentSession;
use ara2_bridge_testkit::{
    build_test_factory, test_audio_source_properties, TestHost, TestPluginTrace,
};

#[test]
fn concurrent_sessions_cancel_real_analysis_jobs_without_late_callbacks() {
    let workers = (0..4)
        .map(|index| {
            std::thread::spawn(move || {
                let trace = TestPluginTrace::new();
                let factory = build_test_factory(trace.clone()).unwrap();
                let host = TestHost::new(ApiGeneration::V23Final).unwrap();
                let loaded = host.load_factory(&factory).unwrap();
                let mut session = DocumentSession::new(
                    &loaded,
                    host.services(),
                    DocumentProperties::new(Some(&format!("Analysis worker {index}"))).unwrap(),
                )
                .unwrap();
                let source = {
                    let mut edit = session.edit().unwrap();
                    let source = edit
                        .create_audio_source(test_audio_source_properties().unwrap())
                        .unwrap();
                    edit.finish().unwrap();
                    source
                };

                session
                    .set_audio_source_samples_access(source, true)
                    .unwrap();
                session
                    .request_audio_source_content_analysis::<Notes>(source)
                    .unwrap();
                session
                    .set_audio_source_samples_access(source, false)
                    .unwrap();
                session.notify_model_updates().unwrap();

                assert_eq!(trace.count("request_analysis"), 1);
                assert_eq!(trace.count("cancel_analysis"), 1);
                assert_eq!(host.trace().count("analysis_progress"), 0);
                session.close().unwrap();
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        worker.join().unwrap();
    }
}
