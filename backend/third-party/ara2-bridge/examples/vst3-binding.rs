use ara2_bridge::companion::vst3::Vst3MainFactoryAdapter;
use ara2_bridge::companion::CompanionFactory;
use ara2_bridge::plugin::FactoryBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure `ARA_VST3_SDK_DIR` to the pinned VST3 SDK before compiling this example.
    let factory = Box::leak(Box::new(
        FactoryBuilder::new("Example ARA", "org.example.archive")
            .display(
                "Example ARA",
                "Example Audio",
                "https://example.invalid",
                "2.0",
            )
            .build()?,
    ));
    // SAFETY: a VST3 module publishes its ARA factory for the module/process lifetime.
    let association = unsafe { CompanionFactory::from_raw("Example ARA", &*factory.as_raw())? };
    let adapter = Vst3MainFactoryAdapter::new("Example ARA", association)?;

    assert!(!adapter.as_raw().is_null());
    println!("VST3 ARA main-factory class ready");
    Ok(())
}
