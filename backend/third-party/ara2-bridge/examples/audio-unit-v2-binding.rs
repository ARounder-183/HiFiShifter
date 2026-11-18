use ara2_bridge::companion::audio_unit::AudioUnitPluginAdapter;
use ara2_bridge::companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
use ara2_bridge::core::AraError;
use ara2_bridge::plugin::FactoryBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build this example on macOS with `ARA_AUDIO_UNIT_SDK_DIR` set to the pinned SDK.
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
    // SAFETY: an Audio Unit module retains its published ARA factory for process lifetime.
    let association = unsafe { CompanionFactory::from_raw("Example ARA", &*factory.as_raw())? };
    let processor = CompanionProcessorBinding::new([association], CompanionRoles::all())?;
    let _adapter = AudioUnitPluginAdapter::new(processor, "Example ARA", |_, _| {
        Err(AraError::Unsupported(
            "install the plug-in's ExtensionBinding callback",
        ))
    })?;

    println!("Audio Unit v2 ARA property handler ready");
    Ok(())
}
