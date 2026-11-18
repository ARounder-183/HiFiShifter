use ara2_bridge::companion::clap::sys::CLAP_EXT_ARA_FACTORY;
use ara2_bridge::companion::clap::ClapAraEntry;
use ara2_bridge::companion::CompanionFactory;
use ara2_bridge::plugin::FactoryBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factory = Box::leak(Box::new(
        FactoryBuilder::new("org.example.ara", "org.example.archive")
            .display(
                "Example ARA",
                "Example Audio",
                "https://example.invalid",
                "2.0",
            )
            .build()?,
    ));
    // SAFETY: dynamic-library factory storage is intentionally retained for process lifetime.
    let association = unsafe { CompanionFactory::from_raw("org.example.ara", &*factory.as_raw())? };
    let entry = ClapAraEntry::new([association])?;

    assert_eq!(entry.factory_count(), 1);
    println!("publish {CLAP_EXT_ARA_FACTORY} from the CLAP entry");
    Ok(())
}
