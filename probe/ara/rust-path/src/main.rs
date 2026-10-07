use ara2_bridge::core::ApiGeneration;
use ara2_bridge::plugin::PluginBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = PluginBuilder::new("document model").build()?;
    let interface = plugin.document_controller_interface(ApiGeneration::V23Final)?;

    assert!(interface.represented_callbacks_are_non_null());
    println!("model: {}", plugin.model());
    Ok(())
}
