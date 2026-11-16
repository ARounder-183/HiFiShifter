use ara2_bridge::core::{MemoryArchive, ReadAt, WriteAt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = MemoryArchive::with_id(Vec::new(), "document-state-v1")?;
    archive.write_at(4, b"ARA")?;

    let mut restored = [0_u8; 3];
    archive.read_at(4, &mut restored)?;
    assert_eq!(&restored, b"ARA");
    println!("{} bytes in {:?}", archive.len()?, archive.archive_id());
    Ok(())
}
