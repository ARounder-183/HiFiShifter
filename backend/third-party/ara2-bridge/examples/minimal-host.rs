use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::host::{
    ArchiveReaderId, ArchiveWriterId, ArchivingProvider, AudioAccessProvider, AudioSourceId,
    HostAudioReader, HostServicesBuilder,
};

struct Audio;

impl AudioAccessProvider for Audio {
    fn create_reader(
        &self,
        _: AudioSourceId,
        _: bool,
    ) -> Result<Box<dyn HostAudioReader>, AraError> {
        Err(AraError::Peer("no audio source is loaded"))
    }
}

struct Archives;

impl ArchivingProvider for Archives {
    fn len(&self, _: ArchiveReaderId) -> Result<usize, AraError> {
        Ok(0)
    }

    fn read_at(&self, _: ArchiveReaderId, _: usize, _: &mut [u8]) -> Result<(), AraError> {
        Ok(())
    }

    fn write_at(&self, _: ArchiveWriterId, _: usize, _: &[u8]) -> Result<(), AraError> {
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let services = HostServicesBuilder::new()
        .audio(Audio)
        .archiving(Archives)
        .build(ApiGeneration::V23Final)?;
    assert!(!services.instance().audioAccessControllerHostRef.is_null());
    println!("ARA 2.3 host services ready");
    Ok(())
}
