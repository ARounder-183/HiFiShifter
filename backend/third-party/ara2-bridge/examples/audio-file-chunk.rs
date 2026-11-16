use ara2_bridge::core::{replace_ara_in_path, AraChunkSet};

const LEGACY_CHUNK: &[u8] = br#"<?xml version="1.0" encoding="UTF-8"?>
<BWFXML><IXML_VERSION>2.0</IXML_VERSION><ARA><audioSources><audioSource><documentArchiveID>com.example.archive</documentArchiveID><persistentID>source-legacy</persistentID><archiveData></archiveData></audioSource></audioSources></ARA></BWFXML>"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let chunk = AraChunkSet::parse(LEGACY_CHUNK)?;
    let entry = chunk.get("com.example.archive").expect("embedded entry");
    println!("archive bytes: {}", entry.archive_data().len());

    if let Some(path) = std::env::args_os().nth(1) {
        replace_ara_in_path(path, &chunk)?;
    } else {
        println!("pass a WAVE/AIFF path to atomically write this chunk");
    }
    Ok(())
}
