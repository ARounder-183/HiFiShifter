use ara2_bridge::core::{validate_event_sequence, NoteEvent, Notes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let notes = vec![
        NoteEvent::new(Some(261.625_55), Some(60), 0.8, 0.0, 0.02, 0.5, 0.6)?,
        NoteEvent::new(Some(329.627_56), Some(64), 0.7, 0.5, 0.01, 0.5, 0.6)?,
    ];
    validate_event_sequence::<Notes>(&notes)?;
    for note in notes {
        println!(
            "pitch={} start={}",
            note.pitch_number(),
            note.start_position()
        );
    }
    Ok(())
}
