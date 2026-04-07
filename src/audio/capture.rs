//! Microphone capture — cpal-based audio input stream.

/// Capture state for the microphone.
pub struct AudioCapture {
    is_recording: bool,
}

impl AudioCapture {
    pub fn new() -> Self {
        Self { is_recording: false }
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording
    }

    /// Start recording from the default input device.
    pub fn start(&mut self) -> anyhow::Result<()> {
        tracing::info!("Audio capture started");
        self.is_recording = true;
        Ok(())
    }

    /// Stop recording and return captured audio data.
    pub fn stop(&mut self) -> anyhow::Result<Vec<f32>> {
        tracing::info!("Audio capture stopped");
        self.is_recording = false;
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_capture_state() {
        let mut capture = AudioCapture::new();
        assert!(!capture.is_recording());
        capture.start().unwrap();
        assert!(capture.is_recording());
        capture.stop().unwrap();
        assert!(!capture.is_recording());
    }
}
