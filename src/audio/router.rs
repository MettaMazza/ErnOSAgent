//! Audio router — decides whether to use native audio model or Whisper STT.

/// Route audio based on available models.
pub struct AudioRouter;

impl AudioRouter {
    pub fn new() -> Self { Self }
}
