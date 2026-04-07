//! Tier 5: Scratchpad — Neo4j-backed pinned notes.

pub struct ScratchpadStore {
    entries: Vec<ScratchpadEntry>,
}

#[derive(Debug, Clone)]
pub struct ScratchpadEntry {
    pub key: String,
    pub value: String,
    pub pinned: bool,
}

impl ScratchpadStore {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn pin(&mut self, key: &str, value: &str) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.value = value.to_string();
            entry.pinned = true;
        } else {
            self.entries.push(ScratchpadEntry {
                key: key.to_string(), value: value.to_string(), pinned: true,
            });
        }
    }

    pub fn unpin(&mut self, key: &str) {
        self.entries.retain(|e| e.key != key);
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.value.as_str())
    }

    pub fn all(&self) -> &[ScratchpadEntry] { &self.entries }
    pub fn count(&self) -> usize { self.entries.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_and_get() {
        let mut store = ScratchpadStore::new();
        store.pin("lang", "Rust");
        assert_eq!(store.get("lang"), Some("Rust"));
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_unpin() {
        let mut store = ScratchpadStore::new();
        store.pin("key", "val");
        store.unpin("key");
        assert!(store.get("key").is_none());
    }
}
