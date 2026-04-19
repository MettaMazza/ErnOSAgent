// Ern-OS — Synaptic KG: Relationship types

use serde::{Deserialize, Serialize};

/// Standard relationship types for the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationType {
    IsA,
    HasA,
    RelatedTo,
    CausedBy,
    Requires,
    Contradicts,
    Supports,
    Custom(String),
}

impl RelationType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::IsA => "is_a",
            Self::HasA => "has_a",
            Self::RelatedTo => "related_to",
            Self::CausedBy => "caused_by",
            Self::Requires => "requires",
            Self::Contradicts => "contradicts",
            Self::Supports => "supports",
            Self::Custom(s) => s.as_str(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "is_a" => Self::IsA,
            "has_a" => Self::HasA,
            "related_to" => Self::RelatedTo,
            "caused_by" => Self::CausedBy,
            "requires" => Self::Requires,
            "contradicts" => Self::Contradicts,
            "supports" => Self::Supports,
            other => Self::Custom(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let r = RelationType::IsA;
        assert_eq!(RelationType::from_str(r.as_str()), RelationType::IsA);
    }

    #[test]
    fn test_custom() {
        let r = RelationType::from_str("my_custom_rel");
        assert_eq!(r.as_str(), "my_custom_rel");
    }
}
