// Ern-OS — ReAct Observer integration
//! Observer audit for ReAct loop replies. On rejection, feeds back
//! full context + rejection reason. Bailout after 2 consecutive rejections.

use crate::observer;
use crate::provider::Provider;
use anyhow::Result;

/// Observer verdict on a ReAct reply.
pub enum ObserverVerdict {
    /// Reply passes audit — deliver to user.
    Approved,
    /// Reply rejected — retry with feedback.
    Rejected {
        reason: String,
        guidance: String,
    },
}

/// Audit a ReAct reply through the observer.
/// Returns Approved or Rejected with feedback for re-prompting.
pub async fn audit_reply(
    provider: &dyn Provider,
    conversation: &[crate::provider::Message],
    reply: &str,
    _observer_enabled: bool,
    user_message: &str,
) -> Result<ObserverVerdict> {
    if !_observer_enabled {
        return Ok(ObserverVerdict::Approved);
    }

    let output = observer::audit_response(provider, conversation, reply, "", user_message).await?;

    if output.result.verdict.is_allowed() {
        Ok(ObserverVerdict::Approved)
    } else {
        Ok(ObserverVerdict::Rejected {
            reason: output.result.what_went_wrong.clone(),
            guidance: output.result.how_to_fix.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verdict_types() {
        let approved = ObserverVerdict::Approved;
        assert!(matches!(approved, ObserverVerdict::Approved));

        let rejected = ObserverVerdict::Rejected {
            reason: "Too short".into(),
            guidance: "Add more detail".into(),
        };
        assert!(matches!(rejected, ObserverVerdict::Rejected { .. }));
    }
}
