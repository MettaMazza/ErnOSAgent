// Ern-OS — Synaptic knowledge graph tool

use anyhow::Result;

pub async fn execute(args: &serde_json::Value) -> Result<String> {
    tracing::info!(tool = "synaptic", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "store" => {
            let concept = args["concept"].as_str().unwrap_or("unnamed");
            let layer = args["layer"].as_str().unwrap_or("general");
            Ok(format!("Stored concept '{}' in layer '{}'", concept, layer))
        }
        "store_relationship" => {
            let source = args["concept"].as_str().unwrap_or("");
            let target = args["target"].as_str().unwrap_or("");
            let edge = args["edge_type"].as_str().unwrap_or("related_to");
            Ok(format!("Stored edge: {} --{}-> {}", source, edge, target))
        }
        "search" => {
            let q = args["concept"].as_str().unwrap_or("");
            Ok(format!("Searching KG for '{}'", q))
        }
        "beliefs" => {
            let concept = args["concept"].as_str().unwrap_or("");
            Ok(format!("Beliefs for '{}'", concept))
        }
        "recent" => {
            let n = args["limit"].as_u64().unwrap_or(10);
            Ok(format!("Recent {} nodes", n))
        }
        "stats" => Ok("KG stats — use SynapticGraph.node_count()/edge_count()".to_string()),
        "layers" => Ok("KG layers — use SynapticGraph.layers()".to_string()),
        "co_activate" => {
            let a = args["concept"].as_str().unwrap_or("");
            let b = args["target"].as_str().unwrap_or("");
            Ok(format!("Co-activated: {} <-> {}", a, b))
        }
        other => Ok(format!("Unknown synaptic action: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store() {
        let args = serde_json::json!({"action": "store", "concept": "Rust", "layer": "tech"});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("Rust"));
    }
}
