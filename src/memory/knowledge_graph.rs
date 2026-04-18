// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tier 4: Knowledge Graph — Neo4j-backed entity-relation store.
//!
//! Stores entities (concepts, people, places) and weighted relations between them.
//! Uses decay-weighted retrieval: edges lose strength over time, frequently
//! reinforced connections persist.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use neo4rs::{query, Graph};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// An entity in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub label: String,
    pub entity_type: String,
    pub properties: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
}

/// A relation between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgRelation {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String,
    pub weight: f64,
    pub created_at: DateTime<Utc>,
    pub last_reinforced: DateTime<Utc>,
    pub reinforcement_count: u64,
}

/// Knowledge graph backed by Neo4j.
pub struct KnowledgeGraph {
    graph: Arc<Graph>,
    decay_rate: f64,
}

impl KnowledgeGraph {
    /// Connect to Neo4j and initialize the schema.
    ///
    /// Attempts to connect to the specified database. If that database doesn't exist
    /// (Neo4j Community only supports one database), falls back to `neo4j` database
    /// and uses namespace-prefixed labels to isolate ErnOS data.
    pub async fn connect(uri: &str, user: &str, password: &str, database: &str) -> Result<Self> {
        tracing::info!(uri = %uri, database = %database, "Connecting to Neo4j");

        // Try the specified database first
        let graph = match Self::try_connect(uri, user, password, database).await {
            Ok(g) => {
                tracing::info!(database = %database, "Connected to dedicated database");
                g
            }
            Err(e) => {
                let err_str = format!("{:?}", e);
                tracing::warn!(
                    error = %err_str,
                    database = %database,
                    "try_connect failed, checking if fallback is possible"
                );
                if database != "neo4j"
                    && (err_str.contains("DatabaseNotFound")
                        || err_str.contains("not accessible")
                        || err_str.contains("not found"))
                {
                    tracing::info!("Falling back to 'neo4j' database");
                    Self::try_connect(uri, user, password, "neo4j")
                        .await
                        .context("Failed to connect to fallback 'neo4j' database")?
                } else {
                    return Err(e);
                }
            }
        };

        let kg = Self {
            graph: Arc::new(graph),
            decay_rate: 0.99,
        };

        kg.ensure_schema().await?;

        tracing::info!(database = %database, "Knowledge graph connected and schema verified");
        Ok(kg)
    }

    async fn try_connect(uri: &str, user: &str, password: &str, database: &str) -> Result<Graph> {
        let config = neo4rs::ConfigBuilder::default()
            .uri(uri)
            .user(user)
            .password(password)
            .db(database)
            .build()
            .with_context(|| format!("Failed to build Neo4j config for {}", uri))?;

        let graph = Graph::connect(config)
            .await
            .with_context(|| format!("Failed to connect to Neo4j at {} (db: {})", uri, database))?;

        // Verify the database is accessible by running a trivial query
        graph
            .run(query("RETURN 1"))
            .await
            .with_context(|| format!("Database '{}' is not accessible", database))?;

        Ok(graph)
    }

    /// Create constraints and indexes for performance.
    async fn ensure_schema(&self) -> Result<()> {
        let queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
        ];

        for q in &queries {
            if let Err(e) = self.graph.run(query(q)).await {
                tracing::warn!(query = %q, error = %e, "Schema query failed (may already exist)");
            }
        }

        // Detect and purge legacy data from other projects
        self.ensure_clean().await?;

        Ok(())
    }

    /// Detect and remove legacy entities that don't belong to ErnOSAgent.
    ///
    /// Checks for the `_source: 'ernosagent'` marker on existing entities.
    /// If entities exist without this marker, they're from another project
    /// (e.g. HIVE/HIVENET) and should be purged.
    async fn ensure_clean(&self) -> Result<()> {
        let total = self.entity_count().await.unwrap_or(0);
        if total == 0 {
            return Ok(());
        }

        // Count entities that have the ErnOS source marker
        let q = query("MATCH (e:Entity) WHERE e._source = 'ernosagent' RETURN count(e) AS count");
        let mut result = self.graph.execute(q).await?;
        let owned_count: u64 = if let Some(row) = result.next().await? {
            row.get::<i64>("count").unwrap_or(0) as u64
        } else {
            0
        };

        if owned_count == 0 && total > 0 {
            tracing::warn!(
                total = total,
                "Detected {} legacy entities without ErnOS marker — purging stale data",
                total
            );
            // Delete all entities and relations — they belong to another project
            let _ = self.graph.run(query("MATCH (n) DETACH DELETE n")).await;
            tracing::info!("Legacy knowledge graph data purged");
        }

        Ok(())
    }

    /// Add or update an entity.
    pub async fn upsert_entity(
        &self,
        label: &str,
        entity_type: &str,
        properties: &serde_json::Value,
    ) -> Result<String> {
        let id = format!("{}:{}", entity_type, label.to_lowercase().replace(' ', "_"));
        let now = Utc::now().to_rfc3339();
        let props_str = serde_json::to_string(properties).unwrap_or_default();

        let q = query(
            "MERGE (e:Entity {id: $id})
             ON CREATE SET e.label = $label,
                           e.entity_type = $type,
                           e.properties = $props,
                           e.created_at = $now,
                           e.last_accessed = $now,
                           e.access_count = 1,
                           e._source = 'ernosagent'
             ON MATCH SET e.last_accessed = $now,
                          e.access_count = e.access_count + 1,
                          e.properties = $props,
                          e._source = 'ernosagent'
             RETURN e.id AS id",
        )
        .param("id", id.clone())
        .param("label", label.to_string())
        .param("type", entity_type.to_string())
        .param("props", props_str)
        .param("now", now);

        self.graph
            .run(q)
            .await
            .with_context(|| format!("Failed to upsert entity: {}", label))?;

        tracing::debug!(id = %id, label = %label, entity_type = %entity_type, "Entity upserted");
        Ok(id)
    }

    /// Create or reinforce a relation between two entities.
    pub async fn upsert_relation(
        &self,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        initial_weight: f64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let rel_id = format!("{}->{}:{}", source_id, target_id, relation_type);

        let q = query(
            "MATCH (s:Entity {id: $source_id}), (t:Entity {id: $target_id})
             MERGE (s)-[r:RELATES_TO {rel_id: $rel_id}]->(t)
             ON CREATE SET r.relation_type = $rel_type,
                           r.weight = $weight,
                           r.created_at = $now,
                           r.last_reinforced = $now,
                           r.reinforcement_count = 1
             ON MATCH SET r.weight = CASE
                              WHEN r.weight + $weight * 0.1 > 1.0 THEN 1.0
                              ELSE r.weight + $weight * 0.1
                          END,
                          r.last_reinforced = $now,
                          r.reinforcement_count = r.reinforcement_count + 1",
        )
        .param("source_id", source_id.to_string())
        .param("target_id", target_id.to_string())
        .param("rel_id", rel_id)
        .param("rel_type", relation_type.to_string())
        .param("weight", initial_weight)
        .param("now", now);

        self.graph.run(q).await.with_context(|| {
            format!(
                "Failed to upsert relation: {} -[{}]-> {}",
                source_id, relation_type, target_id
            )
        })?;

        tracing::debug!(
            source = %source_id,
            target = %target_id,
            relation = %relation_type,
            "Relation upserted"
        );

        Ok(())
    }

    /// Retrieve entities related to a query entity, ranked by decayed weight.
    pub async fn recall(
        &self,
        entity_id: &str,
        _max_depth: usize,
        limit: usize,
    ) -> Result<Vec<RecallResult>> {
        let q = query(
            "MATCH (s:Entity {id: $id})-[r:RELATES_TO*1..3]-(t:Entity)
             WITH t, r,
                  REDUCE(w = 1.0, rel IN r |
                      w * rel.weight *
                      (0.99 ^ (duration.between(datetime(rel.last_reinforced), datetime()).days))
                  ) AS decayed_weight
             RETURN t.id AS id,
                    t.label AS label,
                    t.entity_type AS entity_type,
                    t.properties AS properties,
                    decayed_weight
             ORDER BY decayed_weight DESC
             LIMIT $limit",
        )
        .param("id", entity_id.to_string())
        .param("limit", limit as i64);

        let mut result =
            self.graph.execute(q).await.with_context(|| {
                format!("Knowledge graph recall failed for entity: {}", entity_id)
            })?;

        let mut recalls = Vec::new();
        while let Some(row) = result.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            let label: String = row.get("label").unwrap_or_default();
            let entity_type: String = row.get("entity_type").unwrap_or_default();
            let properties: String = row.get("properties").unwrap_or_default();
            let weight: f64 = row.get("decayed_weight").unwrap_or(0.0);

            recalls.push(RecallResult {
                id,
                label,
                entity_type,
                properties: serde_json::from_str(&properties).unwrap_or_default(),
                weight,
            });
        }

        tracing::debug!(
            entity = %entity_id,
            results = recalls.len(),
            "Knowledge graph recall complete"
        );

        Ok(recalls)
    }

    /// Search entities by label (fuzzy text match).
    pub async fn search_entities(&self, query_text: &str, limit: usize) -> Result<Vec<Entity>> {
        let pattern = format!("(?i).*{}.*", regex_escape(query_text));

        let q = query(
            "MATCH (e:Entity)
             WHERE e.label =~ $pattern
             RETURN e.id AS id,
                    e.label AS label,
                    e.entity_type AS entity_type,
                    e.properties AS properties,
                    e.created_at AS created_at,
                    e.last_accessed AS last_accessed,
                    e.access_count AS access_count
             ORDER BY e.access_count DESC
             LIMIT $limit",
        )
        .param("pattern", pattern)
        .param("limit", limit as i64);

        let mut result = self
            .graph
            .execute(q)
            .await
            .with_context(|| format!("Entity search failed for: {}", query_text))?;

        let mut entities = Vec::new();
        while let Some(row) = result.next().await? {
            entities.push(Entity {
                id: row.get("id").unwrap_or_default(),
                label: row.get("label").unwrap_or_default(),
                entity_type: row.get("entity_type").unwrap_or_default(),
                properties: serde_json::from_str(
                    &row.get::<String>("properties").unwrap_or_default(),
                )
                .unwrap_or_default(),
                created_at: parse_datetime(&row.get::<String>("created_at").unwrap_or_default()),
                last_accessed: parse_datetime(
                    &row.get::<String>("last_accessed").unwrap_or_default(),
                ),
                access_count: row.get("access_count").unwrap_or(0),
            });
        }

        Ok(entities)
    }

    /// Count all entities in the graph.
    pub async fn entity_count(&self) -> Result<u64> {
        let q = query("MATCH (e:Entity) RETURN count(e) AS count");
        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("count").unwrap_or(0);
            Ok(count as u64)
        } else {
            Ok(0)
        }
    }

    /// Count all relations in the graph.
    pub async fn relation_count(&self) -> Result<u64> {
        let q = query("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count");
        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("count").unwrap_or(0);
            Ok(count as u64)
        } else {
            Ok(0)
        }
    }

    /// Apply time-based decay to all relations (run periodically).
    pub async fn apply_decay(&self) -> Result<u64> {
        let q = query(
            "MATCH ()-[r:RELATES_TO]->()
             SET r.weight = r.weight * $decay_rate
             WITH r WHERE r.weight < 0.01
             DELETE r
             RETURN count(r) AS pruned",
        )
        .param("decay_rate", self.decay_rate);

        let mut result = self.graph.execute(q).await?;
        let pruned = if let Some(row) = result.next().await? {
            row.get::<i64>("pruned").unwrap_or(0) as u64
        } else {
            0
        };

        if pruned > 0 {
            tracing::info!(
                pruned = pruned,
                "Knowledge graph decay pruned weak relations"
            );
        }

        Ok(pruned)
    }

    /// Format the knowledge graph state for the HUD prompt.
    pub async fn status_summary(&self) -> String {
        let entities = self.entity_count().await.unwrap_or(0);
        let relations = self.relation_count().await.unwrap_or(0);
        format!("KG: {} entities, {} relations", entities, relations)
    }
}

/// Result from a knowledge graph recall query.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub id: String,
    pub label: String,
    pub entity_type: String,
    pub properties: serde_json::Value,
    pub weight: f64,
}

impl RecallResult {
    /// Format this recall result for context injection.
    pub fn format_for_context(&self) -> String {
        format!(
            "[KG:{} ({})] {} (weight: {:.3})",
            self.entity_type,
            self.label,
            self.properties
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            self.weight
        )
    }
}

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn regex_escape(s: &str) -> String {
    let special = [
        '\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '^', '$', '|',
    ];
    let mut escaped = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if special.contains(&c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

#[cfg(test)]
#[path = "kg_tests.rs"]
mod tests;
