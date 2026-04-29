// Ern-OS — Curriculum handler tests
// Sibling test file per §3.2 (>100 lines of tests must be extracted).

use super::*;
use crate::learning::curriculum::*;
use axum::extract::State;
use axum::response::IntoResponse;

/// Build a minimal test AppState for handler tests.
fn build_test_state(tmp: &std::path::Path) -> AppState {
    AppState {
        config: std::sync::Arc::new(crate::config::AppConfig::default()),
        model_spec: std::sync::Arc::new(crate::model::ModelSpec::default()),
        memory: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::memory::MemoryManager::new(tmp).unwrap(),
        )),
        sessions: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::session::SessionManager::new(&tmp.join("s")).unwrap(),
        )),
        provider: std::sync::Arc::new(crate::provider::llamacpp::LlamaCppProvider::new(
            &crate::config::LlamaCppConfig::default(),
        )),
        golden_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::learning::buffers::GoldenBuffer::new(500),
        )),
        rejection_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::learning::buffers_rejection::RejectionBuffer::new(),
        )),
        scheduler: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::scheduler::store::JobStore::load(tmp).unwrap(),
        )),
        agents: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::agents::AgentRegistry::new(tmp).unwrap(),
        )),
        teams: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::agents::teams::TeamRegistry::new(tmp).unwrap(),
        )),
        browser: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::tools::browser_tool::BrowserState::new(),
        )),
        platforms: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::platform::registry::PlatformRegistry::new(),
        )),
        mutable_config: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::config::AppConfig::default(),
        )),
        resume_message: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
        sae: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
        live_monitor: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::interpretability::live::LiveMonitor::new(50),
        )),
        snapshot_store: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::interpretability::snapshot::SnapshotStore::new(
                &tmp.join("snapshots"),
            ).unwrap(),
        )),
        cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        curriculum: std::sync::Arc::new(tokio::sync::RwLock::new(
            CurriculumStore::open(&tmp.join("curriculum")).unwrap(),
        )),
        quarantine: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::learning::verification::QuarantineBuffer::new(),
        )),
        review_deck: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::learning::review::ReviewDeck::new(),
        )),
    }
}

fn sample_course() -> Course {
    Course {
        id: "test-101".into(),
        title: "Test Course".into(),
        description: "A test course".into(),
        level: EducationLevel::Primary,
        subject: Subject::Mathematics,
        lessons: vec![
            Lesson {
                id: "lesson-1".into(),
                title: "Lesson 1".into(),
                order: 0,
                scenes: vec![],
                objectives: vec!["Basic arithmetic".into()],
                prerequisites: vec![],
            },
        ],
        prerequisites: vec![],
        completion_criteria: CompletionCriteria {
            min_lessons_completed: 1,
            min_quiz_score: 0.7,
            min_essay_score: 0.0,
            requires_original_work: false,
            requires_defense: false,
        },
        source: CurriculumSource::CustomJsonl { path: "test.jsonl".into() },
        created_at: chrono::Utc::now(),
    }
}

async fn extract_json(resp: impl IntoResponse) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_response().into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn test_list_courses_empty() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    let json = extract_json(list_courses(State(state)).await).await;
    assert_eq!(json["count"], 0);
    assert!(json["courses"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_list_courses_with_data() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    state.curriculum.write().await.add_course(sample_course()).unwrap();
    let json = extract_json(list_courses(State(state)).await).await;
    assert_eq!(json["count"], 1);
    assert_eq!(json["courses"][0]["id"], "test-101");
}

#[tokio::test]
async fn test_add_course_valid() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    let body = serde_json::to_value(sample_course()).unwrap();
    let json = extract_json(add_course(State(state.clone()), Json(body)).await).await;
    assert_eq!(json["ok"], true);
    assert_eq!(state.curriculum.read().await.course_count(), 1);
}

#[tokio::test]
async fn test_add_course_duplicate_rejected() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    state.curriculum.write().await.add_course(sample_course()).unwrap();
    let body = serde_json::to_value(sample_course()).unwrap();
    let json = extract_json(add_course(State(state), Json(body)).await).await;
    assert!(json["error"].as_str().is_some());
}

#[tokio::test]
async fn test_add_course_invalid_body() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    let body = serde_json::json!({"not": "a course"});
    let json = extract_json(add_course(State(state), Json(body)).await).await;
    assert!(json["error"].as_str().unwrap().contains("Invalid course JSON"));
}

#[tokio::test]
async fn test_remove_course_exists() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    state.curriculum.write().await.add_course(sample_course()).unwrap();
    let json = extract_json(remove_course(
        State(state.clone()),
        axum::extract::Path("test-101".into()),
    ).await).await;
    assert_eq!(json["ok"], true);
    assert_eq!(state.curriculum.read().await.course_count(), 0);
}

#[tokio::test]
async fn test_remove_course_not_found() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    let json = extract_json(remove_course(
        State(state),
        axum::extract::Path("nonexistent".into()),
    ).await).await;
    assert!(json["error"].as_str().is_some());
}

#[tokio::test]
async fn test_review_stats_empty_deck() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = build_test_state(tmp.path());
    let json = extract_json(review_stats(State(state)).await).await;
    assert_eq!(json["total_cards"], 0);
    assert_eq!(json["cards_due"], 0);
}
