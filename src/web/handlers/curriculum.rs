//! Curriculum handlers — course CRUD, progress, and review deck statistics.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

/// GET /api/curriculum — List all courses with progress summary.
pub async fn list_courses(State(state): State<AppState>) -> impl IntoResponse {
    let store = state.curriculum.read().await;
    let courses: Vec<serde_json::Value> = store.courses().iter().map(|c| {
        let progress = store.get_progress(&c.id);
        let completed = progress.map(|p| p.completed_lessons.len()).unwrap_or(0);
        let avg_score = progress.map(|p| p.average_quiz_score()).unwrap_or(0.0);
        serde_json::json!({
            "id": c.id, "title": c.title, "description": c.description,
            "level": format!("{:?}", c.level), "subject": c.subject.as_str(),
            "total_lessons": c.lessons.len(), "completed_lessons": completed,
            "average_score": avg_score,
            "is_complete": store.is_course_complete(c),
        })
    }).collect();
    Json(serde_json::json!({ "count": courses.len(), "courses": courses }))
}

/// POST /api/curriculum — Add a course from JSON body.
pub async fn add_course(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let course: crate::learning::curriculum::Course = match serde_json::from_value(body) {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({ "error": format!("Invalid course JSON: {}", e) }));
        }
    };
    tracing::info!(
        course_id = %course.id, title = %course.title,
        lessons = course.lessons.len(), "Adding course via API"
    );
    let mut store = state.curriculum.write().await;
    match store.add_course(course) {
        Ok(()) => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

/// DELETE /api/curriculum/{id} — Remove a course by ID.
pub async fn remove_course(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    tracing::info!(course_id = %id, "Removing course via API");
    let mut store = state.curriculum.write().await;
    match store.remove_course(&id) {
        Ok(()) => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

/// GET /api/curriculum/{id}/progress — Detailed progress for one course.
pub async fn course_progress(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let store = state.curriculum.read().await;
    let course = match store.get_course(&id) {
        Some(c) => c,
        None => return Json(serde_json::json!({ "error": format!("Course '{}' not found", id) })),
    };
    let progress = store.get_progress(&id);
    let completed = progress.map(|p| p.completed_lessons.clone()).unwrap_or_default();
    let scores = progress.map(|p| p.quiz_scores.clone()).unwrap_or_default();
    let avg = progress.map(|p| p.average_quiz_score()).unwrap_or(0.0);
    let next_lesson = store.next_lesson(course).map(|l| l.title.clone());
    Json(serde_json::json!({
        "course_id": id, "title": course.title,
        "level": format!("{:?}", course.level),
        "total_lessons": course.lessons.len(),
        "completed_lessons": completed,
        "quiz_scores": scores,
        "average_score": avg,
        "is_complete": store.is_course_complete(course),
        "next_lesson": next_lesson,
    }))
}

/// GET /api/curriculum/review — Review deck statistics.
pub async fn review_stats(State(state): State<AppState>) -> impl IntoResponse {
    let deck = state.review_deck.read().await;
    let stats = deck.retention_stats();
    Json(serde_json::json!({
        "total_cards": stats.total_cards,
        "cards_due": stats.cards_due,
        "avg_box_level": stats.avg_box_level,
        "retention_rate": stats.retention_rate,
    }))
}

#[cfg(test)]
#[path = "curriculum_tests.rs"]
mod tests;
