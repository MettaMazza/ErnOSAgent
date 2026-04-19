//! Static content handlers — HTML, CSS, JS, and dynamically generated images.

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};

pub async fn index() -> impl IntoResponse {
    Html(include_str!("../static/index.html"))
}

pub async fn css() -> impl IntoResponse {
    ([
        ("content-type", "text/css; charset=utf-8"),
        ("cache-control", "no-cache, no-store, must-revalidate"),
    ], include_str!("../static/app.css"))
}

pub async fn js() -> impl IntoResponse {
    ([
        ("content-type", "application/javascript; charset=utf-8"),
        ("cache-control", "no-cache, no-store, must-revalidate"),
    ], include_str!("../static/app.js"))
}

/// Serve generated images from `data/images/{filename}`.
pub async fn serve_image(Path(filename): Path<String>) -> impl IntoResponse {
    // Sanitize: reject path traversal
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return (StatusCode::BAD_REQUEST, "Invalid filename".to_string()).into_response();
    }

    let path = std::path::PathBuf::from("data/images").join(&filename);
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let content_type = if filename.ends_with(".png") {
                "image/png"
            } else if filename.ends_with(".jpg") || filename.ends_with(".jpeg") {
                "image/jpeg"
            } else if filename.ends_with(".webp") {
                "image/webp"
            } else {
                "application/octet-stream"
            };
            ([("content-type", content_type), ("cache-control", "public, max-age=86400")], bytes).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "Image not found".to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_index_html_included() {
        let html = include_str!("../static/index.html");
        assert!(html.contains("Ern-OS"));
    }

    #[test]
    fn test_css_included() {
        let css = include_str!("../static/app.css");
        assert!(css.contains("--bg-primary"));
    }

    #[test]
    fn test_js_included() {
        let js = include_str!("../static/app.js");
        assert!(js.contains("ErnOS"));
    }
}
