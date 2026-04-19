//! Dispatch handlers for planning and verification tools.

use crate::web::state::AppState;

/// Dispatch verify_code tool — runs the verification pipeline.
pub async fn dispatch_verify_code(args: &serde_json::Value) -> anyhow::Result<String> {
    let run_tests = args["run_tests"].as_bool().unwrap_or(true);
    let browser_url = args["browser_url"].as_str().map(String::from);

    let config = crate::verification::pipeline::VerificationConfig {
        run_tests,
        browser_url,
        ..Default::default()
    };

    let result = crate::verification::pipeline::run_verification(&config).await?;

    if result.overall_pass {
        Ok(format!(
            "[VERIFICATION PASSED]\nBuild: ✅\nTests: {}\nBrowser: {}",
            if config.run_tests { "✅" } else { "skipped" },
            if result.browser_result.is_some() { "✅" } else { "skipped" },
        ))
    } else {
        Ok(crate::verification::pipeline::format_fix_prompt(&result))
    }
}

/// Dispatch plan_and_execute tool — decomposes and executes a task DAG.
pub async fn dispatch_plan_and_execute(
    state: &AppState,
    args: &serde_json::Value,
) -> anyhow::Result<String> {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DAG_RUNNING: AtomicBool = AtomicBool::new(false);

    // Prevent recursive DAG execution
    if DAG_RUNNING.swap(true, Ordering::SeqCst) {
        anyhow::bail!("plan_and_execute cannot be called recursively. Use individual tools instead.");
    }

    let result = dispatch_plan_inner(state, args).await;
    DAG_RUNNING.store(false, Ordering::SeqCst);
    result
}

async fn dispatch_plan_inner(
    state: &AppState,
    args: &serde_json::Value,
) -> anyhow::Result<String> {
    let objective = args["objective"].as_str().unwrap_or("");
    let context = args["project_context"].as_str().unwrap_or("");

    if objective.is_empty() {
        anyhow::bail!("Missing 'objective' parameter");
    }

    let provider = state.provider.as_ref();
    let dag = crate::planning::planner::decompose_objective(
        provider, objective, context,
    ).await?;

    let task_count = dag.nodes.len();
    let mut dag = dag;
    let result = crate::planning::executor::execute_dag(
        provider, state, &mut dag,
    ).await?;

    Ok(format!(
        "[DAG Execution Complete]\nObjective: {}\n\
         Tasks: {} total, {} completed, {} failed, {} blocked\n\
         Success: {}\n\n{}",
        objective, task_count, result.completed, result.failed, result.blocked,
        result.overall_success, result.summary
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        assert!(true);
    }
}
