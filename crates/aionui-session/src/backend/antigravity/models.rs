//! Model discovery for agy (`agy models`).
//!
//! agy prints one model id per line. Those ids ALREADY encode reasoning effort
//! (`-high` / `-medium` / `-low`), and agy only accepts a complete id: a
//! stripped one is silently dropped and it falls back to another model without
//! reporting anything. So the ids are surfaced verbatim and no separate effort
//! axis is advertised.

use std::sync::Arc;

use aionui_common::CommandSpec;
use aionui_process::Spawner;

use crate::capability::ModelInfo;

/// A model id is a single lowercase-ish token: no spaces, no punctuation beyond
/// `-`/`.`/`_`. Used to tell ids apart from the human-readable errors agy
/// prints to stdout (e.g. the signed-out notice).
fn looks_like_model_id(line: &str) -> bool {
    !line.is_empty()
        && line
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
}

/// Split a catalog line into `(id, label)`.
///
/// agy 1.1.12 prints `id<TAB>Human Label`; earlier releases printed a bare id
/// (the 2026-07-31 capture in the tests below). Both are accepted: the id is
/// whatever precedes the first tab, and a line with no tab is all id — so the
/// parser does not break again the next time the label column moves.
///
/// The id still has to look like an id. That is what keeps agy's prose — the
/// signed-out notice above all — from being offered as a model, and it is the
/// check the whole-line version was accidentally applying to the label too,
/// which is why every 1.1.12 line was dropped and the picker opened empty.
fn split_catalog_line(line: &str) -> Option<(&str, &str)> {
    let (id, label) = match line.split_once('\t') {
        Some((id, label)) => (id.trim(), label.trim()),
        None => (line, line),
    };
    looks_like_model_id(id).then_some((id, if label.is_empty() { id } else { label }))
}

pub(crate) fn parse_agy_models(stdout: &str) -> Vec<ModelInfo> {
    stdout
        .lines()
        .map(str::trim)
        .filter_map(split_catalog_line)
        .map(|(id, label)| ModelInfo {
            id: id.to_owned(),
            name: label.to_owned(),
            description: None,
            // Deliberately empty — see the module docs.
            reasoning_efforts: Vec::new(),
        })
        .collect()
}

/// Ask agy which models this account can use.
///
/// Best-effort by contract: any failure (agy missing, signed out, slow) yields
/// an empty list rather than an error, because a model picker that cannot be
/// populated must not stop the user from opening a session.
pub(crate) async fn probe_models(
    spawner: &Arc<dyn Spawner>,
    program: &std::path::Path,
    owner_tag: &str,
) -> Vec<ModelInfo> {
    let spec = CommandSpec {
        command: program.to_path_buf(),
        args: vec!["models".to_owned()],
        env: Vec::new(),
        cwd: None,
    };
    let Ok(proc) = spawner.spawn(spec, &[], owner_tag).await else {
        return Vec::new();
    };
    let Some((stdin, stdout)) = proc.take_stdio().await else {
        return Vec::new();
    };

    // `agy models` does NOT exit while its stdin is open — it prints the list
    // and then keeps waiting, so stdout never reaches EOF and the read below
    // would hang forever. Verified against agy 1.1.9: stdin left open = still
    // running after 40 minutes; stdin closed = exits in ~3s. Dropping the
    // handle closes the pipe and delivers the EOF that lets agy finish.
    drop(stdin);

    let text = match tokio::time::timeout(PROBE_TIMEOUT, read_to_end(stdout)).await {
        Ok(text) => text,
        Err(_) => {
            // Never leave the child behind: this task is detached, so a hung
            // agy would leak a process for the lifetime of the app.
            tracing::warn!("antigravity: `agy models` did not finish in time; model list left empty");
            let _ = proc.kill(std::time::Duration::from_secs(1)).await;
            return Vec::new();
        }
    };
    parse_agy_models(&text)
}

/// How long to wait for `agy models` before giving up on the model list.
/// Generous: a cold agy takes ~3s, but a slow network sign-in check is slower.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

async fn read_to_end(stdout: aionui_process::BoxedStdout) -> String {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut lines = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = lines.next_line().await {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// agy 1.1.12 prints `id<TAB>label`, not a bare id — captured live
    /// 2026-08-12 from `agy models` (1.1.12), whose changelog entry only
    /// mentioned moving the spinner off stdout.
    ///
    /// Every one of these lines fails `looks_like_model_id` (tab and spaces are
    /// not in its allowed set), so the whole catalog is dropped and the model
    /// picker opens empty.
    #[test]
    fn parses_the_tab_separated_catalog_agy_prints_today() {
        let out = "gemini-3.6-flash-high\tGemini 3.6 Flash (High)\n\
                   gemini-3.5-flash-low\tGemini 3.5 Flash (Low)\n\
                   claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)\n\
                   gpt-oss-120b-medium\tGPT-OSS 120B (Medium)\n";
        let models = parse_agy_models(out);
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec![
                "gemini-3.6-flash-high",
                "gemini-3.5-flash-low",
                "claude-sonnet-4-6",
                "gpt-oss-120b-medium",
            ],
            "the id is the first field, not the whole line"
        );
        assert_eq!(
            models.first().map(|m| m.name.as_str()),
            Some("Gemini 3.6 Flash (High)"),
            "the label agy now supplies is what the picker should show"
        );
    }

    #[test]
    fn parses_one_id_per_line() {
        // Real `agy models` output (2026-07-31).
        let out = "gemini-3.6-flash-high\ngemini-3.6-flash-low\ngemini-3.1-pro-high\nclaude-sonnet-4-6\ngpt-oss-120b-medium\n";
        let models = parse_agy_models(out);
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec![
                "gemini-3.6-flash-high",
                "gemini-3.6-flash-low",
                "gemini-3.1-pro-high",
                "claude-sonnet-4-6",
                "gpt-oss-120b-medium",
            ]
        );
    }

    #[test]
    fn model_ids_keep_their_effort_suffix_and_expose_no_effort_axis() {
        // agy's ids already encode effort. Exposing a separate effort picker
        // would let the UI build a stripped id, which agy silently ignores
        // while falling back to another model — with no error anywhere.
        let models = parse_agy_models("gemini-3.6-flash-high\n");
        assert_eq!(models[0].id, "gemini-3.6-flash-high");
        assert!(models.iter().all(|m| m.reasoning_efforts.is_empty()));
    }

    #[test]
    fn blank_lines_and_padding_are_ignored() {
        let models = parse_agy_models("\n  gemini-3.6-flash-low  \n\n\tclaude-sonnet-4-6\n   \n");
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["gemini-3.6-flash-low", "claude-sonnet-4-6"]
        );
    }

    #[test]
    fn a_sign_in_error_yields_no_models_rather_than_a_bogus_one() {
        // Logged out, `agy models` prints this on stdout and exits 1. Treating
        // it as a model id would put a sentence in the model picker.
        let models = parse_agy_models(
            "Error: Please sign in to view available models. Launch the CLI without arguments to sign in.\n",
        );
        assert!(models.is_empty(), "got {models:?}");
    }

    #[test]
    fn empty_output_is_not_an_error() {
        // Probing must never block session creation.
        assert!(parse_agy_models("").is_empty());
    }

    /// Build a stand-in for agy that mirrors the ONE behaviour this test is
    /// about: print the model list, then stay alive until stdin reaches EOF.
    #[cfg(unix)]
    fn fake_agy_that_waits_on_stdin(dir: &std::path::Path) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("fake-agy");
        std::fs::write(&script, "#!/bin/sh\necho gemini-3.6-flash-high\ncat > /dev/null\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn probing_closes_stdin_so_agy_can_exit() {
        // Regression: `agy models` prints its list and then waits on stdin.
        // Holding the stdin handle open meant stdout never reached EOF, so the
        // probe hung forever and the model picker stayed silently empty — the
        // failure showed up only as `available_models: None`, with no error.
        let tmp = tempfile::tempdir().unwrap();
        let program = fake_agy_that_waits_on_stdin(tmp.path());
        let spawner: Arc<dyn Spawner> = Arc::new(aionui_process::RealSpawner::new(
            Arc::new(aionui_process::FileRegistryStore::new(tmp.path())),
            uuid::Uuid::now_v7(),
            "test-machine",
        ));

        let models = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            probe_models(&spawner, &program, "probe-test"),
        )
        .await
        .expect("probe hung — stdin was left open, so stdout never reached EOF");

        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["gemini-3.6-flash-high"]
        );
    }
}
