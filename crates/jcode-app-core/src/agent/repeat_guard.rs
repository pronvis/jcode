//! Detects an agent re-fetching the same read-only observation over and over.
//!
//! Long analysis tasks fail in a characteristic way: a conclusion is reached in
//! reasoning but never written down, so every later pass over the same file
//! re-derives it from scratch. Traces show the same verdict reached a dozen
//! times, each pass adding nothing. The harness cannot see reasoning, but it
//! can see the tool calls that feed it, and an identical read/search repeated
//! many times is the visible shadow of that loop.
//!
//! On the Nth identical fetch this appends a one-line notice to the tool
//! output telling the agent to consult the conclusion it already reached. The
//! notice rides inside the tool result rather than a separate message so it
//! lands next to the evidence that triggered it.

use std::collections::HashMap;

use serde_json::Value;

/// A repeat must reach this count before the first notice, and every further
/// multiple of it re-notices. Escalating on multiples keeps a stuck agent
/// nudged without one notice per call.
const NOTICE_EVERY: u32 = 3;

/// Read-only observations worth fingerprinting. Deliberately a small allowlist:
/// a tool whose repeat is legitimate (`todo`, `bg`, `swarm`) must never be
/// nudged, and a tool whose effects are invisible here must not be guessed at.
const OBSERVATION_TOOLS: &[&str] = &[
    "read",
    "agentgrep",
    "ls",
    "webfetch",
    "jcode_docs",
    "session_search",
    "conversation_search",
];

/// Tools that change files, invalidating any conclusion drawn from a prior read.
const MUTATION_TOOLS: &[&str] = &["write", "edit", "multiedit", "patch", "apply_patch"];

/// Argument keys that carry model prose rather than identity. `intent` is
/// required on every jcode tool and is rewritten on each call, so leaving it in
/// the fingerprint would make every call unique and disable the guard entirely.
const NON_IDENTIFYING_KEYS: &[&str] = &["intent", "description", "reason"];

/// Keys that name the file a call reads or writes, used to invalidate reads of
/// a path once something edits it.
const PATH_KEYS: &[&str] = &["file_path", "path", "notebook_path"];

#[derive(Debug, Default)]
pub(super) struct RepeatGuard {
    /// Fingerprint -> times observed since the last invalidation.
    counts: HashMap<String, u32>,
    /// Fingerprint -> the path it read, for mutation-driven invalidation.
    paths: HashMap<String, String>,
}

impl RepeatGuard {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn reset(&mut self) {
        self.counts.clear();
        self.paths.clear();
    }

    /// Record a completed tool call and return a notice to append to its output.
    ///
    /// Mutating calls invalidate rather than count: after an edit, re-reading
    /// the file is the correct move, not a loop.
    pub(super) fn observe(&mut self, tool_name: &str, input: &Value) -> Option<String> {
        if MUTATION_TOOLS.contains(&tool_name) {
            self.invalidate_path(extract_path(input).as_deref());
            return None;
        }
        if tool_name == "bash" {
            self.invalidate_mentioned_paths(input);
            return None;
        }
        if !OBSERVATION_TOOLS.contains(&tool_name) {
            return None;
        }

        let fingerprint = fingerprint(tool_name, input);
        let count = self.counts.entry(fingerprint.clone()).or_insert(0);
        *count += 1;
        let count = *count;
        if let Some(path) = extract_path(input) {
            self.paths.insert(fingerprint, path);
        }

        if count < NOTICE_EVERY || !count.is_multiple_of(NOTICE_EVERY) {
            return None;
        }
        Some(notice(tool_name, count))
    }

    fn invalidate_path(&mut self, path: Option<&str>) {
        let Some(path) = path else {
            return;
        };
        self.forget(|tracked| tracked == path);
    }

    /// A shell command can edit anything, and the harness cannot tell a `cat`
    /// from a `sed -i`. Rather than guess, drop every tracked path the command
    /// text mentions: an unnecessary reset only costs a missed notice, while a
    /// missed reset would nudge against a legitimate re-read.
    fn invalidate_mentioned_paths(&mut self, input: &Value) {
        let Some(command) = input.get("command").and_then(Value::as_str) else {
            return;
        };
        let command = command.to_string();
        self.forget(|tracked| command.contains(tracked));
    }

    /// Drop every tracked observation whose path matches `stale`.
    fn forget(&mut self, stale: impl Fn(&str) -> bool) {
        let dropped: Vec<String> = self
            .paths
            .iter()
            .filter(|(_, path)| stale(path))
            .map(|(fingerprint, _)| fingerprint.clone())
            .collect();
        for fingerprint in dropped {
            self.counts.remove(&fingerprint);
            self.paths.remove(&fingerprint);
        }
    }

    #[cfg(test)]
    fn count_of(&self, tool_name: &str, input: &Value) -> u32 {
        self.counts
            .get(&fingerprint(tool_name, input))
            .copied()
            .unwrap_or(0)
    }
}

fn notice(tool_name: &str, count: u32) -> String {
    format!(
        "\n\n<system-reminder>\nYou have now run this exact `{tool_name}` call {count} times in this session, and nothing has changed the source in between. If you already reached a conclusion here, state it and act on it rather than deriving it again. Record findings as you reach them so a later pass reads the record instead of re-inferring it. If you came back because the conclusion was never written down, write it down now.\n</system-reminder>",
    )
}

/// Canonical identity of an observation: tool name plus its identifying
/// arguments, sorted so key order cannot split one fingerprint into two.
fn fingerprint(tool_name: &str, input: &Value) -> String {
    let mut parts: Vec<String> = match input.as_object() {
        Some(map) => map
            .iter()
            .filter(|(key, _)| !NON_IDENTIFYING_KEYS.contains(&key.as_str()))
            .map(|(key, value)| format!("{key}={value}"))
            .collect(),
        None => vec![input.to_string()],
    };
    parts.sort();
    format!("{tool_name}({})", parts.join(","))
}

fn extract_path(input: &Value) -> Option<String> {
    PATH_KEYS.iter().find_map(|key| {
        input
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read(path: &str, intent: &str) -> Value {
        json!({ "file_path": path, "intent": intent })
    }

    #[test]
    fn silent_until_the_third_identical_read() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        assert!(guard.observe("read", &input).is_none());
        assert!(guard.observe("read", &input).is_none());
        assert!(guard.observe("read", &input).is_some());
    }

    #[test]
    fn renotices_on_every_further_multiple() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        let notices = (1..=9)
            .filter(|_| guard.observe("read", &input).is_some())
            .count();
        assert_eq!(notices, 3, "expected notices at 3, 6 and 9");
    }

    #[test]
    fn intent_prose_does_not_split_the_fingerprint() {
        let mut guard = RepeatGuard::new();
        guard.observe("read", &read("/a.rs", "first look"));
        guard.observe("read", &read("/a.rs", "checking again"));
        let notice = guard.observe("read", &read("/a.rs", "one more time"));
        assert!(notice.is_some(), "intent must not be part of the identity");
    }

    #[test]
    fn different_arguments_are_different_observations() {
        let mut guard = RepeatGuard::new();
        for _ in 0..5 {
            guard.observe("read", &read("/a.rs", "x"));
        }
        assert_eq!(guard.count_of("read", &read("/a.rs", "x")), 5);
        assert!(guard.observe("read", &read("/b.rs", "x")).is_none());
        assert_eq!(guard.count_of("read", &read("/b.rs", "x")), 1);
    }

    #[test]
    fn key_order_does_not_split_the_fingerprint() {
        let a = json!({ "file_path": "/a.rs", "offset": 10, "intent": "x" });
        let b = json!({ "offset": 10, "file_path": "/a.rs", "intent": "y" });
        assert_eq!(fingerprint("read", &a), fingerprint("read", &b));
    }

    #[test]
    fn editing_the_file_clears_its_history() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        guard.observe("read", &input);
        guard.observe("read", &input);
        guard.observe("edit", &json!({ "file_path": "/a.rs", "intent": "fix" }));
        assert_eq!(guard.count_of("read", &input), 0);
        assert!(guard.observe("read", &input).is_none());
    }

    #[test]
    fn editing_another_file_leaves_history_intact() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        guard.observe("read", &input);
        guard.observe("read", &input);
        guard.observe("edit", &json!({ "file_path": "/b.rs", "intent": "fix" }));
        assert!(guard.observe("read", &input).is_some());
    }

    #[test]
    fn bash_mentioning_the_path_clears_its_history() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        guard.observe("read", &input);
        guard.observe("read", &input);
        guard.observe("bash", &json!({ "command": "sed -i s/x/y/ /a.rs" }));
        assert_eq!(guard.count_of("read", &input), 0);
    }

    #[test]
    fn bash_elsewhere_leaves_history_intact() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        guard.observe("read", &input);
        guard.observe("read", &input);
        guard.observe("bash", &json!({ "command": "cargo test" }));
        assert!(guard.observe("read", &input).is_some());
    }

    #[test]
    fn bookkeeping_tools_are_never_nudged() {
        let mut guard = RepeatGuard::new();
        let input = json!({ "intent": "track" });
        for _ in 0..10 {
            assert!(guard.observe("todo", &input).is_none());
        }
    }

    #[test]
    fn reset_forgets_everything() {
        let mut guard = RepeatGuard::new();
        let input = read("/a.rs", "look");
        guard.observe("read", &input);
        guard.observe("read", &input);
        guard.reset();
        assert!(guard.observe("read", &input).is_none());
    }
}
