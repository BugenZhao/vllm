// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Environment variables that the Python frontend reads but the Rust frontend
//! does not honor yet.
//!
//! Like [`super::unsupported::Unsupported`] arguments, they fail startup when
//! set to a non-default value instead of being silently ignored.

use itertools::Itertools;

/// Predicate that tells whether an environment variable value differs from the
/// Python default.
type IsNonDefault = fn(&str) -> bool;

/// Unsupported environment variables, each with its [`IsNonDefault`] check.
const UNSUPPORTED_ENVS: &[(&str, IsNonDefault)] = &[
    // Download models and tokenizers from ModelScope instead of the
    // Hugging Face Hub.
    ("VLLM_USE_MODELSCOPE", is_truthy),
    // Accept any requested model name and serve it with the base model.
    ("VLLM_SKIP_MODEL_NAME_VALIDATION", is_truthy),
    // Disabling strict tool calling turns off structural tags for required
    // and named `tool_choice`.
    ("VLLM_ENFORCE_STRICT_TOOL_CALLING", |value| {
        !is_truthy(value)
    }),
    // SageMaker stateful sessions on `/invocations`, read by the
    // `model_hosting_container_standards` package as a pydantic bool.
    ("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", is_pydantic_truthy),
];

/// Python parses these flags as `value.strip().lower() in ("1", "true")`.
fn is_truthy(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(), "1" | "true")
}

/// Pydantic's lax bool parsing of a string.
fn is_pydantic_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_lowercase().as_str(),
        "1" | "on" | "t" | "true" | "y" | "yes"
    )
}

/// Check that no unsupported environment variable is set to a non-default
/// value, using `lookup` to read the environment.
pub fn check(lookup: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    let set = UNSUPPORTED_ENVS
        .iter()
        .filter(|(name, is_non_default)| lookup(name).is_some_and(|value| is_non_default(&value)))
        .map(|(name, _)| *name)
        .sorted_unstable()
        .collect_vec();

    if set.is_empty() {
        return Ok(());
    }

    let bullets = set.into_iter().map(|name| format!("- {name}")).join("\n");
    Err(format!(
        "
The following environment variables are not implemented in Rust frontend yet:
{bullets}

Unset them or restore their default values to continue."
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use expect_test::expect;

    use super::check;

    fn check_with(vars: &[(&str, &str)]) -> Result<(), String> {
        let vars: HashMap<_, _> = vars.iter().copied().collect();
        check(|name| vars.get(name).map(|value| value.to_string()))
    }

    #[test]
    fn default_values_pass() {
        check_with(&[]).unwrap();
        check_with(&[("VLLM_USE_MODELSCOPE", "0")]).unwrap();
        check_with(&[("VLLM_SKIP_MODEL_NAME_VALIDATION", "false")]).unwrap();
        check_with(&[("VLLM_ENFORCE_STRICT_TOOL_CALLING", "true")]).unwrap();
        check_with(&[("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", "false")]).unwrap();
    }

    #[test]
    fn non_default_values_fail_startup() {
        let cases = [
            ("VLLM_USE_MODELSCOPE", "True"),
            ("VLLM_SKIP_MODEL_NAME_VALIDATION", "1"),
            ("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0"),
            ("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", "yes"),
        ];
        for (name, value) in cases {
            let error = check_with(&[(name, value)]).unwrap_err();
            assert!(error.contains(&format!("- {name}\n")), "{name}: {error}");
        }
    }

    #[test]
    fn error_message_names_the_variables() {
        expect![[r#"

            The following environment variables are not implemented in Rust frontend yet:
            - VLLM_USE_MODELSCOPE

            Unset them or restore their default values to continue."#]]
        .assert_eq(&check_with(&[("VLLM_USE_MODELSCOPE", "True")]).unwrap_err());
    }
}
