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
    // Disabling strict tool calling turns off structural tags for required
    // and named `tool_choice`.
    ("VLLM_ENFORCE_STRICT_TOOL_CALLING", |value| {
        !is_truthy(value)
    }),
    // Log level of the request JSON body under `--enable-log-requests`,
    // upper-cased by Python.
    ("VLLM_REQUEST_BODY_LOG_LEVEL", |value| {
        value.to_uppercase() != "DEBUG"
    }),
];

/// Python parses these flags as `value.strip().lower() in ("1", "true")`.
fn is_truthy(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(), "1" | "true")
}

/// LoRA resolver variables that Python would act on. Its resolvers are Python
/// plugins Rust cannot load: they run only under
/// `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, and each loads only through the
/// `VLLM_PLUGINS` allowlist, which also covers the engine's unrelated plugins.
fn active_lora_resolver_envs(lookup: &impl Fn(&str) -> Option<String>) -> Vec<&'static str> {
    if !lookup("VLLM_ALLOW_RUNTIME_LORA_UPDATING").is_some_and(|value| is_truthy(&value)) {
        return Vec::new();
    }
    let allowed_plugins = lookup("VLLM_PLUGINS");
    let is_named = |plugin: &str| {
        allowed_plugins
            .as_deref()
            .is_some_and(|plugins| plugins.split(',').any(|name| name == plugin))
    };
    // Python allows every plugin when `VLLM_PLUGINS` is unset, but always asks
    // for the Hugging Face Hub resolver by name because it downloads adapters.
    let resolvers = [
        (
            "VLLM_LORA_RESOLVER_CACHE_DIR",
            allowed_plugins.is_none() || is_named("lora_filesystem_resolver"),
        ),
        (
            "VLLM_LORA_RESOLVER_HF_REPO_LIST",
            is_named("lora_hf_hub_resolver"),
        ),
    ];
    resolvers
        .into_iter()
        .filter(|(name, enabled)| *enabled && lookup(name).is_some_and(|value| !value.is_empty()))
        .map(|(name, _)| name)
        .collect()
}

/// Check that no unsupported environment variable is set to a non-default
/// value, using `lookup` to read the environment.
pub fn check(lookup: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    let set = UNSUPPORTED_ENVS
        .iter()
        .filter(|(name, is_non_default)| lookup(name).is_some_and(|value| is_non_default(&value)))
        .map(|(name, _)| *name)
        .chain(active_lora_resolver_envs(&lookup))
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
        check_with(&[("VLLM_ENFORCE_STRICT_TOOL_CALLING", "true")]).unwrap();
        check_with(&[("VLLM_REQUEST_BODY_LOG_LEVEL", "debug")]).unwrap();
    }

    #[test]
    fn non_default_values_fail_startup() {
        let cases = [
            ("VLLM_USE_MODELSCOPE", "True"),
            ("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0"),
            ("VLLM_REQUEST_BODY_LOG_LEVEL", "INFO"),
        ];
        for (name, value) in cases {
            let error = check_with(&[(name, value)]).unwrap_err();
            assert!(error.contains(&format!("- {name}\n")), "{name}: {error}");
        }
    }

    #[test]
    fn lora_resolver_envs_fail_only_where_python_consults_them() {
        const RUNTIME: (&str, &str) = ("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True");
        const DIR: (&str, &str) = ("VLLM_LORA_RESOLVER_CACHE_DIR", "/loras");
        const REPOS: (&str, &str) = ("VLLM_LORA_RESOLVER_HF_REPO_LIST", "org/adapters");
        let plugins = |value| ("VLLM_PLUGINS", value);

        for vars in [
            vec![RUNTIME, DIR],
            vec![RUNTIME, DIR, plugins("other,lora_filesystem_resolver")],
            vec![RUNTIME, REPOS, plugins("lora_hf_hub_resolver")],
        ] {
            check_with(&vars).expect_err(&format!("{vars:?}"));
        }

        for vars in [
            vec![DIR],
            vec![REPOS, plugins("lora_hf_hub_resolver")],
            vec![RUNTIME, ("VLLM_LORA_RESOLVER_CACHE_DIR", "")],
            vec![RUNTIME, DIR, plugins("")],
            vec![RUNTIME, DIR, plugins("some_engine_plugin")],
            vec![RUNTIME, REPOS],
            vec![RUNTIME, REPOS, plugins("lora_filesystem_resolver")],
            vec![RUNTIME, plugins("some_engine_plugin")],
        ] {
            check_with(&vars).unwrap_or_else(|error| panic!("{vars:?}: {error}"));
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
