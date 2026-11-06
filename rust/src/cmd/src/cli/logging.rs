// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use vllm_tracing::LogLevel;

/// Python's `LoggingConfig`, from `--logging-config` and its dotted forms
/// (`--logging-config.log_level DEBUG`).
///
/// The Rust frontend's own logs honor `log_level` and `formatter`.
/// `configure_logging` and `pylogging_config_file` configure Python logging
/// only; they are kept as given and forwarded to the managed engine, since
/// Python's defaults for them come from env vars.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    #[serde(default)]
    pub log_level: Option<LogLevel>,
    #[serde(default)]
    pub formatter: Option<LogFormatter>,
    #[serde(default)]
    pub configure_logging: Option<bool>,
    /// An explicit `null` clears `VLLM_LOGGING_CONFIG_PATH` in Python, so it is
    /// kept apart from omission.
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub pylogging_config_file: Option<Option<String>>,
}

impl LoggingConfig {
    /// The fields whose value changes Python logging in a way the Rust
    /// frontend's own logs do not follow.
    pub fn python_only_fields(&self) -> Vec<&'static str> {
        [
            ("configure_logging", self.configure_logging == Some(false)),
            (
                "pylogging_config_file",
                matches!(self.pylogging_config_file, Some(Some(_))),
            ),
        ]
        .into_iter()
        .filter_map(|(field, set)| set.then_some(field))
        .collect()
    }
}

/// Python's `LoggingConfig` formatters that the Rust frontend implements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormatter {
    #[default]
    Text,
}
