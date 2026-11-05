// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Deserializer, Serialize};
use serde_with::{DisplayFromStr, serde_as};
use vllm_tracing::LogLevel;

use crate::cli::unsupported::DefaultOnly;

/// Python's `LoggingConfig`, from `--logging-config` and its dotted forms
/// (`--logging-config.log_level DEBUG`).
///
/// The Rust frontend honors `log_level` and always configures its own text
/// logging, so the other fields accept only those defaults. Set fields are
/// forwarded to the managed engine as given, since Python's defaults for them
/// come from env vars the user may be overriding.
#[serde_as]
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    #[serde_as(as = "Option<DisplayFromStr>")]
    #[serde(default)]
    pub log_level: Option<LogLevel>,
    #[serde(default)]
    pub formatter: Option<DefaultOnly<LogFormatter>>,
    #[serde(default)]
    pub configure_logging: Option<DefaultOnly<ConfigureLogging>>,
    /// An explicit `null` clears `VLLM_LOGGING_CONFIG_PATH` in Python, so it is
    /// kept apart from omission.
    #[serde(default, deserialize_with = "deserialize_present")]
    pub pylogging_config_file: Option<DefaultOnly<Option<String>>>,
}

/// Python's `LoggingConfig` formatters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormatter {
    #[default]
    Text,
    Json,
}

/// Python's `configure_logging`, `true` unless `VLLM_CONFIGURE_LOGGING=0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct ConfigureLogging(bool);

impl Default for ConfigureLogging {
    fn default() -> Self {
        Self(true)
    }
}

/// Deserialize a present value, including `null`, as `Some`.
fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}
