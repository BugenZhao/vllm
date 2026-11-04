// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::cli::unsupported::unsupported_args_error;

/// Python's logging level names, the choices of `--log-level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[value(rename_all = "UPPER")]
#[serde(rename_all = "UPPERCASE")]
pub enum LogLevel {
    Critical,
    Error,
    Warning,
    Info,
    Debug,
    Notset,
}

impl LogLevel {
    /// The Python level name, as `VLLM_LOGGING_LEVEL` spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "CRITICAL",
            Self::Error => "ERROR",
            Self::Warning => "WARNING",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Notset => "NOTSET",
        }
    }
}

/// Python's `LoggingConfig` formatters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormatter {
    Text,
    Json,
}

/// Python's `LoggingConfig`, from `--logging-config` and its dotted forms
/// (`--logging-config.log_level DEBUG`).
///
/// The Rust frontend honors `log_level` and always configures its own text
/// logging, so the other fields are accepted only at those defaults. Set
/// fields are forwarded to the managed engine as given, since Python's
/// defaults come from env vars the user may be overriding.
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
    /// An explicit `null` clears `VLLM_LOGGING_CONFIG_PATH` in Python.
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub pylogging_config_file: Option<Option<String>>,
}

impl LoggingConfig {
    /// Fail on fields set to values the Rust frontend does not implement.
    pub(crate) fn check(&self) -> Result<(), String> {
        let unsupported = [
            (self.formatter == Some(LogFormatter::Json)).then_some("logging_config.formatter"),
            (self.configure_logging == Some(false)).then_some("logging_config.configure_logging"),
            matches!(self.pylogging_config_file, Some(Some(_)))
                .then_some("logging_config.pylogging_config_file"),
        ];
        if unsupported.iter().any(Option::is_some) {
            return Err(unsupported_args_error(unsupported.into_iter().flatten()));
        }
        Ok(())
    }
}
