// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `--config <FILE>` for `vllm-rs serve`, following Python's
//! `FlexibleArgumentParser`: the YAML file's options are expanded into
//! command-line options, which the caller places before the ones given on the
//! command line, so the command line takes precedence.

use std::ffi::OsString;

use clap::error::ErrorKind;
use serde_json::{Map, Value};

use super::find_long_arg;

/// Options expanded from a `--config` file.
#[derive(Debug, Default)]
pub(super) struct ConfigFileArgs {
    /// The `model` key, used when the command line gives no model.
    pub model: Option<OsString>,
    /// Every other key as command-line options.
    pub args: Vec<OsString>,
}

/// Load the file named by the first `--config` in `args`, if any. The
/// `--config` option itself stays in `args`, so clap still sees it.
pub(super) fn load_config_file(
    args: &[OsString],
    command: &clap::Command,
) -> Result<ConfigFileArgs, clap::Error> {
    let Some(path) = find_config_path(args) else {
        return Ok(ConfigFileArgs::default());
    };
    let error = |message: String| command.clone().error(ErrorKind::InvalidValue, message);
    let content = std::fs::read_to_string(&path).map_err(|e| {
        error(format!(
            "failed to read the config file {}: {e}",
            path.display()
        ))
    })?;
    // Resolve `<<` merge keys like PyYAML's `safe_load`.
    let config: Map<String, Value> = serde_yaml_ng::from_str(&content)
        .and_then(|mut config: serde_yaml_ng::Value| {
            config.apply_merge()?;
            serde_yaml_ng::from_value(config)
        })
        .map_err(|e| {
            error(format!(
                "invalid config file {}: expected a YAML mapping of options: {e}",
                path.display()
            ))
        })?;
    Ok(expand_config(config, command))
}

/// The value of the first `--config <FILE>` or `--config=<FILE>`.
fn find_config_path(args: &[OsString]) -> Option<OsString> {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--config" {
            return args.next().cloned();
        }
        if let Some(path) = arg.to_str().and_then(|arg| arg.strip_prefix("--config=")) {
            return Some(path.into());
        }
    }
    None
}

/// Python's `load_config_file`: a key becomes `--<key>`, a mapping its JSON
/// text, a list one argument per element, and `true` a bare switch. Python
/// passes `false` as `--no-<key>` when the parser has it and drops it
/// otherwise; an option `vllm-rs` declares gets `--<key> false` when it takes
/// a value and stays off when it is a switch, and the rest go to the managed
/// engine as `--no-<key>`. A `null` value leaves the option at its default
/// (Python would pass the string `None`).
fn expand_config(config: Map<String, Value>, command: &clap::Command) -> ConfigFileArgs {
    let mut expanded = ConfigFileArgs::default();
    for (key, value) in config {
        if key == "model" {
            expanded.model = Some(scalar_arg(value).into());
            continue;
        }
        let option = format!("--{key}");
        let mut push = |arg: String| expanded.args.push(arg.into());
        // A dotted key (`hf-overrides.rope_scaling: null`) takes its value as
        // one argument, like `--hf-overrides.rope_scaling null`.
        if key.contains('.') {
            push(option);
            push(scalar_arg(value));
            continue;
        }
        let declared = find_long_arg(command, &key.replace('_', "-"));
        match value {
            Value::Null => {}
            Value::Bool(value) => match declared {
                // An option that takes a value gets it, as on the command line.
                Some(arg) if arg.get_action().takes_values() => {
                    push(option);
                    push(value.to_string());
                }
                Some(_) => {
                    if value {
                        push(option);
                    }
                }
                None => push(if value { option } else { format!("--no-{key}") }),
            },
            // An option `vllm-rs` declares with one value per occurrence, such
            // as `--api-key`, is repeated per element.
            Value::Array(items)
                if declared
                    .is_some_and(|arg| arg.get_num_args().is_none_or(|n| n.max_values() <= 1)) =>
            {
                for item in items {
                    push(option.clone());
                    push(scalar_arg(item));
                }
            }
            Value::Array(items) => {
                if !items.is_empty() {
                    push(option);
                    items.into_iter().map(scalar_arg).for_each(push);
                }
            }
            value => {
                push(option);
                push(scalar_arg(value));
            }
        }
    }
    expanded
}

/// A value as one command-line argument: strings as they are, anything else
/// as JSON.
fn scalar_arg(value: Value) -> String {
    match value {
        Value::String(value) => value,
        value => value.to_string(),
    }
}
