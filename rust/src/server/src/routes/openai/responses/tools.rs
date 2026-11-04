// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Function tools of a Responses request, including `namespace` tools.
//!
//! Mirrors the Python frontend (`get_function_tools` and
//! `build_responses_tool_call_name_map` in `vllm/tool_parsers/utils.py`,
//! `_resolve_named_tool_choice` in `responses/protocol.py`): the model sees
//! each namespaced function as one flat `<namespace>__<name>` function tool,
//! and a call of that flat name is reported with the function name and its
//! namespace.

use std::collections::HashMap;

use itertools::Itertools as _;
use vllm_chat::ChatTool;

use super::types::{NamespacedTool, ResponsesTool};
use crate::error::{ApiError, bail_invalid_request};

const NAMESPACE_TOOL_SEPARATOR: &str = "__";

/// The flat name the model sees for one namespaced function.
pub(super) fn flat_tool_name(namespace: &str, name: &str) -> String {
    format!("{namespace}{NAMESPACE_TOOL_SEPARATOR}{name}")
}

/// Return the function tools the model is shown, in request order.
pub(super) fn function_tools(tools: &[ResponsesTool]) -> Vec<ChatTool> {
    let mut function_tools = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool {
            ResponsesTool::Function(function) => {
                function_tools.push(ChatTool::from(function.clone()));
            }
            ResponsesTool::Namespace(namespace) => {
                function_tools.extend(namespace.tools.iter().map(
                    |NamespacedTool::Function(function)| ChatTool {
                        name: flat_tool_name(&namespace.name, &function.name),
                        ..ChatTool::from(function.clone())
                    },
                ));
            }
        }
    }
    function_tools
}

/// Resolve a named tool choice to the tool name the model sees.
///
/// An exact tool name (including a flat `<namespace>__<name>`) wins; otherwise
/// a function name defined in exactly one namespace selects that function.
pub(super) fn resolve_named_tool_choice(
    name: String,
    tools: &[ResponsesTool],
) -> Result<String, ApiError> {
    let mut candidates = Vec::new();
    for tool in tools {
        match tool {
            ResponsesTool::Function(function) => {
                if function.name == name {
                    return Ok(name);
                }
            }
            ResponsesTool::Namespace(namespace) => {
                for NamespacedTool::Function(function) in &namespace.tools {
                    let flat_name = flat_tool_name(&namespace.name, &function.name);
                    if flat_name == name {
                        return Ok(name);
                    }
                    if function.name == name {
                        candidates.push(flat_name);
                    }
                }
            }
        }
    }
    match candidates.len() {
        0 => bail_invalid_request!(
            param = "tool_choice",
            "Tool choice 'function' not found in 'tools' parameter."
        ),
        1 => Ok(candidates.pop().expect("one candidate")),
        _ => bail_invalid_request!(
            param = "tool_choice",
            "Tool choice '{name}' is ambiguous: it is defined in more than one namespace ({}). \
             Select it by its full '<namespace>__<name>' name.",
            candidates.iter().join(", ")
        ),
    }
}

/// Namespaced functions keyed by the flat name the model calls them by.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct NamespacedToolNames(HashMap<String, NamespacedName>);

#[derive(Debug, Clone, PartialEq)]
struct NamespacedName {
    namespace: String,
    name: String,
}

impl NamespacedToolNames {
    pub(super) fn new(tools: &[ResponsesTool]) -> Self {
        let mut names = HashMap::new();
        for tool in tools {
            let ResponsesTool::Namespace(namespace) = tool else {
                continue;
            };
            for NamespacedTool::Function(function) in &namespace.tools {
                names.insert(
                    flat_tool_name(&namespace.name, &function.name),
                    NamespacedName {
                        namespace: namespace.name.clone(),
                        name: function.name.clone(),
                    },
                );
            }
        }
        Self(names)
    }

    /// Split a called tool name into the function name and its namespace.
    /// Names of other tools pass through without a namespace.
    pub(super) fn resolve<'a>(&'a self, called_name: &'a str) -> (&'a str, Option<&'a str>) {
        match self.0.get(called_name) {
            Some(NamespacedName { namespace, name }) => (name, Some(namespace)),
            None => (called_name, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use thiserror_ext::AsReport as _;

    use super::*;

    fn request_tools(tools: serde_json::Value) -> Vec<ResponsesTool> {
        serde_json::from_value(tools).expect("deserialize tools")
    }

    fn namespace(name: &str, functions: &[&str]) -> serde_json::Value {
        json!({
            "type": "namespace",
            "name": name,
            "description": "Grouped tools.",
            "tools": functions
                .iter()
                .map(|function| json!({"type": "function", "name": function}))
                .collect::<Vec<_>>(),
        })
    }

    #[test]
    fn namespaced_functions_are_shown_flat_in_request_order() {
        let tools = request_tools(json!([
            {"type": "function", "name": "lookup", "description": "Top level."},
            {
                "type": "namespace",
                "name": "crm",
                "description": "CRM tools.",
                "tools": [{
                    "type": "function",
                    "name": "get_contact",
                    "description": "Get a contact.",
                    "parameters": {"type": "object"},
                    "strict": true
                }]
            }
        ]));

        let shown = function_tools(&tools);

        let names = shown.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>();
        assert_eq!(names, ["lookup", "crm__get_contact"]);
        assert_eq!(shown[1].description.as_deref(), Some("Get a contact."));
        assert_eq!(shown[1].parameters, json!({"type": "object"}));
        assert_eq!(shown[1].strict, Some(true));
    }

    #[test]
    fn custom_tools_inside_a_namespace_are_rejected() {
        let error = serde_json::from_value::<Vec<ResponsesTool>>(json!([{
            "type": "namespace",
            "name": "crm",
            "description": "CRM tools.",
            "tools": [{"type": "custom", "name": "raw"}]
        }]))
        .expect_err("custom tool inside a namespace");

        let report = error.to_report_string();
        assert!(report.contains("unknown variant `custom`"), "{report}");
    }

    #[test]
    fn called_flat_names_resolve_to_the_namespaced_function() {
        let names = NamespacedToolNames::new(&request_tools(json!([
            {"type": "function", "name": "lookup"},
            namespace("mcp__computer_use", &["get_app_state"]),
        ])));

        assert_eq!(
            names.resolve("mcp__computer_use__get_app_state"),
            ("get_app_state", Some("mcp__computer_use"))
        );
        assert_eq!(names.resolve("lookup"), ("lookup", None));
        assert_eq!(names.resolve("get_app_state"), ("get_app_state", None));
    }

    #[test]
    fn named_tool_choice_resolves_exact_then_unique_function_names() {
        let tools = request_tools(json!([
            {"type": "function", "name": "lookup"},
            namespace("crm", &["get_contact", "lookup", "search"]),
            namespace("docs", &["search"]),
        ]));
        let resolve = |name: &str| resolve_named_tool_choice(name.to_string(), &tools);

        assert_eq!(resolve("lookup").unwrap(), "lookup");
        assert_eq!(resolve("crm__lookup").unwrap(), "crm__lookup");
        assert_eq!(resolve("get_contact").unwrap(), "crm__get_contact");
        assert_eq!(resolve("docs__search").unwrap(), "docs__search");

        let invalid_tool_choice = |name: &str| {
            let error = resolve(name).expect_err("invalid tool choice");
            let ApiError::InvalidRequest { message, param } = error else {
                panic!("unexpected error: {error:?}");
            };
            assert_eq!(param, Some("tool_choice"));
            message
        };
        assert_eq!(
            invalid_tool_choice("search"),
            "Tool choice 'search' is ambiguous: it is defined in more than one namespace \
             (crm__search, docs__search). Select it by its full '<namespace>__<name>' name."
        );
        assert_eq!(
            invalid_tool_choice("missing"),
            "Tool choice 'function' not found in 'tools' parameter."
        );
    }
}
