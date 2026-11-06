// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Function tools of a Responses request, including `namespace` tools.
//!
//! Mirrors the Python frontend (`get_function_tools` and
//! `build_responses_tool_call_name_map` in `vllm/tool_parsers/utils.py`,
//! `_resolve_named_tool_choice` in `responses/protocol.py`): the model sees
//! each namespaced function as one flat `<namespace>__<name>` function tool,
//! and a call of that flat name is reported with the function name and its
//! namespace. The flat spelling is defined only by [`ToolIdentity::flat_name`].
//!
//! A client-executed `tool_search` tool is shown as a `tool_search` function,
//! and the tools a `tool_search_output` input item loaded join the request's
//! tools as deferred ones (`defer_loading`), so renderers list them where the
//! search result references them.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use itertools::Itertools as _;
use vllm_chat::ChatTool;

use super::types::{NamespacedTool, ResponsesTool, ToolSearchExecution, ToolSearchTool};
use crate::error::{ApiError, bail_invalid_request};

const NAMESPACE_TOOL_SEPARATOR: &str = "__";

/// The function name the model calls a client-executed tool search by.
pub(super) const TOOL_SEARCH_NAME: &str = "tool_search";

/// A function tool as the client names it: a namespace tool's function
/// carries its namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ToolIdentity<'a> {
    pub namespace: Option<&'a str>,
    pub name: &'a str,
}

impl<'a> ToolIdentity<'a> {
    /// The name the model sees: `<namespace>__<name>`, or the bare name.
    pub(super) fn flat_name(self) -> Cow<'a, str> {
        match self.namespace {
            Some(namespace) => Cow::Owned(format!(
                "{namespace}{NAMESPACE_TOOL_SEPARATOR}{}",
                self.name
            )),
            None => Cow::Borrowed(self.name),
        }
    }
}

/// Request-local map from the flat names of namespaced functions back to
/// their identities, for reporting calls and resolving `tool_choice`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ResponsesToolNames {
    /// Flat name of each namespaced function -> byte length of its namespace.
    /// Like Python, a later function with the same flat name wins.
    namespaced: HashMap<String, usize>,
    /// Whether the request declares a client-executed `tool_search` tool,
    /// whose calls are reported as `tool_search_call` items.
    client_tool_search: bool,
}

impl ResponsesToolNames {
    /// Convert the request tools to the function tools the model is shown, in
    /// request order, and the tools loaded by tool search (`loaded`, in input
    /// order) to deferred ones, together with the names to map calls back. A
    /// loaded tool whose name is already shown is skipped, keeping the first
    /// definition and its identity, as Python merges repeated search results.
    pub(super) fn convert(
        tools: &[ResponsesTool],
        loaded: &[ResponsesTool],
    ) -> Result<ConvertedTools, ApiError> {
        let mut names = Self::default();
        let mut request_tools = Vec::with_capacity(tools.len());
        for tool in tools {
            match tool {
                ResponsesTool::ToolSearch(tool_search) => {
                    request_tools.push(tool_search_function(tool_search)?);
                    names.client_tool_search = true;
                }
                tool => {
                    for (chat_tool, namespace_len) in flatten(tool) {
                        names.record(&chat_tool, namespace_len);
                        request_tools.push(chat_tool);
                    }
                }
            }
        }
        let mut shown = request_tools.iter().map(|tool| tool.name.clone()).collect::<HashSet<_>>();
        let mut loaded_tools = Vec::new();
        for tool in loaded {
            if matches!(tool, ResponsesTool::ToolSearch(_)) {
                bail_invalid_request!(
                    param = "input",
                    "tool_search_output can load only function and namespace tools."
                );
            }
            for (chat_tool, namespace_len) in flatten(tool) {
                if shown.insert(chat_tool.name.clone()) {
                    names.record(&chat_tool, namespace_len);
                    loaded_tools.push(ChatTool {
                        defer_loading: Some(true),
                        ..chat_tool
                    });
                }
            }
        }
        Ok(ConvertedTools {
            request_tools,
            loaded_tools,
            names,
        })
    }

    /// Record the namespace of a shown namespaced function.
    fn record(&mut self, tool: &ChatTool, namespace_len: Option<usize>) {
        if let Some(namespace_len) = namespace_len {
            self.namespaced.insert(tool.name.clone(), namespace_len);
        }
    }

    /// Whether a called tool name is the client-executed tool search.
    pub(super) fn is_tool_search(&self, called_name: &str) -> bool {
        self.client_tool_search && called_name == TOOL_SEARCH_NAME
    }

    /// The identity of a called tool name. Names of other tools pass through
    /// without a namespace.
    pub(super) fn decode<'a>(&self, called_name: &'a str) -> ToolIdentity<'a> {
        match self.namespaced.get(called_name) {
            Some(&namespace_len) => ToolIdentity {
                namespace: Some(&called_name[..namespace_len]),
                name: &called_name[namespace_len + NAMESPACE_TOOL_SEPARATOR.len()..],
            },
            None => ToolIdentity {
                namespace: None,
                name: called_name,
            },
        }
    }

    /// Resolve a named tool choice to the name the model sees, among the
    /// tools it is shown (`shown`, deduplicated).
    ///
    /// An exact tool name (including a flat `<namespace>__<name>`) wins;
    /// otherwise a function name defined in exactly one namespace selects
    /// that function.
    pub(super) fn resolve_choice(
        &self,
        name: String,
        shown: &[ChatTool],
    ) -> Result<String, ApiError> {
        if shown.iter().any(|tool| tool.name == name) {
            return Ok(name);
        }
        let mut candidates = shown
            .iter()
            .filter(|tool| {
                let identity = self.decode(&tool.name);
                identity.namespace.is_some() && identity.name == name
            })
            .map(|tool| tool.name.clone())
            .collect_vec();
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
}

/// The function tools of a Responses request, as the model is shown them.
pub(super) struct ConvertedTools {
    /// The request's tools, in request order.
    pub request_tools: Vec<ChatTool>,
    /// The tools that tool search results in the input loaded, deferred.
    pub loaded_tools: Vec<ChatTool>,
    pub names: ResponsesToolNames,
}

/// The flat function tools of one function or namespace tool, each with the
/// byte length of its namespace when it has one.
fn flatten(tool: &ResponsesTool) -> Vec<(ChatTool, Option<usize>)> {
    match tool {
        ResponsesTool::Function(function) => vec![(ChatTool::from(function.clone()), None)],
        ResponsesTool::Namespace(namespace) => namespace
            .tools
            .iter()
            .map(|NamespacedTool::Function(function)| {
                let flat_name = ToolIdentity {
                    namespace: Some(&namespace.name),
                    name: &function.name,
                }
                .flat_name()
                .into_owned();
                let tool = ChatTool {
                    name: flat_name,
                    ..ChatTool::from(function.clone())
                };
                (tool, Some(namespace.name.len()))
            })
            .collect(),
        ResponsesTool::ToolSearch(_) => {
            unreachable!("tool search tools are converted before flattening")
        }
    }
}

/// The function a client-executed tool search is shown to the model as, like
/// Python's `build_model_tools`. Server-executed tool search is rejected: this
/// frontend executes no tools.
fn tool_search_function(tool_search: &ToolSearchTool) -> Result<ChatTool, ApiError> {
    if tool_search.execution != Some(ToolSearchExecution::Client) {
        bail_invalid_request!(
            param = "tools",
            "Only client-executed tool search is supported; set `execution` to \"client\"."
        );
    }
    Ok(ChatTool {
        name: TOOL_SEARCH_NAME.to_string(),
        description: tool_search.description.clone(),
        parameters: tool_search
            .parameters
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
        strict: None,
        defer_loading: None,
    })
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

        let shown = ResponsesToolNames::convert(&tools, &[]).unwrap().request_tools;

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
        let names = ResponsesToolNames::convert(
            &request_tools(json!([
            {"type": "function", "name": "lookup"},
            namespace("mcp__computer_use", &["get_app_state"]),
            ])),
            &[],
        )
        .unwrap()
        .names;

        assert_eq!(
            names.decode("mcp__computer_use__get_app_state"),
            ToolIdentity {
                namespace: Some("mcp__computer_use"),
                name: "get_app_state",
            }
        );
        let bare = |name| ToolIdentity {
            namespace: None,
            name,
        };
        assert_eq!(names.decode("lookup"), bare("lookup"));
        assert_eq!(names.decode("get_app_state"), bare("get_app_state"));
    }

    #[test]
    fn named_tool_choice_resolves_exact_then_unique_function_names() {
        let tools = request_tools(json!([
            {"type": "function", "name": "lookup"},
            namespace("crm", &["get_contact", "lookup", "search"]),
            namespace("docs", &["search"]),
        ]));
        let converted = ResponsesToolNames::convert(&tools, &[]).unwrap();
        let resolve =
            |name: &str| converted.names.resolve_choice(name.to_string(), &converted.request_tools);

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
