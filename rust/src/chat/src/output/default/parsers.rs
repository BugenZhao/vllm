// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Output parsers resolved once per model from the server's parser selections.

use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tracing::{info, warn};
use vllm_parser::unified::{CombinedParser, UnifiedParser};
use vllm_text::tokenizer::DynTokenizer;

use crate::parser::ParserSelection;
use crate::parser::reasoning::ReasoningParserFactory;
use crate::parser::tool::{ToolParser, ToolParserFactory};
use crate::parser::unified::{
    BeamUnifiedParser, HfTemplateError, HfUnifiedParser, ResponseTemplate, UnifiedParserFactory,
    names,
};
use crate::request::{ChatRequest, ChatTool, ResolvedToolContext};
use crate::{Error, Result};

/// The output parsers of one model, resolved from the server's tool-call and
/// reasoning parser selections.
///
/// The selections are fixed for the server, so they are resolved and each
/// parser is built once when the model is loaded: an explicit selection the
/// model cannot use fails there instead of in every request, and a parser that
/// `auto` matched by model name but cannot be built is disabled with a warning.
/// Requests then only build the resolved parsers with their own tools.
#[derive(Clone)]
pub struct ResolvedParsers {
    model_id: String,
    tokenizer: DynTokenizer,
    parsers: Parsers,
}

#[derive(Clone)]
enum Parsers {
    /// One parser handles both reasoning and tool calls.
    Unified(UnifiedParserSpec),
    /// Separate, optional reasoning and tool-call parsers.
    Split {
        tool: ToolParserSlot,
        reasoning: Option<String>,
    },
}

#[derive(Clone)]
enum UnifiedParserSpec {
    /// Built by its registry constructor.
    Registered(String),
    // Parsers driven by the model's template are built from it, never by
    // their registry constructors.
    Hf(Arc<ResponseTemplate>),
    Beam(Arc<ResponseTemplate>),
}

/// The tool parser of [`Parsers::Split`], or why there is none.
#[derive(Clone)]
enum ToolParserSlot {
    Parser(String),
    /// Tool parsing is turned off (`none`).
    Disabled,
    /// `auto` found no usable tool parser for the model.
    Unavailable,
}

impl ResolvedParsers {
    /// Resolve and build the selected parsers for `model_id`.
    pub fn new(
        model_id: &str,
        tokenizer: DynTokenizer,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
    ) -> Result<Self> {
        Self::with_response_template(
            model_id,
            tokenizer,
            &Err(HfTemplateError::Missing),
            tool_call_parser,
            reasoning_parser,
        )
    }

    /// Like [`Self::new`], additionally providing the model's `response_template`
    /// for the `hf` parser, or why it is unavailable.
    pub fn with_response_template(
        model_id: &str,
        tokenizer: DynTokenizer,
        response_template: &std::result::Result<Arc<ResponseTemplate>, HfTemplateError>,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
    ) -> Result<Self> {
        let tool_auto = matches!(tool_call_parser, ParserSelection::Auto);
        let reasoning_auto = matches!(reasoning_parser, ParserSelection::Auto);
        let tool_name = tool_call_parser.resolve_tool_name(model_id);
        let reasoning_name = reasoning_parser.resolve_reasoning_name(model_id);
        // `hf` is opt-in: `Auto` never selects it, but a usable template is worth a hint.
        if (tool_auto || reasoning_auto)
            && tool_name.is_none()
            && reasoning_name.is_none()
            && response_template.is_ok()
        {
            info!(
                "the model ships a response_template; pass `--tool-call-parser hf \
                 --reasoning-parser hf` to parse with it"
            );
        }

        let unified_factory = UnifiedParserFactory::global();
        let parsers = if let Some(unified_name) = tool_name
            .into_iter()
            .chain(reasoning_name)
            .find(|name| unified_factory.contains(name))
        {
            // Another name of a parser is the same parser.
            let tool_canonical = tool_name.map(|name| unified_factory.canonical_name(name));
            let reasoning_canonical =
                reasoning_name.map(|name| unified_factory.canonical_name(name));
            let unified = if tool_canonical == reasoning_canonical {
                UnifiedParserSpec::resolve(unified_name, response_template).and_then(|spec| {
                    spec.create(&[], tokenizer.clone())?;
                    Ok(spec)
                })
            } else {
                Err(Error::IncompatibleParserSelections {
                    tool: tool_name.unwrap_or("none").to_owned(),
                    reasoning: reasoning_name.unwrap_or("none").to_owned(),
                })
            };
            match unified {
                Ok(spec) => {
                    info!(parser_name = spec.name(), "using unified parser");
                    Parsers::Unified(spec)
                }
                Err(error) if tool_auto && reasoning_auto => {
                    warn_unusable_auto_parser("unified", unified_name, model_id, &error);
                    Parsers::Split {
                        tool: ToolParserSlot::Unavailable,
                        reasoning: None,
                    }
                }
                Err(error) => return Err(error),
            }
        } else {
            let tool = match tool_name {
                Some(name) => {
                    match ToolParserFactory::global().create(name, &[], tokenizer.clone()) {
                        Ok(_) => {
                            info!(parser_name = name, "using tool parser");
                            ToolParserSlot::Parser(name.to_owned())
                        }
                        Err(error) if tool_auto => {
                            warn_unusable_auto_parser("tool", name, model_id, &error);
                            ToolParserSlot::Unavailable
                        }
                        Err(error) => return Err(error),
                    }
                }
                None if matches!(tool_call_parser, ParserSelection::None) => {
                    ToolParserSlot::Disabled
                }
                None => ToolParserSlot::Unavailable,
            };
            let reasoning = match reasoning_name {
                Some(name) => {
                    match ReasoningParserFactory::global().create(name, tokenizer.clone()) {
                        Ok(_) => {
                            info!(parser_name = name, "using reasoning parser");
                            Some(name.to_owned())
                        }
                        Err(error) if reasoning_auto => {
                            warn_unusable_auto_parser("reasoning", name, model_id, &error);
                            None
                        }
                        Err(error) => return Err(error),
                    }
                }
                None => None,
            };
            if reasoning.is_none() {
                info!("reasoning parsing disabled");
            }
            Parsers::Split { tool, reasoning }
        };

        Ok(Self {
            model_id: model_id.to_owned(),
            tokenizer,
            parsers,
        })
    }

    /// Effective tool-call parser name, if tool calls are parsed.
    pub fn tool_call_parser_name(&self) -> Option<&str> {
        match &self.parsers {
            Parsers::Unified(spec) => Some(spec.name()),
            Parsers::Split {
                tool: ToolParserSlot::Parser(name),
                ..
            } => Some(name),
            Parsers::Split { .. } => None,
        }
    }

    /// Effective reasoning parser name, if reasoning is parsed.
    pub fn reasoning_parser_name(&self) -> Option<&str> {
        match &self.parsers {
            Parsers::Unified(spec) => Some(spec.name()),
            Parsers::Split { reasoning, .. } => reasoning.as_deref(),
        }
    }

    /// Build the parser for one request, given its tools.
    pub(super) fn create(&self, request: &ChatRequest) -> Result<Box<dyn UnifiedParser>> {
        match &self.parsers {
            Parsers::Unified(spec) => spec.create(request.tools(), self.tokenizer.clone()),
            Parsers::Split { tool, reasoning } => {
                let tool_parser = self.create_tool_parser(tool, &request.tool_context)?;
                let reasoning_parser = reasoning
                    .as_deref()
                    .map(|name| {
                        ReasoningParserFactory::global().create(name, self.tokenizer.clone())
                    })
                    .transpose()?;
                Ok(Box::new(CombinedParser::new(reasoning_parser, tool_parser)))
            }
        }
    }

    /// Build the configured tool parser for a request with tools.
    ///
    /// With tool parsing off (`tool_choice=none`), a configured parser still
    /// parses the calls so that `max_tool_calls` drops them instead of leaking
    /// their markup into content. This holds for every parser, as for Python's
    /// parser engines; Python's other parsers (legacy tool parsers, the HF
    /// response-template parser, the Llama JSON parser) keep the raw call as
    /// content instead. A missing parser is an error only when tool parsing is
    /// on.
    fn create_tool_parser(
        &self,
        slot: &ToolParserSlot,
        tool_context: &ResolvedToolContext,
    ) -> Result<Option<Box<dyn ToolParser>>> {
        let tools = &tool_context.effective_tools;
        if tools.is_empty() {
            return Ok(None);
        }
        let parser_name = match slot {
            ToolParserSlot::Parser(name) => name,
            _ if !tool_context.parsing_enabled() => return Ok(None),
            ToolParserSlot::Disabled => return Err(Error::ParserDisabled { kind: "tool" }),
            ToolParserSlot::Unavailable => {
                return Err(Error::ParserUnavailableForModel {
                    kind: "tool",
                    model_id: self.model_id.clone(),
                });
            }
        };
        ToolParserFactory::global()
            .create(parser_name, tools, self.tokenizer.clone())
            .map(Some)
    }
}

impl UnifiedParserSpec {
    fn resolve(
        name: &str,
        response_template: &std::result::Result<Arc<ResponseTemplate>, HfTemplateError>,
    ) -> Result<Self> {
        let template = || {
            response_template.clone().map_err(|error| Error::ParserInitialization {
                kind: "unified",
                name: name.to_string(),
                error: Box::new(error),
            })
        };
        Ok(match name {
            names::HF => Self::Hf(template()?),
            names::BEAM => Self::Beam(template()?),
            _ => Self::Registered(name.to_owned()),
        })
    }

    fn name(&self) -> &str {
        match self {
            Self::Registered(name) => name,
            Self::Hf(_) => names::HF,
            Self::Beam(_) => names::BEAM,
        }
    }

    fn create(
        &self,
        tools: &[ChatTool],
        tokenizer: DynTokenizer,
    ) -> Result<Box<dyn UnifiedParser>> {
        Ok(match self {
            Self::Registered(name) => {
                UnifiedParserFactory::global().create(name, tools, tokenizer)?
            }
            Self::Hf(template) => {
                Box::new(HfUnifiedParser::new(template.clone(), tools, tokenizer))
            }
            Self::Beam(template) => {
                Box::new(BeamUnifiedParser::new(template.clone(), tools, tokenizer))
            }
        })
    }
}

/// Disable a parser that `auto` matched by model name but cannot build for
/// this model: `auto` picks a parser that fits the model, and without one the
/// model is served without that parsing, like Python's default.
fn warn_unusable_auto_parser(kind: &str, name: &str, model_id: &str, error: &Error) {
    warn!(
        model_id,
        parser_name = name,
        error = %error.as_report(),
        "the {kind} parser selected automatically for this model cannot be used with it; \
         {kind} parsing is disabled, select a parser explicitly to require one"
    );
}
