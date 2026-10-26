// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::str::FromStr;

use minijinja::machinery::ast::{Call, CallArg, Expr, ForLoop, Macro, Set, Stmt};
use minijinja::machinery::{WhitespaceConfig, parse};
use minijinja::syntax::SyntaxConfig;
use serde_with::{DeserializeFromStr, SerializeDisplay};

/// Chat template content format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAi,
}

/// Configurable chat-template content format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay)]
pub enum ChatTemplateContentFormatOption {
    /// Detect the format from the template source.
    #[default]
    Auto,
    /// Always flatten content into plain strings before rendering.
    String,
    /// Always pass content through in OpenAI-compatible structured form.
    OpenAi,
}

impl ChatTemplateContentFormatOption {
    pub const AUTO_LITERAL: &str = "auto";
    pub const OPENAI_LITERAL: &str = "openai";
    pub const STRING_LITERAL: &str = "string";
}

impl FromStr for ChatTemplateContentFormatOption {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::STRING_LITERAL) {
            Ok(Self::String)
        } else if value.eq_ignore_ascii_case(Self::OPENAI_LITERAL) {
            Ok(Self::OpenAi)
        } else {
            Err(format!(
                "invalid content format `{value}`; expected one of: {}, {}, {}",
                Self::AUTO_LITERAL,
                Self::STRING_LITERAL,
                Self::OPENAI_LITERAL
            ))
        }
    }
}

impl fmt::Display for ChatTemplateContentFormatOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::String => f.write_str(Self::STRING_LITERAL),
            Self::OpenAi => f.write_str(Self::OPENAI_LITERAL),
        }
    }
}

fn is_var_access(expr: &Expr, varname: &str) -> bool {
    match expr {
        Expr::Var(v) => v.id == varname,
        // `ns.rest` matches the dotted alias "ns.rest".
        Expr::GetAttr(g) => varname
            .rsplit_once('.')
            .is_some_and(|(base, attr)| g.name == attr && is_var_access(&g.expr, base)),
        _ => false,
    }
}

/// Render an attribute-assignment target like `ns.rest` as a dotted name.
fn attr_dotted_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Var(v) => Some(v.id.to_string()),
        Expr::GetAttr(g) => attr_dotted_name(&g.expr).map(|base| format!("{base}.{}", g.name)),
        _ => None,
    }
}

fn is_const_str(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
}

fn is_attr_access(expr: &Expr, varname: &str, key: &str) -> bool {
    match expr {
        Expr::GetItem(g) => is_var_access(&g.expr, varname) && is_const_str(&g.subscript_expr, key),
        Expr::GetAttr(g) => is_var_access(&g.expr, varname) && g.name == key,
        _ => false,
    }
}

fn is_var_or_elems_access(expr: &Expr, varname: &str, key: Option<&str>) -> bool {
    match expr {
        Expr::Filter(f) => {
            f.expr.as_ref().is_some_and(|inner| is_var_or_elems_access(inner, varname, key))
        }
        Expr::Test(t) => is_var_or_elems_access(&t.expr, varname, key),
        Expr::Slice(s) => is_var_or_elems_access(&s.expr, varname, key),
        _ => key.map_or_else(
            || is_var_access(expr, varname),
            |key| is_attr_access(expr, varname, key),
        ),
    }
}

/// The AST nodes content-format detection inspects, collected in one walk.
#[derive(Default)]
struct TemplateNodes<'a> {
    assignments: Vec<&'a Set<'a>>,
    /// Every for loop, with the indices into `macros` of the macros enclosing
    /// it (outermost first).
    loops: Vec<(&'a ForLoop<'a>, Vec<usize>)>,
    macros: Vec<&'a Macro<'a>>,
    /// Every call expression, including the call of a call block.
    calls: Vec<&'a Call<'a>>,
}

impl<'a> TemplateNodes<'a> {
    fn collect(root: &'a Stmt<'a>) -> Self {
        let mut nodes = Self::default();
        nodes.visit_stmt(root, &mut Vec::new());
        nodes
    }

    fn visit_stmts(&mut self, stmts: &'a [Stmt<'a>], macros: &mut Vec<usize>) {
        for stmt in stmts {
            self.visit_stmt(stmt, macros);
        }
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt<'a>, macros: &mut Vec<usize>) {
        match stmt {
            Stmt::Template(t) => self.visit_stmts(&t.children, macros),
            Stmt::EmitExpr(e) => self.visit_expr(&e.expr),
            Stmt::ForLoop(fl) => {
                self.loops.push((fl, macros.clone()));
                self.visit_expr(&fl.iter);
                if let Some(filter_expr) = &fl.filter_expr {
                    self.visit_expr(filter_expr);
                }
                self.visit_stmts(&fl.body, macros);
                self.visit_stmts(&fl.else_body, macros);
            }
            Stmt::IfCond(ic) => {
                self.visit_expr(&ic.expr);
                self.visit_stmts(&ic.true_body, macros);
                self.visit_stmts(&ic.false_body, macros);
            }
            Stmt::WithBlock(wb) => {
                for (_, expr) in &wb.assignments {
                    self.visit_expr(expr);
                }
                self.visit_stmts(&wb.body, macros);
            }
            Stmt::Set(set_stmt) => {
                self.assignments.push(set_stmt);
                self.visit_expr(&set_stmt.expr);
            }
            Stmt::SetBlock(sb) => {
                if let Some(filter) = &sb.filter {
                    self.visit_expr(filter);
                }
                self.visit_stmts(&sb.body, macros);
            }
            Stmt::AutoEscape(ae) => {
                self.visit_expr(&ae.enabled);
                self.visit_stmts(&ae.body, macros);
            }
            Stmt::FilterBlock(fb) => {
                self.visit_expr(&fb.filter);
                self.visit_stmts(&fb.body, macros);
            }
            Stmt::Block(b) => self.visit_stmts(&b.body, macros),
            Stmt::Macro(m) => {
                for default in &m.defaults {
                    self.visit_expr(default);
                }
                macros.push(self.macros.len());
                self.macros.push(m);
                self.visit_stmts(&m.body, macros);
                macros.pop();
            }
            // The body of a call block is the anonymous `caller` macro, which
            // Python's detection does not treat as a macro.
            Stmt::CallBlock(cb) => {
                self.visit_call(&cb.call);
                for default in &cb.macro_decl.defaults {
                    self.visit_expr(default);
                }
                self.visit_stmts(&cb.macro_decl.body, macros);
            }
            Stmt::Do(d) => self.visit_call(&d.call),
            _ => {}
        }
    }

    fn visit_call(&mut self, call: &'a Call<'a>) {
        self.calls.push(call);
        self.visit_expr(&call.expr);
        for arg in &call.args {
            self.visit_call_arg(arg);
        }
    }

    fn visit_call_arg(&mut self, arg: &'a CallArg<'a>) {
        match arg {
            CallArg::Pos(expr)
            | CallArg::Kwarg(_, expr)
            | CallArg::PosSplat(expr)
            | CallArg::KwargSplat(expr) => self.visit_expr(expr),
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr<'a>) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Slice(s) => {
                self.visit_expr(&s.expr);
                for bound in [&s.start, &s.stop, &s.step].into_iter().flatten() {
                    self.visit_expr(bound);
                }
            }
            Expr::UnaryOp(u) => self.visit_expr(&u.expr),
            Expr::BinOp(b) => {
                self.visit_expr(&b.left);
                self.visit_expr(&b.right);
            }
            Expr::Compare(c) => {
                self.visit_expr(&c.expr);
                for op in &c.ops {
                    self.visit_expr(&op.expr);
                }
            }
            Expr::IfExpr(i) => {
                self.visit_expr(&i.test_expr);
                self.visit_expr(&i.true_expr);
                if let Some(false_expr) = &i.false_expr {
                    self.visit_expr(false_expr);
                }
            }
            Expr::Filter(f) => {
                if let Some(inner) = &f.expr {
                    self.visit_expr(inner);
                }
                for arg in &f.args {
                    self.visit_call_arg(arg);
                }
            }
            Expr::Test(t) => {
                self.visit_expr(&t.expr);
                for arg in &t.args {
                    self.visit_call_arg(arg);
                }
            }
            Expr::GetAttr(g) => self.visit_expr(&g.expr),
            Expr::GetItem(g) => {
                self.visit_expr(&g.expr);
                self.visit_expr(&g.subscript_expr);
            }
            Expr::Call(c) => self.visit_call(c),
            Expr::List(l) => {
                for item in &l.items {
                    self.visit_expr(item);
                }
            }
            Expr::Map(m) => {
                for item in m.keys.iter().chain(&m.values) {
                    self.visit_expr(item);
                }
            }
        }
    }
}

fn iter_nodes_assign_var_or_elems(nodes: &TemplateNodes<'_>, varname: &str) -> Vec<String> {
    let mut discovered = vec![varname.to_string()];
    let mut seen = HashSet::from([varname.to_string()]);
    let mut related = VecDeque::from([varname.to_string()]);

    while let Some(related_varname) = related.pop_front() {
        for assign in &nodes.assignments {
            let aliases = match (&assign.target, &assign.expr) {
                // `{% set ns = namespace(rest=messages) %}` aliases `ns.rest`.
                (Expr::Var(lhs), Expr::Call(call)) if is_var_access(&call.expr, "namespace") => {
                    call.args
                        .iter()
                        .filter_map(|arg| match arg {
                            CallArg::Kwarg(key, value)
                                if is_var_or_elems_access(value, &related_varname, None) =>
                            {
                                Some(format!("{}.{key}", lhs.id))
                            }
                            _ => None,
                        })
                        .collect()
                }
                (lhs @ (Expr::Var(_) | Expr::GetAttr(_)), rhs)
                    if is_var_or_elems_access(rhs, &related_varname, None) =>
                {
                    attr_dotted_name(lhs).into_iter().collect()
                }
                _ => Vec::new(),
            };

            for alias in aliases {
                if seen.insert(alias.clone()) {
                    discovered.push(alias.clone());
                    if alias != related_varname {
                        related.push_back(alias);
                    }
                }
            }
        }
    }

    discovered
}

fn iter_nodes_assign_messages_item(nodes: &TemplateNodes<'_>) -> Vec<String> {
    let message_varnames = iter_nodes_assign_var_or_elems(nodes, "messages");

    let mut discovered = Vec::new();
    let mut seen = HashSet::new();

    for (loop_ast, _) in &nodes.loops {
        let Expr::Var(target) = &loop_ast.target else {
            continue;
        };

        if message_varnames
            .iter()
            .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, None))
        {
            let target_name = target.id.to_string();
            if seen.insert(target_name.clone()) {
                discovered.push(target_name);
            }
        }
    }

    discovered
}

/// Parameters of a macro that some call binds to `message.content`, by
/// position or by keyword.
fn macro_content_params<'a>(
    macro_decl: &'a Macro<'a>,
    calls: &[&Call<'_>],
    message_varnames: &[String],
) -> HashSet<&'a str> {
    let is_message_content = |expr: &Expr<'_>| {
        message_varnames
            .iter()
            .any(|varname| is_var_or_elems_access(expr, varname, Some("content")))
    };
    let param_names = macro_decl
        .args
        .iter()
        .map(|arg| match arg {
            Expr::Var(v) => Some(v.id),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut bound = HashSet::new();
    for call in calls {
        if !is_var_access(&call.expr, macro_decl.name) {
            continue;
        }
        let positional = call.args.iter().filter_map(|arg| match arg {
            CallArg::Pos(expr) => Some(expr),
            _ => None,
        });
        for (param, arg) in param_names.iter().zip(positional) {
            if let Some(param) = param
                && is_message_content(arg)
            {
                bound.insert(*param);
            }
        }
        for arg in &call.args {
            if let CallArg::Kwarg(key, value) = arg
                && let Some(param) = param_names.iter().flatten().find(|param| *param == key)
                && is_message_content(value)
            {
                bound.insert(*param);
            }
        }
    }
    bound
}

fn has_content_item_loop(root: &Stmt<'_>) -> bool {
    let nodes = TemplateNodes::collect(root);
    let message_varnames = iter_nodes_assign_messages_item(&nodes);
    let content_params = nodes
        .macros
        .iter()
        .map(|m| macro_content_params(m, &nodes.calls, &message_varnames))
        .collect::<Vec<_>>();

    nodes.loops.iter().any(|(loop_ast, macros)| {
        if !matches!(loop_ast.target, Expr::Var(_)) {
            return false;
        }
        // `{% for part in message.content %}`, anywhere.
        if message_varnames
            .iter()
            .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, Some("content")))
        {
            return true;
        }
        let Expr::Var(iter) = &loop_ast.iter else {
            return false;
        };
        if macros.is_empty() {
            // `{% for part in content %}` outside any macro.
            iter.id == "content"
        } else {
            // Inside a macro, only a loop over a parameter bound to
            // `message.content` counts; the nearest enclosing macro with bound
            // parameters decides, like Python's detection.
            macros
                .iter()
                .rev()
                .map(|&index| &content_params[index])
                .find(|params| !params.is_empty())
                .is_some_and(|params| params.contains(iter.id))
        }
    })
}

/// Detect the content format expected by a Jinja2 chat template based on AST
/// analysis.
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return ChatTemplateContentFormat::String,
    };

    if has_content_item_loop(&ast) {
        ChatTemplateContentFormat::OpenAi
    } else {
        ChatTemplateContentFormat::String
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use expect_test::expect;

    use super::{ChatTemplateContentFormat, detect_chat_template_content_format};

    fn detect(template: &str) -> ChatTemplateContentFormat {
        detect_chat_template_content_format(template)
    }

    fn vllm_examples_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/templates/vllm_examples")
            .canonicalize()
            .expect("vLLM example template directory should exist locally")
    }

    fn read_vllm_example(relative_path: &str) -> String {
        fs::read_to_string(vllm_examples_dir().join(relative_path))
            .unwrap_or_else(|_| panic!("failed to read vLLM example template: {relative_path}"))
    }

    fn iter_vllm_example_template_paths() -> impl Iterator<Item = PathBuf> {
        let mut paths = fs::read_dir(vllm_examples_dir())
            .expect("failed to read vLLM example template directory")
            .map(|entry| entry.expect("failed to read vLLM example template dir entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jinja"))
            .collect::<Vec<_>>();
        paths.sort();
        paths.into_iter()
    }

    #[test]
    fn detects_string_template_without_content_loop() {
        assert_eq!(
            detect("{% for message in messages %}{{ message.content }}{% endfor %}"),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn detects_openai_template_with_direct_content_loop() {
        assert_eq!(
            detect(
                "{% for message in messages %}{% for content in message['content'] %}{{ content }}{% endfor %}{% endfor %}"
            ),
            ChatTemplateContentFormat::OpenAi
        );
    }

    #[test]
    fn detects_openai_template_with_content_parameter_loop() {
        assert_eq!(
            detect(
                "{% macro render(content) %}{% for item in content %}{{ item }}{% endfor %}{% endmacro %}{% for message in messages %}{{ render(message.content) }}{% endfor %}"
            ),
            ChatTemplateContentFormat::OpenAi
        );
    }

    /// Cases from Python's `test_detect_content_format` (vllm-project/vllm#53824
    /// and vllm-project/vllm#62622), plus a macro reached through a namespace
    /// alias.
    #[test]
    fn detects_content_loops_through_macro_parameters_like_python() {
        let cases = [
            (
                "positional parameter bound to message content",
                "{% macro render_message(role, message_content) %}\
                 {% for item in message_content %}{{ item['type'] }}{% endfor %}\
                 {% endmacro %}\
                 {% for message in messages %}\
                 {{ render_message(message['role'], message['content']) }}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "keyword parameter bound to message content",
                "{% macro render_message(role, message_content) %}\
                 {% for item in message_content %}{{ item['type'] }}{% endfor %}\
                 {% endmacro %}\
                 {% for message in messages %}\
                 {{ render_message(role=message['role'], message_content=message['content']) }}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "direct loop over message content",
                "{% for message in messages %}\
                 {% for item in message['content'] %}{{ item['type'] }}{% endfor %}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "macro parameter named content but bound to a literal",
                "{% macro render_message(content) %}\
                 {% for item in content %}{{ item['type'] }}{% endfor %}\
                 {% endmacro %}\
                 {% for message in messages %}\
                 {{ render_message(content=['text only']) }}\
                 {% endfor %}",
                ChatTemplateContentFormat::String,
            ),
            (
                "no content loop",
                "{% for message in messages %}{{ message['content'] }}{% endfor %}",
                ChatTemplateContentFormat::String,
            ),
            (
                "messages aliased by a namespace argument",
                "{% set ns = namespace(rest=messages) %}\
                 {% for message in ns.rest %}\
                 {% for item in message['content'] %}{{ item['type'] }}{% endfor %}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "messages aliased by a namespace attribute assignment",
                "{% set ns = namespace(rest=[]) %}\
                 {% set ns.rest = messages %}\
                 {% for message in ns.rest %}\
                 {% for item in message['content'] %}{{ item['type'] }}{% endfor %}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "macro called from a call block's caller defaults",
                "{% macro render(content) %}{% for p in content %}{{ p.text }}{% endfor %}{% endmacro %}\
                 {% macro wrap() %}{{ caller() }}{% endmacro %}\
                 {% for message in messages %}\
                 {% call(text=render(message.content)) wrap() %}{{ text }}{% endcall %}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "macro called from an autoescape condition",
                "{% macro render(content) %}{% for p in content %}{{ p.text }}{% endfor %}{% endmacro %}\
                 {% for message in messages %}\
                 {% autoescape render(message.content) == 'hello' %}{{ message.role }}{% endautoescape %}\
                 {% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
            (
                "macro parameter bound through a namespace alias",
                "{% macro render_content(content) %}\
                 {% for part in content %}{{ part['type'] }}{% endfor %}\
                 {% endmacro %}\
                 {% set ns = namespace(rest=messages[1:]) %}\
                 {% for message in ns.rest %}{{ render_content(message['content']) }}{% endfor %}",
                ChatTemplateContentFormat::OpenAi,
            ),
        ];

        for (name, template, expected) in cases {
            assert_eq!(detect(template), expected, "{name}");
        }
    }

    #[test]
    fn detects_openai_template_with_messages_alias() {
        assert_eq!(
            detect(
                "{% set msgs = messages %}{% for message in msgs %}{% for content in message.content %}{{ content }}{% endfor %}{% endfor %}"
            ),
            ChatTemplateContentFormat::OpenAi
        );
    }

    #[test]
    fn does_not_detect_content_alias_loop_as_openai() {
        assert_eq!(
            detect(
                "{% for message in messages %}{% set parts = message.content %}{% for item in parts %}{{ item }}{% endfor %}{% endfor %}"
            ),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn does_not_treat_length_or_index_access_as_openai() {
        assert_eq!(
            detect("{% for message in messages %}{{ message.content|length }}{% endfor %}"),
            ChatTemplateContentFormat::String
        );
        assert_eq!(
            detect("{% for message in messages %}{{ message.content[0] }}{% endfor %}"),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn matches_vllm_example_template_formats() {
        let snapshot = iter_vllm_example_template_paths()
            .map(|path| {
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .expect("template file name should be valid UTF-8");
                let template = read_vllm_example(file_name);
                let format = detect(&template);
                format!("{file_name:50} => {format:?}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        expect![[r#"
            template_alpaca.jinja                              => String
            template_chatglm.jinja                             => String
            template_chatglm2.jinja                            => String
            template_chatml.jinja                              => String
            template_falcon.jinja                              => String
            template_falcon_180b.jinja                         => String
            template_inkbot.jinja                              => String
            template_teleflm.jinja                             => String
            tool_chat_template_deepseekr1.jinja                => String
            tool_chat_template_deepseekv3.jinja                => String
            tool_chat_template_deepseekv31.jinja               => String
            tool_chat_template_functiongemma.jinja             => String
            tool_chat_template_gemma3_pythonic.jinja           => OpenAi
            tool_chat_template_gemma4.jinja                    => OpenAi
            tool_chat_template_glm4.jinja                      => String
            tool_chat_template_granite.jinja                   => String
            tool_chat_template_granite_20b_fc.jinja            => String
            tool_chat_template_hermes.jinja                    => String
            tool_chat_template_hunyuan_a13b.jinja              => String
            tool_chat_template_internlm2_tool.jinja            => String
            tool_chat_template_llama3.1_json.jinja             => OpenAi
            tool_chat_template_llama3.2_json.jinja             => OpenAi
            tool_chat_template_llama3.2_pythonic.jinja         => String
            tool_chat_template_llama4_json.jinja               => OpenAi
            tool_chat_template_llama4_pythonic.jinja           => OpenAi
            tool_chat_template_mistral.jinja                   => String
            tool_chat_template_mistral3.jinja                  => OpenAi
            tool_chat_template_mistral_parallel.jinja          => String
            tool_chat_template_phi4_mini.jinja                 => String
            tool_chat_template_qwen3coder.jinja                => String
            tool_chat_template_toolace.jinja                   => String
            tool_chat_template_xlam_llama.jinja                => String
            tool_chat_template_xlam_qwen.jinja                 => String"#]]
        .assert_eq(&snapshot);
    }
}
