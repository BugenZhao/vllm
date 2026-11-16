// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Where request media may be fetched from, shared by the OpenAI frontend and
//! the render server.

use std::path::PathBuf;

use clap::Args;
use serde::Deserialize;
use vllm_chat::multimodal::MediaFetchPolicy;

#[derive(Clone, Debug, Default, Args, PartialEq, Eq, Deserialize)]
pub struct MediaFetchArgs {
    /// Allowing API requests to read local images or videos from directories
    /// specified by the server file system. This is a security risk. Should
    /// only be enabled in trusted environments.
    #[arg(long, value_name = "PATH")]
    #[serde(default)]
    pub allowed_local_media_path: Option<PathBuf>,

    /// If set, only media URLs that belong to this domain can be used for
    /// multi-modal inputs. Listed hosts are fetched by name even when they
    /// resolve to loopback or private addresses; without the list only
    /// public addresses are fetched.
    #[arg(long, num_args = 1.., value_name = "DOMAIN")]
    #[serde(default)]
    pub allowed_media_domains: Option<Vec<String>>,
}

impl MediaFetchArgs {
    /// The `--allowed-local-media-path` / `--allowed-media-domains` policy;
    /// an empty path disables local media, as in Python.
    pub fn policy(&self) -> MediaFetchPolicy {
        MediaFetchPolicy {
            allowed_local_media_path: self
                .allowed_local_media_path
                .clone()
                .filter(|path| !path.as_os_str().is_empty()),
            allowed_media_domains: self.allowed_media_domains.clone(),
        }
    }
}
