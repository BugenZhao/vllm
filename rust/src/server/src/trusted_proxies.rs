// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::fmt;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::str::FromStr;

use axum::http::HeaderMap;
use educe::Educe;
use ipnet::IpNet;
use serde_with::SerializeDisplay;

/// Uvicorn's `FORWARDED_ALLOW_IPS` trust list: the peers whose
/// `X-Forwarded-For` header names the client. Like Uvicorn, it never fails to
/// parse: an entry that is neither an IP nor a network is kept as a literal.
#[derive(Educe, Clone, PartialEq, Eq, SerializeDisplay)]
#[educe(Debug)]
pub struct TrustedProxies {
    /// The configured value, as reported by `/server_info`.
    source: String,
    #[educe(Debug(ignore))]
    networks: Vec<IpNet>,
    #[educe(Debug(ignore))]
    literals: BTreeSet<String>,
    #[educe(Debug(ignore))]
    always_trust: bool,
}

impl Default for TrustedProxies {
    /// Uvicorn's default when `FORWARDED_ALLOW_IPS` is unset.
    fn default() -> Self {
        Self::new("127.0.0.1,::1")
    }
}

impl FromStr for TrustedProxies {
    type Err = Infallible;

    fn from_str(value: &str) -> Result<Self, Infallible> {
        Ok(Self::new(value))
    }
}

impl fmt::Display for TrustedProxies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source)
    }
}

impl TrustedProxies {
    pub fn new(value: &str) -> Self {
        let mut networks = Vec::new();
        let mut literals = BTreeSet::new();
        for host in value.split(',').map(str::trim).filter(|host| !host.is_empty()) {
            if let Ok(ip) = host.parse::<IpAddr>() {
                networks.push(IpNet::from(ip));
            } else if let Some((address, _)) = host.split_once('/')
                && address.parse::<IpAddr>().is_ok()
                && let Ok(network) = host.parse::<IpNet>()
                && network.addr() == network.network()
            {
                networks.push(network);
            } else {
                literals.insert(host.to_owned());
            }
        }
        Self {
            source: value.to_owned(),
            networks,
            literals,
            always_trust: value == "*",
        }
    }

    /// The client address to log: the forwarded client when the transport
    /// peer is trusted and the header names one, otherwise the peer itself
    /// (`None` for Unix sockets).
    pub(crate) fn client_addr(
        &self,
        peer: Option<SocketAddr>,
        headers: &HeaderMap,
    ) -> Option<SocketAddr> {
        let trusted_peer =
            self.always_trust || peer.is_some_and(|peer| self.contains_ip(peer.ip()));
        if trusted_peer && let Some(client) = self.forwarded_client(headers) {
            return Some(client);
        }
        peer
    }

    fn contains_ip(&self, ip: IpAddr) -> bool {
        self.networks.iter().any(|network| network.contains(&ip))
    }

    fn contains_host(&self, host: &[u8]) -> bool {
        self.always_trust
            || std::str::from_utf8(host).is_ok_and(|host| match parse_hop(host) {
                Some(address) => self.contains_ip(address.ip()),
                None => self.literals.contains(host),
            })
    }

    fn forwarded_client(&self, headers: &HeaderMap) -> Option<SocketAddr> {
        let mut hosts = headers
            .get_all("x-forwarded-for")
            .iter()
            .flat_map(|value| value.as_bytes().split(|byte| *byte == b','))
            .map(<[u8]>::trim_ascii);
        let first = hosts.next()?;
        let host = hosts.rfind(|host| !self.contains_host(host)).unwrap_or(first);
        // Do not skip an invalid hop: that could expose a spoofed address
        // farther left in the chain.
        parse_hop(std::str::from_utf8(host).ok()?)
    }
}

fn parse_hop(host: &str) -> Option<SocketAddr> {
    if let Ok(address) = host.parse::<SocketAddr>() {
        return Some(address);
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(SocketAddr::new(ip, 0));
    }
    let ip = host.strip_prefix('[')?.strip_suffix(']')?.parse::<Ipv6Addr>().ok()?;
    Some(SocketAddr::new(ip.into(), 0))
}
