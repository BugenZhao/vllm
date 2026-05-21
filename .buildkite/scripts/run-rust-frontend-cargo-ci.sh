#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" != "style-clippy" && "$MODE" != "test" ]]; then
  echo "Usage: $0 {style-clippy|test}" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"

log_section() {
  echo "--- $*"
}

install_system_deps() {
  local missing=0
  for cmd in protoc cc make perl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing=1
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    return
  fi

  log_section "Installing system dependencies"
  sudo dnf install -y \
    ca-certificates \
    curl \
    gcc \
    make \
    perl \
    protobuf-compiler
}

rust_toolchain() {
  awk -F '"' '/channel[[:space:]]*=/ { print $2; exit }' rust-toolchain.toml
}

install_rust_toolchain() {
  log_section "Installing Rust toolchain"
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain none
  fi

  local toolchain
  toolchain="$(rust_toolchain)"
  rustup toolchain install "$toolchain" --profile minimal --component rustfmt,clippy
  rustup component add --toolchain "$toolchain" rustfmt clippy
}

install_cargo_binstall() {
  if command -v cargo-binstall >/dev/null 2>&1; then
    return
  fi

  log_section "Installing cargo-binstall"
  curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh \
    | bash
}

install_cargo_sort() {
  if command -v cargo-sort >/dev/null 2>&1; then
    return
  fi

  log_section "Installing cargo-sort"
  install_cargo_binstall
  cargo binstall --no-confirm cargo-sort
}

install_cargo_nextest() {
  if command -v cargo-nextest >/dev/null 2>&1; then
    return
  fi

  log_section "Installing cargo-nextest"
  install_cargo_binstall
  cargo binstall --no-confirm --secure cargo-nextest
}

run_style_clippy() {
  install_cargo_sort

  log_section "Checking Rust formatting"
  cargo fmt --manifest-path rust/Cargo.toml --all -- --check

  log_section "Checking Cargo.toml ordering"
  cargo sort --workspace --check rust

  log_section "Running clippy"
  cargo clippy \
    --manifest-path rust/Cargo.toml \
    --workspace \
    --all-targets \
    --all-features \
    --locked \
    -- \
    -D warnings
}

run_tests() {
  install_cargo_nextest

  log_section "Running cargo nextest"
  cargo nextest run \
    --manifest-path rust/Cargo.toml \
    --workspace \
    --all-features \
    --locked \
    --no-fail-fast
}

install_system_deps
install_rust_toolchain

case "$MODE" in
  style-clippy)
    run_style_clippy
    ;;
  test)
    run_tests
    ;;
esac
