//! `bonsai-ctl` — the Bonsai coordinator.
//!
//! ```text
//! bonsai-ctl --config config/bonsai-ctl.toml [--check] [--listen ADDR]
//! ```
//! See docs/CONTROL_PLANE.md.

use std::{path::PathBuf, process::ExitCode};

use bonsai_control::{Coordinator, config::CtlConfig, server};

const USAGE: &str = "usage: bonsai-ctl --config <file.toml> [--check] [--listen <addr:port>]

  --config <file>   coordinator configuration (template: config/bonsai-ctl.example.toml)
  --check           validate the config, print wire sizes per precision, and exit
  --listen <addr>   override [server].listen";

struct Args {
    config: PathBuf,
    check: bool,
    listen: Option<std::net::SocketAddr>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut config, mut check, mut listen) = (None, false, None);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--config" | "-c" => config = Some(PathBuf::from(it.next().ok_or("--config needs a path")?)),
            "--check" => check = true,
            "--listen" => {
                let v = it.next().ok_or("--listen needs addr:port")?;
                listen = Some(v.parse().map_err(|e| format!("--listen {v}: {e}"))?);
            }
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args { config: config.ok_or("--config is required")?, check, listen })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let cfg = match CtlConfig::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}: {e}", args.config.display());
            return ExitCode::FAILURE;
        }
    };
    let resolved = match cfg.resolve() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}: {e}", args.config.display());
            return ExitCode::FAILURE;
        }
    };
    if args.check {
        match cfg.summary() {
            Ok(s) => {
                println!("{} OK\n\n{s}", args.config.display());
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let sink = match cfg.open_sink() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: opening metrics sink: {e}");
            return ExitCode::FAILURE;
        }
    };
    let listen = args.listen.unwrap_or(resolved.listen);
    let coordinator = Coordinator::new(resolved.coordinator, resolved.placement, resolved.codec_policy, sink);
    let _server_cfg = server::ServerConfig { listen, tick: resolved.tick };
    let _ = coordinator;
    // The TCP front end (server::serve) lands in milestone 1.
    eprintln!("bonsai-ctl: config OK, but the TCP server is not implemented yet (milestone 1). Use --check.");
    ExitCode::FAILURE
}
