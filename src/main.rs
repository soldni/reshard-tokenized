use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use reshard_tokenized::{MergeConfig, merge_files};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    about = "Merge .npy and .csv.gz files from a directory tree into one or more shards",
    version
)]
struct Cli {
    #[arg(long)]
    input_path: PathBuf,
    #[arg(long)]
    num_files: usize,
    #[arg(long)]
    output_path: PathBuf,
}

fn main() -> ExitCode {
    init_tracing();

    let args = Cli::parse();
    let config = MergeConfig {
        input_path: args.input_path,
        num_files: args.num_files,
        output_path: args.output_path,
    };

    match merge_files(&config) {
        Ok(report) => {
            info!(
                npy_inputs = report.npy_inputs,
                csv_gz_inputs = report.csv_gz_inputs,
                npy_outputs = report.npy_outputs.len(),
                csv_gz_outputs = report.csv_gz_outputs.len(),
                "merge succeeded",
            );
            ExitCode::SUCCESS
        }
        Err(error_value) => {
            error!("{error_value}");
            eprintln!("error: {error_value}");
            ExitCode::FAILURE
        }
    }
}

fn init_tracing() {
    let default_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(default_filter)
        .with_target(false)
        .compact()
        .init();
}
