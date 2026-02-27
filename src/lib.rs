use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use thiserror::Error;
use tracing::{debug, info};
use walkdir::WalkDir;

const IO_BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct MergeConfig {
    pub input_path: PathBuf,
    pub num_files: usize,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeReport {
    pub npy_inputs: usize,
    pub csv_gz_inputs: usize,
    pub npy_outputs: Vec<PathBuf>,
    pub csv_gz_outputs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFiles {
    pub npy_files: Vec<PathBuf>,
    pub csv_gz_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputPlan {
    pub npy_outputs: Vec<PathBuf>,
    pub csv_gz_outputs: Vec<PathBuf>,
}

#[derive(Debug, Error)]
pub enum MergeError {
    #[error("`--num-files` must be at least 1, got {0}")]
    InvalidNumFiles(usize),
    #[error("failed to read metadata for path {path}: {source}")]
    ReadPathMetadata {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("input path is not a directory: {0}")]
    InputPathNotDirectory(PathBuf),
    #[error("failed to walk input directory {path}: {source}")]
    WalkInputDirectory {
        path: PathBuf,
        #[source]
        source: walkdir::Error,
    },
    #[error("output path exists and is not a directory: {0}")]
    OutputPathNotDirectory(PathBuf),
    #[error("failed to create directory {path}: {source}")]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "shard count mismatch for {file_type}: {input_shards} input shard sets but {output_paths} output paths"
    )]
    ShardConfigurationMismatch {
        file_type: &'static str,
        input_shards: usize,
        output_paths: usize,
    },
    #[error("failed to open source file {path}: {source}")]
    OpenSourceFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to create destination file {path}: {source}")]
    CreateDestinationFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to copy data from {source_path} to {destination_path}: {source}")]
    CopyFileData {
        source_path: PathBuf,
        destination_path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to flush destination file {path}: {source}")]
    FlushDestinationFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

pub fn merge_files(config: &MergeConfig) -> Result<MergeReport, MergeError> {
    validate_config(config)?;
    let discovered = discover_files(&config.input_path)?;
    let plan = build_output_plan(&config.output_path, config.num_files)?;
    create_output_directories(config, &plan)?;

    info!(
        input_path = %config.input_path.display(),
        output_path = %config.output_path.display(),
        num_files = config.num_files,
        "starting merge",
    );
    info!(
        npy_files = discovered.npy_files.len(),
        csv_gz_files = discovered.csv_gz_files.len(),
        "discovered source files",
    );
    debug!(?plan, "resolved output files");

    let npy_shards = shard_paths(&discovered.npy_files, config.num_files);
    let csv_gz_shards = shard_paths(&discovered.csv_gz_files, config.num_files);
    let progress =
        build_progress_bar((discovered.npy_files.len() + discovered.csv_gz_files.len()) as u64);

    let (npy_result, csv_result) = rayon::join(
        || merge_npy_shards(&npy_shards, &plan.npy_outputs, &progress),
        || merge_csv_gz_shards(&csv_gz_shards, &plan.csv_gz_outputs, &progress),
    );
    npy_result?;
    csv_result?;

    progress.finish_with_message("merge complete");
    info!("merge complete");

    Ok(MergeReport {
        npy_inputs: discovered.npy_files.len(),
        csv_gz_inputs: discovered.csv_gz_files.len(),
        npy_outputs: plan.npy_outputs,
        csv_gz_outputs: plan.csv_gz_outputs,
    })
}

fn validate_config(config: &MergeConfig) -> Result<(), MergeError> {
    if config.num_files == 0 {
        return Err(MergeError::InvalidNumFiles(config.num_files));
    }

    let metadata =
        fs::metadata(&config.input_path).map_err(|source| MergeError::ReadPathMetadata {
            path: config.input_path.clone(),
            source,
        })?;
    if !metadata.is_dir() {
        return Err(MergeError::InputPathNotDirectory(config.input_path.clone()));
    }

    Ok(())
}

pub fn discover_files(input_path: &Path) -> Result<DiscoveredFiles, MergeError> {
    let mut npy_files = Vec::new();
    let mut csv_gz_files = Vec::new();

    for entry in WalkDir::new(input_path) {
        let entry = entry.map_err(|source| MergeError::WalkInputDirectory {
            path: input_path.to_path_buf(),
            source,
        })?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.into_path();
        if is_npy_file(&path) {
            npy_files.push(path);
        } else if is_csv_gz_file(&path) {
            csv_gz_files.push(path);
        }
    }

    npy_files.sort();
    csv_gz_files.sort();

    Ok(DiscoveredFiles {
        npy_files,
        csv_gz_files,
    })
}

pub fn build_output_plan(output_path: &Path, num_files: usize) -> Result<OutputPlan, MergeError> {
    if num_files == 0 {
        return Err(MergeError::InvalidNumFiles(num_files));
    }

    if num_files == 1 {
        return Ok(OutputPlan {
            npy_outputs: vec![append_extension(output_path, "npy")],
            csv_gz_outputs: vec![append_extension(output_path, "csv.gz")],
        });
    }

    let npy_outputs = (0..num_files)
        .map(|index| output_path.join(format!("{index:08}.npy")))
        .collect();
    let csv_gz_outputs = (0..num_files)
        .map(|index| output_path.join(format!("{index:08}.csv.gz")))
        .collect();

    Ok(OutputPlan {
        npy_outputs,
        csv_gz_outputs,
    })
}

fn create_output_directories(config: &MergeConfig, plan: &OutputPlan) -> Result<(), MergeError> {
    if config.num_files == 1 {
        for output in plan.npy_outputs.iter().chain(plan.csv_gz_outputs.iter()) {
            create_parent_dir(output)?;
        }
        return Ok(());
    }

    if config.output_path.exists() {
        let metadata =
            fs::metadata(&config.output_path).map_err(|source| MergeError::ReadPathMetadata {
                path: config.output_path.clone(),
                source,
            })?;
        if !metadata.is_dir() {
            return Err(MergeError::OutputPathNotDirectory(
                config.output_path.clone(),
            ));
        }
        return Ok(());
    }

    fs::create_dir_all(&config.output_path).map_err(|source| MergeError::CreateDirectory {
        path: config.output_path.clone(),
        source,
    })?;
    Ok(())
}

fn create_parent_dir(path: &Path) -> Result<(), MergeError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    fs::create_dir_all(parent).map_err(|source| MergeError::CreateDirectory {
        path: parent.to_path_buf(),
        source,
    })?;
    Ok(())
}

pub fn shard_paths(paths: &[PathBuf], num_shards: usize) -> Vec<Vec<PathBuf>> {
    let mut shards = vec![Vec::new(); num_shards];
    for (index, path) in paths.iter().enumerate() {
        shards[index % num_shards].push(path.clone());
    }
    shards
}

pub fn merge_npy_shards(
    input_shards: &[Vec<PathBuf>],
    output_paths: &[PathBuf],
    progress: &ProgressBar,
) -> Result<(), MergeError> {
    if input_shards.len() != output_paths.len() {
        return Err(MergeError::ShardConfigurationMismatch {
            file_type: "npy",
            input_shards: input_shards.len(),
            output_paths: output_paths.len(),
        });
    }

    (0..output_paths.len())
        .into_par_iter()
        .try_for_each(|index| {
            let shard_inputs = &input_shards[index];
            let shard_output = &output_paths[index];
            let shard_progress = progress.clone();
            merge_single_npy_shard(shard_inputs, shard_output, &shard_progress)
        })
}

pub fn merge_csv_gz_shards(
    input_shards: &[Vec<PathBuf>],
    output_paths: &[PathBuf],
    progress: &ProgressBar,
) -> Result<(), MergeError> {
    if input_shards.len() != output_paths.len() {
        return Err(MergeError::ShardConfigurationMismatch {
            file_type: "csv.gz",
            input_shards: input_shards.len(),
            output_paths: output_paths.len(),
        });
    }

    (0..output_paths.len())
        .into_par_iter()
        .try_for_each(|index| {
            let shard_inputs = &input_shards[index];
            let shard_output = &output_paths[index];
            let shard_progress = progress.clone();
            merge_single_csv_gz_shard(shard_inputs, shard_output, &shard_progress)
        })
}

fn merge_single_npy_shard(
    input_paths: &[PathBuf],
    output_path: &Path,
    progress: &ProgressBar,
) -> Result<(), MergeError> {
    let output_file =
        File::create(output_path).map_err(|source| MergeError::CreateDestinationFile {
            path: output_path.to_path_buf(),
            source,
        })?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, output_file);
    let mut buffer = vec![0_u8; IO_BUFFER_SIZE];

    for input_path in input_paths {
        let input_file = File::open(input_path).map_err(|source| MergeError::OpenSourceFile {
            path: input_path.clone(),
            source,
        })?;
        let mut reader = BufReader::with_capacity(IO_BUFFER_SIZE, input_file);
        copy_reader_to_writer(&mut reader, &mut writer, &mut buffer).map_err(|source| {
            MergeError::CopyFileData {
                source_path: input_path.clone(),
                destination_path: output_path.to_path_buf(),
                source,
            }
        })?;
        progress.inc(1);
    }

    writer
        .flush()
        .map_err(|source| MergeError::FlushDestinationFile {
            path: output_path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn merge_single_csv_gz_shard(
    input_paths: &[PathBuf],
    output_path: &Path,
    progress: &ProgressBar,
) -> Result<(), MergeError> {
    let output_file =
        File::create(output_path).map_err(|source| MergeError::CreateDestinationFile {
            path: output_path.to_path_buf(),
            source,
        })?;
    let writer = BufWriter::with_capacity(IO_BUFFER_SIZE, output_file);
    let mut encoder = GzEncoder::new(writer, Compression::default());
    let mut buffer = vec![0_u8; IO_BUFFER_SIZE];

    for input_path in input_paths {
        let input_file = File::open(input_path).map_err(|source| MergeError::OpenSourceFile {
            path: input_path.clone(),
            source,
        })?;
        let reader = BufReader::with_capacity(IO_BUFFER_SIZE, input_file);
        let mut decoder = MultiGzDecoder::new(reader);

        copy_reader_to_writer(&mut decoder, &mut encoder, &mut buffer).map_err(|source| {
            MergeError::CopyFileData {
                source_path: input_path.clone(),
                destination_path: output_path.to_path_buf(),
                source,
            }
        })?;
        progress.inc(1);
    }

    let mut writer = encoder
        .finish()
        .map_err(|source| MergeError::FlushDestinationFile {
            path: output_path.to_path_buf(),
            source,
        })?;
    writer
        .flush()
        .map_err(|source| MergeError::FlushDestinationFile {
            path: output_path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn copy_reader_to_writer<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    buffer: &mut [u8],
) -> io::Result<u64> {
    let mut total_written = 0_u64;
    loop {
        let read_bytes = reader.read(buffer)?;
        if read_bytes == 0 {
            break;
        }
        writer.write_all(&buffer[..read_bytes])?;
        total_written += read_bytes as u64;
    }
    Ok(total_written)
}

fn is_npy_file(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "npy")
}

fn is_csv_gz_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|file_name| file_name.to_str())
        .is_some_and(|file_name| file_name.ends_with(".csv.gz"))
}

fn append_extension(path: &Path, extension: &str) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        normalized.push(component.as_os_str());
    }
    if normalized.as_os_str().is_empty() {
        normalized = path.to_path_buf();
    }

    let mut file_name = OsString::from(normalized.as_os_str());
    file_name.push(".");
    file_name.push(extension);
    PathBuf::from(file_name)
}

fn build_progress_bar(total_files: u64) -> ProgressBar {
    if cfg!(test) {
        return ProgressBar::hidden();
    }

    let progress = ProgressBar::new(total_files);
    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta}) {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-");
    progress.set_style(style);
    progress.set_message("merging");
    progress
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::tempdir;

    fn write_gzip(path: &Path, content: &str) {
        let file = File::create(path).expect("create gzip input");
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder
            .write_all(content.as_bytes())
            .expect("write gzip content");
        encoder.finish().expect("finish gzip input");
    }

    fn read_gzip(path: &Path) -> String {
        let file = File::open(path).expect("open gzip output");
        let reader = BufReader::new(file);
        let mut decoder = MultiGzDecoder::new(reader);
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .expect("read gzip output");
        content
    }

    #[test]
    fn discovers_files_recursively_and_ignores_others() {
        let temp = tempdir().expect("create tempdir");
        let nested = temp.path().join("nested").join("inner");
        fs::create_dir_all(&nested).expect("create nested dirs");

        fs::write(temp.path().join("b.npy"), [1_u8, 2_u8]).expect("write npy");
        fs::write(nested.join("a.npy"), [3_u8]).expect("write nested npy");
        fs::write(temp.path().join("skip.txt"), "ignore").expect("write skip file");
        write_gzip(&nested.join("z.csv.gz"), "zeta\n");
        write_gzip(&temp.path().join("m.csv.gz"), "mu\n");

        let discovered = discover_files(temp.path()).expect("discover files");

        assert_eq!(discovered.npy_files.len(), 2);
        assert_eq!(discovered.csv_gz_files.len(), 2);
        assert!(discovered.npy_files[0] < discovered.npy_files[1]);
        assert!(discovered.csv_gz_files[0] < discovered.csv_gz_files[1]);
    }

    #[test]
    fn builds_single_output_plan() {
        let output = PathBuf::from("/tmp/output/base");
        let plan = build_output_plan(&output, 1).expect("build plan");
        assert_eq!(
            plan.npy_outputs,
            vec![PathBuf::from("/tmp/output/base.npy")]
        );
        assert_eq!(
            plan.csv_gz_outputs,
            vec![PathBuf::from("/tmp/output/base.csv.gz")]
        );
    }

    #[test]
    fn builds_sharded_output_plan() {
        let output = PathBuf::from("/tmp/output/shards");
        let plan = build_output_plan(&output, 3).expect("build plan");
        assert_eq!(
            plan.npy_outputs,
            vec![
                PathBuf::from("/tmp/output/shards/00000000.npy"),
                PathBuf::from("/tmp/output/shards/00000001.npy"),
                PathBuf::from("/tmp/output/shards/00000002.npy"),
            ]
        );
        assert_eq!(
            plan.csv_gz_outputs,
            vec![
                PathBuf::from("/tmp/output/shards/00000000.csv.gz"),
                PathBuf::from("/tmp/output/shards/00000001.csv.gz"),
                PathBuf::from("/tmp/output/shards/00000002.csv.gz"),
            ]
        );
    }

    #[test]
    fn builds_single_output_plan_with_trailing_separator() {
        let output = PathBuf::from("/tmp/output/base/");
        let plan = build_output_plan(&output, 1).expect("build plan");
        assert_eq!(
            plan.npy_outputs,
            vec![PathBuf::from("/tmp/output/base.npy")]
        );
        assert_eq!(
            plan.csv_gz_outputs,
            vec![PathBuf::from("/tmp/output/base.csv.gz")]
        );
    }

    #[test]
    fn shards_paths_round_robin() {
        let paths = vec![
            PathBuf::from("a.npy"),
            PathBuf::from("b.npy"),
            PathBuf::from("c.npy"),
            PathBuf::from("d.npy"),
            PathBuf::from("e.npy"),
        ];

        let shards = shard_paths(&paths, 2);
        assert_eq!(
            shards[0],
            vec![
                PathBuf::from("a.npy"),
                PathBuf::from("c.npy"),
                PathBuf::from("e.npy")
            ]
        );
        assert_eq!(
            shards[1],
            vec![PathBuf::from("b.npy"), PathBuf::from("d.npy")]
        );
    }

    #[test]
    fn merges_npy_shards_by_byte_concatenation() {
        let temp = tempdir().expect("create tempdir");
        let inputs = temp.path().join("inputs");
        let outputs = temp.path().join("outputs");
        fs::create_dir_all(&inputs).expect("create input dir");
        fs::create_dir_all(&outputs).expect("create output dir");

        let a = inputs.join("a.npy");
        let b = inputs.join("b.npy");
        let c = inputs.join("c.npy");
        fs::write(&a, [1_u8, 2_u8]).expect("write a");
        fs::write(&b, [3_u8]).expect("write b");
        fs::write(&c, [4_u8, 5_u8]).expect("write c");

        let shards = vec![vec![a.clone(), c.clone()], vec![b.clone()]];
        let out0 = outputs.join("00000000.npy");
        let out1 = outputs.join("00000001.npy");
        let progress = ProgressBar::hidden();

        merge_npy_shards(&shards, &[out0.clone(), out1.clone()], &progress)
            .expect("merge npy shards");

        assert_eq!(
            fs::read(out0).expect("read out0"),
            vec![1_u8, 2_u8, 4_u8, 5_u8]
        );
        assert_eq!(fs::read(out1).expect("read out1"), vec![3_u8]);
    }

    #[test]
    fn merges_csv_gz_shards_by_decompress_and_recompress() {
        let temp = tempdir().expect("create tempdir");
        let inputs = temp.path().join("inputs");
        let outputs = temp.path().join("outputs");
        fs::create_dir_all(&inputs).expect("create input dir");
        fs::create_dir_all(&outputs).expect("create output dir");

        let first = inputs.join("a.csv.gz");
        let second = inputs.join("b.csv.gz");
        let third = inputs.join("c.csv.gz");
        write_gzip(&first, "a,1\n");
        write_gzip(&second, "b,2\n");
        write_gzip(&third, "c,3\n");

        let shards = vec![vec![first.clone(), third.clone()], vec![second.clone()]];
        let out0 = outputs.join("00000000.csv.gz");
        let out1 = outputs.join("00000001.csv.gz");
        let progress = ProgressBar::hidden();

        merge_csv_gz_shards(&shards, &[out0.clone(), out1.clone()], &progress)
            .expect("merge csv.gz shards");

        assert_eq!(read_gzip(&out0), "a,1\nc,3\n");
        assert_eq!(read_gzip(&out1), "b,2\n");
    }

    #[test]
    fn runs_end_to_end_with_sharded_outputs() {
        let temp = tempdir().expect("create tempdir");
        let input_root = temp.path().join("input");
        let nested = input_root.join("nested");
        fs::create_dir_all(&nested).expect("create nested input dir");

        fs::write(input_root.join("a.npy"), [1_u8]).expect("write a.npy");
        fs::write(nested.join("b.npy"), [2_u8]).expect("write b.npy");
        write_gzip(&input_root.join("a.csv.gz"), "row_a\n");
        write_gzip(&nested.join("b.csv.gz"), "row_b\n");
        fs::write(input_root.join("ignore.bin"), [9_u8]).expect("write ignored file");

        let output_path = temp.path().join("sharded");
        let config = MergeConfig {
            input_path: input_root.clone(),
            num_files: 2,
            output_path: output_path.clone(),
        };

        let report = merge_files(&config).expect("run merge");

        assert_eq!(report.npy_inputs, 2);
        assert_eq!(report.csv_gz_inputs, 2);
        assert!(output_path.join("00000000.npy").exists());
        assert!(output_path.join("00000001.npy").exists());
        assert!(output_path.join("00000000.csv.gz").exists());
        assert!(output_path.join("00000001.csv.gz").exists());
    }

    #[test]
    fn rejects_zero_num_files() {
        let output = PathBuf::from("anything");
        let err = build_output_plan(&output, 0).expect_err("expected invalid num-files error");
        match err {
            MergeError::InvalidNumFiles(value) => assert_eq!(value, 0),
            other => panic!("unexpected error: {other}"),
        }
    }
}
