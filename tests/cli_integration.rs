use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use csv::StringRecord;
use flate2::read::MultiGzDecoder;
use tempfile::tempdir;
use walkdir::WalkDir;

fn fixture_input_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tokenized_input")
}

fn collect_sorted_npy_paths(root: &Path) -> Vec<PathBuf> {
    let mut files = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "npy"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn collect_sorted_csv_gz_paths(root: &Path) -> Vec<PathBuf> {
    let mut files = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".csv.gz"))
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn read_gzip_csv_records(path: &Path) -> Vec<Vec<String>> {
    let input = fs::File::open(path).expect("open gzip");
    let decoder = MultiGzDecoder::new(input);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(decoder);
    reader
        .records()
        .map(|record| {
            record
                .expect("read csv record")
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn concat_npy_bytes(paths: &[PathBuf]) -> Vec<u8> {
    let mut output = Vec::new();
    for path in paths {
        output.extend(fs::read(path).expect("read npy"));
    }
    output
}

fn is_metadata_header(record: &StringRecord) -> bool {
    let Some(start) = record.get(0) else {
        return false;
    };
    let Some(end) = record.get(1) else {
        return false;
    };
    start.eq_ignore_ascii_case("start") && end.eq_ignore_ascii_case("end")
}

fn expected_remapped_csv_records(paths: &[PathBuf]) -> Vec<Vec<String>> {
    let mut output = Vec::new();
    let mut next_start = 0_u64;
    let mut wrote_header = false;

    for path in paths {
        let input = fs::File::open(path).expect("open gzip");
        let decoder = MultiGzDecoder::new(input);
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(decoder);

        for record in reader.records() {
            let record = record.expect("read csv record");
            if record.is_empty() {
                continue;
            }

            if is_metadata_header(&record) {
                if !wrote_header {
                    output.push(
                        record
                            .iter()
                            .map(std::string::ToString::to_string)
                            .collect::<Vec<_>>(),
                    );
                    wrote_header = true;
                }
                continue;
            }

            assert!(record.len() >= 5, "expected at least 5 columns");
            let start: u64 = record.get(0).expect("start").parse().expect("parse start");
            let end: u64 = record.get(1).expect("end").parse().expect("parse end");
            assert!(end >= start, "end must be >= start");
            let length = end - start;
            let new_start = next_start;
            let new_end = new_start + length;
            next_start = new_end;

            let mut row = record
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>();
            row[0] = new_start.to_string();
            row[1] = new_end.to_string();
            output.push(row);
        }
    }

    output
}

fn shard_round_robin(paths: &[PathBuf], num_shards: usize) -> Vec<Vec<PathBuf>> {
    let mut shards = vec![Vec::new(); num_shards];
    for (index, path) in paths.iter().enumerate() {
        shards[index % num_shards].push(path.clone());
    }
    shards
}

#[test]
fn cli_merges_into_single_output_pair() {
    let input = fixture_input_dir();
    let temp = tempdir().expect("create tempdir");
    let output_stem = temp.path().join("single").join("merged").join("base");

    let output = Command::new(env!("CARGO_BIN_EXE_reshard-tokenized"))
        .arg("--input-path")
        .arg(&input)
        .arg("--num-files")
        .arg("1")
        .arg("--output-path")
        .arg(&output_stem)
        .output()
        .expect("run cli");
    assert!(
        output.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let npy_inputs = collect_sorted_npy_paths(&input);
    let csv_inputs = collect_sorted_csv_gz_paths(&input);
    let expected_npy = concat_npy_bytes(&npy_inputs);
    let expected_csv = expected_remapped_csv_records(&csv_inputs);

    let merged_npy = output_stem.with_extension("npy");
    let merged_csv = output_stem.with_extension("csv.gz");
    assert_eq!(fs::read(merged_npy).expect("read merged npy"), expected_npy);
    assert_eq!(read_gzip_csv_records(&merged_csv), expected_csv);
}

#[test]
fn cli_merges_into_two_shards() {
    let input = fixture_input_dir();
    let temp = tempdir().expect("create tempdir");
    let output_dir = temp.path().join("shards");

    let output = Command::new(env!("CARGO_BIN_EXE_reshard-tokenized"))
        .arg("--input-path")
        .arg(&input)
        .arg("--num-files")
        .arg("2")
        .arg("--output-path")
        .arg(&output_dir)
        .output()
        .expect("run cli");
    assert!(
        output.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    for index in 0..2 {
        assert!(output_dir.join(format!("{index:08}.npy")).exists());
        assert!(output_dir.join(format!("{index:08}.csv.gz")).exists());
    }

    let npy_shards = shard_round_robin(&collect_sorted_npy_paths(&input), 2);
    let csv_shards = shard_round_robin(&collect_sorted_csv_gz_paths(&input), 2);

    for index in 0..2 {
        let expected_npy = concat_npy_bytes(&npy_shards[index]);
        let expected_csv = expected_remapped_csv_records(&csv_shards[index]);
        let output_npy = output_dir.join(format!("{index:08}.npy"));
        let output_csv = output_dir.join(format!("{index:08}.csv.gz"));
        assert_eq!(fs::read(output_npy).expect("read shard npy"), expected_npy);
        assert_eq!(read_gzip_csv_records(&output_csv), expected_csv);
    }
}

#[test]
fn cli_fails_for_invalid_num_files() {
    let input = fixture_input_dir();
    let temp = tempdir().expect("create tempdir");
    let output_path = temp.path().join("bad");

    let output = Command::new(env!("CARGO_BIN_EXE_reshard-tokenized"))
        .arg("--input-path")
        .arg(&input)
        .arg("--num-files")
        .arg("0")
        .arg("--output-path")
        .arg(&output_path)
        .output()
        .expect("run cli");

    assert!(!output.status.success(), "expected failure");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("`--num-files` must be at least 1"),
        "unexpected stderr: {stderr}"
    );
}
