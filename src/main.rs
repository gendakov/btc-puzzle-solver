use primitive_types::U256;
use rayon::prelude::*;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use serde::Deserialize;
use std::env;
use std::fs;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

mod cuda_backend;

const ORDER: U256 = U256([
    0xBFD25E8CD0364141,
    0xBAAEDCE6AF48A03B,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
]);
const KEY_SIZE: usize = 33;
const VALUE_SIZE: usize = 32;
const SLOT_SIZE: usize = KEY_SIZE + VALUE_SIZE;
fn print_paid_feature_notice(features: &[&str]) {
    println!("This option is available in the paid version: {}.", features.join(", "));
    println!("Paid version: BTC Puzzle Solver Pro — EUR 30 (one-time payment). Send the EUR 30 equivalent in one of these currencies:");
    println!("  BTC: bc1qv8dlms8qw7v6lx7p2h9p84xd2yyvrckj08xlm4");
    println!("  LTC: LUM12iDExStPQexVr5XJfY8kXPmwfe2PBi");
    println!("  ETH: 0x21C65E8a759b7d04FaC964d832cE4d70B4C141a");
    println!("Order details: BTC Puzzle Solver Pro. BTC/LTC transfers do not have a reliable memo field; email the transaction ID, product name, and delivery email to gendakov@gmail.com.");
    println!("Do not put personal information in public transaction data. After payment is verified, the product will be sent to the email address you provide.");
}

#[derive(Debug, Deserialize)]
struct PuzzleFile {
    puzzles: Vec<Puzzle>,
}

#[derive(Debug, Deserialize)]
struct Puzzle {
    number: u32,
    bits: u32,
    range: PuzzleRange,
    pubkey: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PuzzleRange {
    start_hex: String,
    end_hex: String,
}

#[derive(Clone, Copy)]
struct MiniEntry {
    key: [u8; KEY_SIZE],
    private_key: U256,
}

struct MiniTable {
    slots: Vec<u8>,
    capacity: usize,
    mask: usize,
}

#[derive(Clone, Copy)]
struct SearchTask {
    direction: i8,
    start_step: U256,
    count: U256,
    worker_id: usize,
}

#[derive(Clone, Copy)]
struct SearchResult {
    private_key: U256,
    steps: U256,
    direction: i8,
    speed: f64,
    worker_id: usize,
}

struct Timings {
    minis: Duration,
    search: Duration,
    found: bool,
}

struct ProgressState {
    completed: std::sync::atomic::AtomicU64,
    total: f64,
    started: Instant,
}

fn parse_u256(value: &str) -> Result<U256, String> {
    let value = value.trim_start_matches("0x").trim_start_matches("0X");
    U256::from_str_radix(value, 16).map_err(|error| format!("Invalid 256-bit number: {error}"))
}

fn u256_to_bytes(value: U256) -> [u8; 32] {
    value.to_big_endian()
}

fn add_mod(a: U256, b: U256) -> U256 {
    (a + b) % ORDER
}

fn sub_mod(a: U256, b: U256) -> U256 {
    if a >= b {
        (a - b) % ORDER
    } else {
        (ORDER - (b - a) % ORDER) % ORDER
    }
}

fn parse_pubkey(hex: &str) -> Result<PublicKey, String> {
    let bytes = decode_hex(hex)?;
    PublicKey::from_slice(&bytes).map_err(|error| format!("Invalid public key: {error}"))
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    let value = value.trim_start_matches("0x").trim_start_matches("0X");
    if value.len() % 2 != 0 {
        return Err("Hex string must contain an even number of characters".to_owned());
    }
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| "Invalid hex character".to_owned())?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| "Invalid hex character".to_owned())?;
        decoded.push(((high << 4) | low) as u8);
    }
    Ok(decoded)
}

fn public_from_scalar(secp: &Secp256k1<secp256k1::All>, scalar: U256) -> Result<PublicKey, String> {
    let scalar = scalar % ORDER;
    if scalar.is_zero() {
        return Err("Zero is not a valid private-key scalar".to_owned());
    }
    let secret = SecretKey::from_byte_array(u256_to_bytes(scalar))
        .map_err(|error| format!("Invalid scalar: {error}"))?;
    Ok(PublicKey::from_secret_key(secp, &secret))
}

fn next_power_of_two(value: usize) -> usize {
    value.next_power_of_two()
}

impl MiniTable {
    fn from_entries(entries: &[MiniEntry]) -> Result<Self, String> {
        let capacity = next_power_of_two(entries.len().saturating_mul(2).max(2));
        let mut table = Self {
            slots: vec![0; capacity * SLOT_SIZE],
            capacity,
            mask: capacity - 1,
        };
        for entry in entries {
            table.insert(entry.key, entry.private_key)?;
        }
        Ok(table)
    }

    #[inline(always)]
    fn slot_offset(&self, slot: usize) -> usize {
        slot * SLOT_SIZE
    }

    fn insert(&mut self, key: [u8; KEY_SIZE], value: U256) -> Result<(), String> {
        let mut slot = u64::from_le_bytes(key[..8].try_into().unwrap()) as usize & self.mask;
        for _ in 0..self.capacity {
            let offset = self.slot_offset(slot);
            let current = &self.slots[offset..offset + KEY_SIZE];
            // Compressed secp256k1 keys start with 0x02 or 0x03;
            // a zero first byte is therefore a valid, fast empty-slot sentinel.
            if current[0] == 0 || current == key {
                self.slots[offset..offset + KEY_SIZE].copy_from_slice(&key);
                self.slots[offset + KEY_SIZE..offset + SLOT_SIZE]
                    .copy_from_slice(&u256_to_bytes(value));
                return Ok(());
            }
            slot = (slot + 1) & self.mask;
        }
        Err("Mini table is full".to_owned())
    }

    #[inline(always)]
    fn get(&self, key: &[u8; KEY_SIZE]) -> Option<U256> {
        let mut slot = u64::from_le_bytes(key[..8].try_into().unwrap()) as usize & self.mask;
        for _ in 0..self.capacity {
            let offset = self.slot_offset(slot);
            let current = &self.slots[offset..offset + KEY_SIZE];
            if current[0] == 0 {
                return None;
            }
            if current == key {
                let mut bytes = [0u8; VALUE_SIZE];
                bytes.copy_from_slice(&self.slots[offset + KEY_SIZE..offset + SLOT_SIZE]);
                return Some(U256::from_big_endian(&bytes));
            }
            slot = (slot + 1) & self.mask;
        }
        None
    }

    fn first_key(&self) -> Result<[u8; KEY_SIZE], String> {
        for slot in 0..self.capacity {
            let offset = self.slot_offset(slot);
            let current = &self.slots[offset..offset + KEY_SIZE];
            if current[0] != 0 {
                let mut key = [0u8; KEY_SIZE];
                key.copy_from_slice(current);
                return Ok(key);
            }
        }
        Err("Mini table is empty".to_owned())
    }

}

fn generate_minis(
    start: U256,
    end: U256,
    count: usize,
    threads: usize,
    secp: &Secp256k1<secp256k1::All>,
) -> Result<(MiniTable, U256, Vec<U256>), String> {
    if count < 2 {
        return Err("num_minis must be at least 2".to_owned());
    }
    let delta = (end - start) / U256::from(count - 1);
    let (table, private_keys, elapsed) = generate_minis_slice(start, delta, 0, count, threads, secp)?;
    println!(
        "[+] Minis generated in {:.2}s | {:.0} keys/s | table RAM ≈ {:.1} MB",
        elapsed.as_secs_f64(),
        count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        table.slots.len() as f64 / 1024.0 / 1024.0
    );
    println!("[*] Generated {count} minis | delta = {delta} (0x{delta:x})");
    Ok((table, delta, private_keys))
}

fn generate_minis_slice(
    start: U256,
    delta: U256,
    first_index: usize,
    count: usize,
    threads: usize,
    secp: &Secp256k1<secp256k1::All>,
) -> Result<(MiniTable, Vec<U256>, Duration), String> {
    let private_keys: Vec<U256> = (0..count)
        .map(|offset| start + U256::from(first_index + offset) * delta)
        .collect();
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| error.to_string())?;
    let entries: Vec<MiniEntry> = pool.install(|| {
        private_keys
            .par_iter()
            .map(|private_key| {
                let public_key = public_from_scalar(secp, *private_key)
                    .expect("private key in puzzle range must be valid");
                let serialized = public_key.serialize();
                MiniEntry {
                    key: serialized,
                    private_key: *private_key,
                }
            })
            .collect()
    });
    let table = MiniTable::from_entries(&entries)?;
    Ok((table, private_keys, started.elapsed()))
}

fn split_tasks(max_steps: U256, workers: usize) -> Vec<SearchTask> {
    if workers == 1 {
        return vec![SearchTask {
            direction: 0,
            start_step: U256::zero(),
            count: max_steps,
            worker_id: 0,
        }];
    }
    let positive_parts = (workers + 1) / 2;
    let negative_parts = workers / 2;
    let mut tasks = Vec::with_capacity(workers);
    for (direction, parts, offset) in [
        (1i8, positive_parts, 0usize),
        (-1i8, negative_parts, positive_parts),
    ] {
        for index in 0..parts {
            let start = max_steps * U256::from(index) / U256::from(parts);
            let end = max_steps * U256::from(index + 1) / U256::from(parts);
            if end > start {
                tasks.push(SearchTask {
                    direction,
                    start_step: start,
                    count: end - start,
                    worker_id: offset + index,
                });
            }
        }
    }
    tasks
}

fn search_one_direction(
    target: PublicKey,
    direction: i8,
    task: SearchTask,
    step: U256,
    table: &MiniTable,
    stop: &AtomicBool,
    secp: &Secp256k1<secp256k1::All>,
    progress: &ProgressState,
) -> Option<SearchResult> {
    let signed_step = if direction > 0 {
        step % ORDER
    } else {
        (ORDER - step % ORDER) % ORDER
    };
    if signed_step.is_zero() {
        return None;
    }
    let shift = (task.start_step * step) % ORDER;
    let current_shift = if direction > 0 {
        shift
    } else {
        (ORDER - shift) % ORDER
    };
    let mut current = if current_shift.is_zero() {
        target
    } else {
        target
            .combine(&public_from_scalar(secp, current_shift).ok()?)
            .ok()?
    };
    let add_point = public_from_scalar(secp, signed_step).ok()?;
    let started = Instant::now();
    let mut local_step = U256::one();
    let mut pending_progress = 0u64;
    while local_step <= task.count {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        pending_progress += 1;
        if pending_progress >= 50_000 {
            progress
                .completed
                .fetch_add(pending_progress, Ordering::Relaxed);
            pending_progress = 0;
        }
        current = current.combine(&add_point).ok()?;
        let serialized = current.serialize();
        if let Some(mini_private) = table.get(&serialized) {
            let absolute_step = task.start_step + local_step;
            let adjustment = (absolute_step * step) % ORDER;
            let private_key = if direction > 0 {
                sub_mod(mini_private, adjustment)
            } else {
                add_mod(mini_private, adjustment)
            };
            let speed = absolute_step.low_u64() as f64
                / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            progress
                .completed
                .fetch_add(pending_progress, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
            return Some(SearchResult {
                private_key,
                steps: absolute_step,
                direction,
                speed,
                worker_id: task.worker_id,
            });
        }
        local_step += U256::one();
    }
    progress
        .completed
        .fetch_add(pending_progress, Ordering::Relaxed);
    None
}

fn search_task(
    target: PublicKey,
    task: SearchTask,
    step: U256,
    table: &MiniTable,
    stop: &AtomicBool,
    secp: &Secp256k1<secp256k1::All>,
    progress: &ProgressState,
) -> Option<SearchResult> {
    if task.direction == 0 {
        search_one_direction(target, 1, task, step, table, stop, secp, progress)
            .or_else(|| search_one_direction(target, -1, task, step, table, stop, secp, progress))
    } else {
        search_one_direction(
            target,
            task.direction,
            task,
            step,
            table,
            stop,
            secp,
            progress,
        )
    }
}

fn start_status_thread(
    progress: Arc<ProgressState>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut last_report = Instant::now();
        loop {
            thread::sleep(Duration::from_millis(100));
            if last_report.elapsed() < Duration::from_secs(1) && !stop.load(Ordering::Relaxed) {
                continue;
            }
            last_report = Instant::now();
            let elapsed = progress.started.elapsed().as_secs_f64();
            let completed = progress.completed.load(Ordering::Relaxed) as f64;
            let speed = completed / elapsed.max(f64::MIN_POSITIVE);
            let fraction = (completed / progress.total).clamp(0.0, 1.0);
            let remaining = (progress.total - completed).max(0.0);
            let eta = if speed > 0.0 {
                remaining / speed
            } else {
                f64::INFINITY
            };
            if eta.is_finite() {
                print!(
                    "\r[status] progress={:6.2}% | points={:.3}M/{:.3}M | speed={:.2} Mpoints/s | ETA={}",
                    fraction * 100.0,
                    completed / 1_000_000.0,
                    progress.total / 1_000_000.0,
                    speed / 1_000_000.0,
                    format_duration(eta),
                );
            } else {
                print!(
                    "\r[status] progress={:6.2}% | points={:.3}M/{:.3}M | speed={:.2} Mpoints/s | ETA=--:--:--",
                    fraction * 100.0,
                    completed / 1_000_000.0,
                    progress.total / 1_000_000.0,
                    speed / 1_000_000.0,
                );
            }
            use std::io::Write;
            let _ = std::io::stdout().flush();
            if stop.load(Ordering::Relaxed) {
                println!();
                return;
            }
        }
    })
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--:--:--".to_owned();
    }
    let total_seconds = seconds.round() as u64;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

fn benchmark_lookup(table: &MiniTable) -> Result<(), String> {
    let key = table.first_key()?;
    let started = Instant::now();
    let repetitions = 100_000usize;
    let mut found = 0usize;
    for _ in 0..repetitions {
        if table.get(&key).is_some() {
            found += 1;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "[benchmark] lookup: {:.2} ns/op ({repetitions} operations, {found} hits)",
        elapsed * 1e9 / repetitions as f64
    );
    Ok(())
}

fn load_puzzle(path: &str, number: u32) -> Result<Puzzle, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("Cannot read {path}: {error}"))?;
    let file: PuzzleFile =
        serde_json::from_str(&content).map_err(|error| format!("Invalid JSON: {error}"))?;
    file.puzzles
        .into_iter()
        .find(|puzzle| puzzle.number == number)
        .ok_or_else(|| format!("Puzzle {number} was not found"))
}

fn write_private_key(path: &str, private_key: U256) -> Result<(), String> {
    fs::write(path, format!("{private_key:064x}\n"))
        .map_err(|error| format!("Cannot write private key to {path}: {error}"))?;
    println!("[+] Private key written to file: {path}");
    Ok(())
}

fn clear_progress_files() -> Result<usize, String> {
    let mut removed = 0;
    for entry in fs::read_dir(".").map_err(|error| format!("Cannot read current directory: {error}"))? {
        let entry = entry.map_err(|error| format!("Error while scanning current directory: {error}"))?;
        if !entry.file_type().map_err(|error| format!("Cannot inspect {}: {error}", entry.path().display()))?.is_file() {
            continue;
        }
        let path = entry.path();
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        let known_checkpoint_name = name.ends_with(".checkpoint") || name.ends_with(".checkpoint.tmp");
        let mut prefix = [0u8; 8];
        let has_checkpoint_signature = fs::File::open(&path)
            .and_then(|mut file| file.read_exact(&mut prefix))
            .is_ok()
            && (&prefix == b"BTCCKP02" || &prefix == b"BTCCKP03" || &prefix == b"BTCCKP04");
        if known_checkpoint_name || has_checkpoint_signature {
            fs::remove_file(&path).map_err(|error| format!("Cannot remove progress file {}: {error}", path.display()))?;
            println!("[+] Removed progress file: {}", path.display());
            removed += 1;
        }
    }
    println!("[*] Progress files removed: {removed}");
    Ok(removed)
}

fn solve(
    puzzle_number: u32,
    mut num_minis: usize,
    step: U256,
    json_path: &str,
    workers: usize,
    show_minis: bool,
    benchmark: bool,
    gpu: bool,
    gpu_device: Option<usize>,
    output_file: Option<&str>,
    checkpoint_file: &str,
    fingerprint_bits: u32,
) -> Result<Timings, String> {
    if num_minis < 2 {
        return Err("num_minis must be at least 2".to_owned());
    }
    let puzzle = load_puzzle(json_path, puzzle_number)?;
    if fingerprint_bits < 8 || fingerprint_bits > 192 || fingerprint_bits % 8 != 0 {
        return Err("--fp must be from 8 to 192 bits in increments of 8".to_owned());
    }
    let target_hex = puzzle
        .pubkey
        .ok_or_else(|| format!("Puzzle {puzzle_number} has no public key"))?;
    let target = parse_pubkey(&target_hex)?;
    let mut start = parse_u256(&puzzle.range.start_hex)?;
    let mut end = parse_u256(&puzzle.range.end_hex)?;
    let search_key_count = end - start + U256::one();
    let delta = (end - start) / U256::from(num_minis - 1);
    println!("{}", "=".repeat(72));
    println!("[*] Puzzle #{}", puzzle.number);
    println!("    Range       : 0x{start:x} → 0x{end:x}");
    println!("    Keys        : {search_key_count}");
    println!("    Bits        : {}", puzzle.bits);
    println!("    Target      : {target_hex}");
    println!("    Minis       : {num_minis}");
    if gpu { println!("    Fingerprint : {fingerprint_bits} bit"); }
    println!("    Step        : {step}");
    let mut found = false;
    if gpu {
        println!("    Mini build  : one GPU producer");
    } else {
        println!("    Workers     : {workers} (Rayon)");
    }
    println!("    Backend     : {}", if gpu {
        gpu_device.map_or_else(
            || "CUDA GPU 0".to_owned(),
            |device| format!("CUDA GPU {device}"),
        )
    } else {
        "CPU Rayon".to_owned()
    });
    println!("{}", "=".repeat(72));
    let secp = Secp256k1::new();

    if gpu {
        println!("\n[*] CUDA BSGS | step = {step} | total minis = {num_minis}");
        println!("[*] The selected GPU stores the mini table and searches its corresponding range segment.");
        let gpu_started = Instant::now();
        let gpu_result = cuda_backend::search(
            target.serialize(),
            public_from_scalar(&secp, step)?.serialize(),
            step,
            num_minis,
            start,
            end,
            gpu_device,
            fingerprint_bits,
            checkpoint_file,
        )?;
        let gpu_time = gpu_started.elapsed();
        match gpu_result {
            Some(result) => {
                println!("\n{}", "=".repeat(72));
                println!("[!!!] CUDA found a private key on GPU {}", result.device);
                println!("      Priv (dec)  : {}", result.private_key);
                println!("      Priv (hex)  : {:064x}", result.private_key);
                println!("      Giant steps : {}", result.tested);
                let equivalent_rate = format_bruteforce_rate(
                    result.tested as f64 / result.elapsed_seconds * num_minis as f64,
                );
                println!("      Brute-force : ~{equivalent_rate} (equivalent)");
                println!("      Total time  : {:.2}s", gpu_time.as_secs_f64());
                if public_from_scalar(&secp, result.private_key)
                    .is_ok_and(|public_key| public_key.serialize() == target.serialize()) {
                    found = true;
                    println!("[+] VERIFICATION : SUCCESS ✓");
                    if let Some(path) = output_file {
                        write_private_key(path, result.private_key)?;
                    }
                    match fs::remove_file(checkpoint_file) {
                        Ok(()) => println!("[+] Checkpoint removed after successful search."),
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                        Err(error) => eprintln!("[!] Cannot remove checkpoint {checkpoint_file}: {error}"),
                    }
                } else {
                    println!("[-] VERIFICATION : FAILED ✗; CUDA search will reject false candidates and continue automatically.");
                }
            }
            None => println!("\n[-] CUDA BSGS found no match ({:.2}s)", gpu_time.as_secs_f64()),
        }
        return Ok(Timings { minis: Duration::ZERO, search: gpu_time, found });
    }

    let minis_started = Instant::now();
    let (table, generated_delta, private_keys) =
        generate_minis(start, end, num_minis, workers, &secp)?;
    let minis_time = minis_started.elapsed();
    debug_assert_eq!(delta, generated_delta);
    if show_minis {
        println!("\n[*] First 10 minis:");
        for (index, private_key) in private_keys.iter().take(10).enumerate() {
            println!("    [{index:3}] 0x{private_key:x}");
        }
        if private_keys.len() > 10 {
            println!("    ... +{} more", private_keys.len() - 10);
        }
    }
    if benchmark {
        benchmark_lookup(&table)?;
    }

    let max_steps = delta / step + U256::from(256u64);
    println!("\n[*] Search | max_steps = {max_steps} | step = {step}");
    println!("{}", "-".repeat(72));
    let total_work = max_steps
        .saturating_mul(U256::from(2u8))
        .to_string()
        .parse::<f64>()
        .unwrap_or(f64::MAX);
    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(ProgressState {
        completed: std::sync::atomic::AtomicU64::new(0),
        total: total_work.max(1.0),
        started: Instant::now(),
    });
    let status_thread = start_status_thread(Arc::clone(&progress), Arc::clone(&stop));
    let search_started = Instant::now();
    let result = split_tasks(max_steps, workers)
        .into_par_iter()
        .find_map_any(|task| search_task(target, task, step, &table, &stop, &secp, &progress));
    stop.store(true, Ordering::Relaxed);
    let _ = status_thread.join();
    let search_time = search_started.elapsed();
    match result {
        Some(result) => {
            println!("\n{}", "=".repeat(72));
            println!("[!!!] PRIVATE KEY FOUND (worker {})", result.worker_id);
            println!("      Steps       : {}", result.steps);
            println!("      Direction   : {:+}", result.direction);
            println!(
                "      Speed       : {:.2} Mpoints/s",
                result.speed / 1_000_000.0
            );
            println!("      Total time  : {:.2}s", search_time.as_secs_f64());
            println!("      Priv (dec)  : {}", result.private_key);
            println!("      Priv (hex)  : {:064x}", result.private_key);
            println!("{}", "=".repeat(72));
            let derived_pubkey = public_from_scalar(&secp, result.private_key)?.serialize();
            let check = derived_pubkey
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            println!("[*] Public key   : {check}");
            if derived_pubkey == target.serialize() {
                found = true;
                println!("[+] VERIFICATION : SUCCESS ✓");
                if let Some(path) = output_file {
                    write_private_key(path, result.private_key)?;
                }
            } else {
                println!("[-] VERIFICATION : FAILED ✗");
            }
        }
        None => println!(
            "\n[-] No match found ({:.2}s)",
            search_time.as_secs_f64()
        ),
    }
    Ok(Timings {
        minis: minis_time,
        search: search_time,
        found,
    })
}

fn format_bruteforce_rate(rate: f64) -> String {
    let units = ["keys/s", "Kkeys/s", "Mkeys/s", "Gkeys/s", "Tkeys/s", "Pkeys/s", "Ekeys/s"];
    let mut scaled = rate.max(0.0);
    let mut unit = 0;
    while scaled >= 1000.0 && unit + 1 < units.len() {
        scaled /= 1000.0;
        unit += 1;
    }
    format!("{scaled:.2} {}", units[unit])
}

fn usage() {
    println!("Usage:");
    println!("  btc_puzzle_solver <puzzle> <num_minis|max> [step] [options]");
    println!();
    println!("Required arguments:");
    println!("  <puzzle>                  Puzzle number from btc_puzzles_160_with_pubkey.json");
    println!("  <num_minis|max>           Number of minis; max is a paid CUDA option");
    println!("  [step]                    Giant-step size; defaults to 1");
    println!();
    println!("Options:");
    println!("  --gpu                     Use CUDA instead of the CPU search");
    println!("  --gpu-device N            Use only CUDA device N; GPU 0 is used by default");
    println!("  -o FILE, --output FILE    Write the verified private key as 64 hexadecimal characters");
    println!("  --checkpoint FILE         Resume checkpoint path; a path is chosen automatically by default");
    println!("  --clear                   Remove checkpoint/progress files in the current directory before searching");
    println!("  --fp N                    CUDA fingerprint size, 8–192 bits in increments of 8 (default: 192); alias: --fingerprint-bits");
    println!("  --sector K/N or A-B/N    [PAID] Search sector K or process sectors A through B from N equal parts");
    println!("  --sector-size-bits B     [PAID] Set chunk size to 2^B keys; use --sector A-B to select chunks");
    println!("  --workers N               Number of CPU workers (CUDA uses one producer per GPU)");
    println!("  --show-minis              Show the first mini private keys");
    println!("  --benchmark               Benchmark the CPU lookup table");
    println!("  --profile FILE            Write timings and worker count to a file");
    println!("  -h, --help                Show this help");
    println!();
    println!("CUDA and memory:");
    println!("  In CUDA mode, the mini table resides in VRAM; bounded RAM staging is used only for transfers.");
    println!("  CUDA uses one GPU device; GPU 0 is selected by default.");
    println!("  num_minis=max is available in the paid version.");
    println!("  The private scalar is reconstructed when a match is found.");
    println!("  The CUDA kernel is compiled with nvcc during the build: cargo build --release");
    println!("  Experimental warp batch-inversion kernel: CUDA_WARP_BATCH=1 cargo build --release");
    println!("  Build-time CUDA tuning: CUDA_BATCH_SIZE (1M–16M), CUDA_THREADS_PER_BLOCK (64/128/256/512), CUDA_MAX_WORKERS (16384/32768/65536/131072)");
    println!();
    println!("Examples:");
    println!("  ./target/release/btc_puzzle_solver 49 1000000 --gpu");
    println!("  ./target/release/btc_puzzle_solver 140 max 100 --gpu -o w140.txt");
    println!("  ./target/release/btc_puzzle_solver 49 100000000 --workers 8");
    println!("  ./target/release/btc_puzzle_solver 44 1000000 1 --workers 8 --benchmark");
    println!("  ./target/release/btc_puzzle_solver 70 max --gpu --sector 3-5/10");
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage();
        return Ok(());
    }
    if args.len() < 2 {
        if !args.is_empty() && args.iter().all(|arg| arg == "--clear") {
            clear_progress_files()?;
            return Ok(());
        }
        usage();
        return Ok(());
    }
    let puzzle_number = args[0]
        .parse::<u32>()
        .map_err(|_| "Invalid puzzle number".to_owned())?;
    let requested_auto_capacity = args[1].eq_ignore_ascii_case("max");
    let num_minis = if requested_auto_capacity {
        2
    } else {
        args[1].parse::<usize>().map_err(|_| "Invalid number of minis".to_owned())?
    };
    let mut step = U256::one();
    let mut workers = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    let mut show_minis = false;
    let mut benchmark = false;
    let mut profile_file: Option<String> = None;
    let mut gpu = false;
    let mut gpu_device = None;
    let mut output_file: Option<String> = None;
    let mut checkpoint_file: Option<String> = None;
    let mut fingerprint_bits = 192u32;
    let mut clear = false;
    let mut paid_sector_requested = false;
    let mut paid_features = Vec::new();
    if requested_auto_capacity {
        paid_features.push("automatic CUDA mini capacity (max)");
    }
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--show-minis" => show_minis = true,
            "--benchmark" => benchmark = true,
            "--gpu" => gpu = true,
            "--clear" => clear = true,
            "-o" | "--output" => {
                index += 1;
                output_file = Some(args.get(index).ok_or("Missing file path for -o/--output")?.clone());
            }
            "--checkpoint" => {
                index += 1;
                checkpoint_file = Some(args.get(index).ok_or("Missing file path for --checkpoint")?.clone());
            }
            "--fp" | "--fingerprint-bits" => {
                index += 1;
                fingerprint_bits = args.get(index).ok_or("Missing value for --fp")?
                    .parse::<u32>().map_err(|_| "Fingerprint size must be a multiple of 8 from 8 to 192")?;
                if fingerprint_bits < 8 || fingerprint_bits > 192 || fingerprint_bits % 8 != 0 {
                    return Err("--fp must be from 8 to 192 bits in increments of 8".to_owned());
                }
            }
            "--sector" => {
                paid_sector_requested = true;
                index += 1;
                let _ = args.get(index).ok_or("Missing value for --sector")?;
            }
            "--sector-size-bits" => {
                paid_sector_requested = true;
                index += 1;
                let _ = args.get(index).ok_or("Missing value for --sector-size-bits")?;
            }
            "--gpu-device" => {
                index += 1;
                gpu_device = Some(args
                    .get(index)
                    .ok_or("Missing value for --gpu-device")?
                    .parse()
                    .map_err(|_| "Invalid CUDA device index")?);
            }
            "--workers" => {
                index += 1;
                workers = args
                    .get(index)
                    .ok_or("Missing value for --workers")?
                    .parse()
                    .map_err(|_| "Invalid number of workers")?;
            }
            "--profile" => {
                index += 1;
                profile_file = Some(args.get(index).ok_or("Missing file path for --profile")?.clone());
            }
            value if !value.starts_with('-') && index == 2 => step = parse_u256(value)?,
            value => return Err(format!("Unknown argument: {value}")),
        }
        index += 1;
    }
    if paid_sector_requested {
        paid_features.push("sector and fixed-size chunk searches");
    }
    if !paid_features.is_empty() {
        // AI/maintainer note: keep paid sector and auto-capacity implementations out of this free source.
        print_paid_feature_notice(&paid_features);
        return Ok(());
    }
    workers = workers.max(1).min(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    );
    if clear {
        clear_progress_files()?;
    }
    let mut total_minis_time = Duration::ZERO;
    let mut total_search_time = Duration::ZERO;
    let default_checkpoint = format!("btc_solver_p{puzzle_number}_n{num_minis}_gpu_single_v1_step_{step}.checkpoint");
    let checkpoint_path = checkpoint_file.as_deref().unwrap_or(&default_checkpoint);
    let timings = solve(
        puzzle_number,
        num_minis,
        step,
        "btc_puzzles_160_with_pubkey.json",
        workers,
        show_minis,
        benchmark,
        gpu,
        gpu_device,
        output_file.as_deref(),
        checkpoint_path,
        fingerprint_bits,
    )?;
    total_minis_time += timings.minis;
    total_search_time += timings.search;
    if let Some(path) = profile_file {
        let report = format!(
            "minis_seconds={:.6}\nsearch_seconds={:.6}\nworkers={workers}\n",
            total_minis_time.as_secs_f64(),
            total_search_time.as_secs_f64()
        );
        fs::write(&path, report)
            .map_err(|error| format!("Cannot write profile: {error}"))?;
        println!("[profile] Report written: {path}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secp() -> Secp256k1<secp256k1::All> {
        Secp256k1::new()
    }

    fn scalar_public(scalar: u64) -> PublicKey {
        public_from_scalar(&secp(), U256::from(scalar)).unwrap()
    }

    fn mini_entry(scalar: u64) -> MiniEntry {
        MiniEntry {
            key: scalar_public(scalar).serialize(),
            private_key: U256::from(scalar),
        }
    }

    #[test]
    fn parses_u256_and_converts_big_endian() {
        let value = parse_u256("0x0102").unwrap();
        assert_eq!(value, U256::from(0x0102));
        assert_eq!(&u256_to_bytes(value)[30..], &[1, 2]);
        assert!(parse_u256("xyz").is_err());
    }

    #[test]
    fn modular_scalar_arithmetic_wraps_at_group_order() {
        assert_eq!(add_mod(ORDER - U256::one(), U256::one()), U256::zero());
        assert_eq!(sub_mod(U256::zero(), U256::one()), ORDER - U256::one());
        assert_eq!(sub_mod(U256::from(9), U256::from(4)), U256::from(5));
    }

    #[test]
    fn decodes_hex_and_parses_compressed_public_key() {
        let decoded = decode_hex("0x0200ff").unwrap();
        assert_eq!(decoded, [2, 0, 255]);
        assert!(decode_hex("abc").is_err());
        assert!(decode_hex("gg").is_err());
        assert_eq!(parse_pubkey(&format!("{:x}", scalar_public(1))).unwrap(), scalar_public(1));
    }

    #[test]
    fn derives_generator_public_key_from_scalar_one() {
        let expected = decode_hex("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798").unwrap();
        assert_eq!(scalar_public(1).serialize().as_slice(), expected);
        assert!(public_from_scalar(&secp(), U256::zero()).is_err());
    }

    #[test]
    fn mini_table_inserts_looks_up_and_stops_at_empty_slot() {
        let entries = [mini_entry(1), mini_entry(2), mini_entry(3)];
        let table = MiniTable::from_entries(&entries).unwrap();
        for entry in entries {
            assert_eq!(table.get(&entry.key), Some(entry.private_key));
        }
        assert_eq!(table.capacity, 8);
        assert_eq!(table.get(&scalar_public(4).serialize()), None);
        assert!(table.first_key().is_ok());
        assert!(MiniTable::from_entries(&[]).unwrap().first_key().is_err());
    }

    #[test]
    fn mini_table_linear_probing_handles_colliding_initial_hashes() {
        let mut first = [0u8; KEY_SIZE];
        let mut second = [0u8; KEY_SIZE];
        first[0] = 2;
        second[0] = 2;
        second[32] = 1;
        let mut table = MiniTable {
            slots: vec![0; 2 * SLOT_SIZE],
            capacity: 2,
            mask: 1,
        };
        table.insert(first, U256::from(11)).unwrap();
        table.insert(second, U256::from(22)).unwrap();
        assert_eq!(table.get(&first), Some(U256::from(11)));
        assert_eq!(table.get(&second), Some(U256::from(22)));
        assert!(table.insert([2; KEY_SIZE], U256::from(33)).is_err());
    }

    #[test]
    fn generated_minis_use_inclusive_endpoints_and_even_delta() {
        let (table, delta, keys) = generate_minis(
            U256::from(1), U256::from(5), 3, 1, &secp(),
        ).unwrap();
        assert_eq!(delta, U256::from(2));
        assert_eq!(keys, [U256::from(1), U256::from(3), U256::from(5)]);
        for scalar in [1, 3, 5] {
            assert_eq!(table.get(&scalar_public(scalar).serialize()), Some(U256::from(scalar)));
        }
        assert!(generate_minis(U256::one(), U256::from(5), 1, 1, &secp()).is_err());
    }

    #[test]
    fn split_tasks_covers_both_directions_without_overlaps() {
        for workers in 1..=8 {
            let tasks = split_tasks(U256::from(101), workers);
            let positive: U256 = tasks.iter().filter(|task| task.direction >= 0)
                .map(|task| task.count).fold(U256::zero(), |a, b| a + b);
            let negative: U256 = tasks.iter().filter(|task| task.direction <= 0)
                .map(|task| task.count).fold(U256::zero(), |a, b| a + b);
            assert_eq!(positive, U256::from(101));
            assert_eq!(negative, U256::from(101));
            assert!(tasks.iter().all(|task| task.count > U256::zero()));
        }
    }

    #[test]
    fn search_checks_positive_and_negative_giant_steps() {
        let secp = secp();
        let stop = AtomicBool::new(false);
        let progress = ProgressState {
            completed: std::sync::atomic::AtomicU64::new(0),
            total: 2.0,
            started: Instant::now(),
        };
        let positive_table = MiniTable::from_entries(&[mini_entry(3)]).unwrap();
        let positive = search_one_direction(
            scalar_public(2), 1,
            SearchTask { direction: 1, start_step: U256::zero(), count: U256::one(), worker_id: 0 },
            U256::one(), &positive_table, &stop, &secp, &progress,
        ).unwrap();
        assert_eq!(positive.private_key, U256::from(2));

        let stop = AtomicBool::new(false);
        let negative_table = MiniTable::from_entries(&[mini_entry(1)]).unwrap();
        let negative = search_one_direction(
            scalar_public(2), -1,
            SearchTask { direction: -1, start_step: U256::zero(), count: U256::one(), worker_id: 1 },
            U256::one(), &negative_table, &stop, &secp, &progress,
        ).unwrap();
        assert_eq!(negative.private_key, U256::from(2));
    }

    #[test]
    fn puzzle_loader_reads_known_entry_and_reports_missing_number() {
        let puzzle = load_puzzle("btc_puzzles_160_with_pubkey.json", 2).unwrap();
        assert_eq!(puzzle.number, 2);
        assert!(puzzle.pubkey.is_some());
        assert!(load_puzzle("btc_puzzles_160_with_pubkey.json", 999).is_err());
    }

    #[test]
    #[test]
    fn private_key_writer_emits_fixed_width_hex() {
        let path = std::env::temp_dir().join(format!("btc-solver-test-{}.key", std::process::id()));
        write_private_key(path.to_str().unwrap(), U256::from(0x1234)).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), format!("{:064x}\n", 0x1234));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn cpu_solver_finds_and_verifies_small_puzzle_key() {
        let timings = solve(
            2, 2, U256::one(), "btc_puzzles_160_with_pubkey.json", 1,
            false, false, false, None, None, "unused.checkpoint", 192,
        ).unwrap();
        assert!(timings.found);
    }
}
