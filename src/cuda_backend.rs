use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use primitive_types::{U256, U512};
use secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

include!(concat!(env!("OUT_DIR"), "/cuda_tuning.rs"));

const CANDIDATES_PER_WORKER: u64 = 16;
const VRAM_SAFETY_RESERVE: usize = 128 * 1024 * 1024;
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
const GROUP_ORDER: U256 = U256([
    0xBFD25E8CD0364141,
    0xBAAEDCE6AF48A03B,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
]);
const KERNEL_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/secp256k1_search.cubin"));

pub struct GpuSearchResult { pub private_key: U256, pub tested: u64, pub elapsed_seconds: f64, pub device: usize }
const MINI_GENERATE_CHUNK: usize = 256;
const MINI_BUILD_BATCH: usize = 16_384;
const MINI_SEED_BYTES: usize = 72;
struct Task { direction: i32, start_step: U256, count: U256, checkpoint_id: usize, checkpoint_base: U256 }
#[derive(Clone, Copy)]
struct MiniPlan { mini_count: usize, mini_start: U256, mini_delta: U256, trailing_gap: U256, max_steps: U256 }
struct Checkpoint { path: PathBuf, header: Vec<u8>, completed: Vec<U256>, last_saved: Instant }
struct SharedState { tested: AtomicU64, stop: AtomicBool, search_started: Mutex<Option<Instant>>, result: Mutex<Option<U256>>, checkpoint: Arc<Mutex<Checkpoint>> }

extern "C" fn interrupt_handler(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

fn install_interrupt_handlers() {
    INTERRUPTED.store(false, Ordering::Relaxed);
    unsafe {
        libc::signal(libc::SIGINT, interrupt_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, interrupt_handler as *const () as libc::sighandler_t);
    }
}

impl Checkpoint {
    fn open(path: &str, header: Vec<u8>, task_count: usize) -> Result<Self, String> {
        let path = PathBuf::from(path);
        let completed = if path.exists() {
            let bytes = fs::read(&path).map_err(|e| format!("Cannot read checkpoint {}: {e}", path.display()))?;
            let expected_len = header.len().checked_add(task_count.checked_mul(32).ok_or("Checkpoint is too large")?).ok_or("Checkpoint is too large")?;
            if bytes.len() != expected_len || bytes.get(..header.len()) != Some(header.as_slice()) {
                return Err(format!("Checkpoint {} does not match the puzzle/step/minis/GPU configuration or is corrupted; remove it to start a new search", path.display()));
            }
            (0..task_count).map(|i| U256::from_big_endian(&bytes[header.len() + i * 32..header.len() + (i + 1) * 32])).collect()
        } else {
            vec![U256::zero(); task_count]
        };
        let mut checkpoint = Self { path, header, completed, last_saved: Instant::now() - Duration::from_secs(10) };
        checkpoint.save(true)?;
        Ok(checkpoint)
    }

    fn save(&mut self, force: bool) -> Result<(), String> {
        if !force && self.last_saved.elapsed() < Duration::from_secs(5) { return Ok(()); }
        let mut bytes = Vec::with_capacity(self.header.len() + self.completed.len() * 32);
        bytes.extend_from_slice(&self.header);
        for progress in &self.completed { bytes.extend_from_slice(&progress.to_big_endian()); }
        let temporary = self.path.with_extension("checkpoint.tmp");
        let mut file = File::create(&temporary).map_err(|e| format!("Cannot write checkpoint {}: {e}", temporary.display()))?;
        file.write_all(&bytes).and_then(|_| file.sync_all()).map_err(|e| format!("Cannot finish writing checkpoint {}: {e}", temporary.display()))?;
        fs::rename(&temporary, &self.path).map_err(|e| format!("Cannot update checkpoint {}: {e}", self.path.display()))?;
        self.last_saved = Instant::now();
        Ok(())
    }

    fn update(&mut self, task: usize, completed: U256, force: bool) -> Result<(), String> {
        self.completed[task] = completed;
        self.save(force)
    }
}

fn make_checkpoint_header(target: [u8; 33], step: U256, fingerprint_bits: u32, plan: MiniPlan, device: usize) -> Result<Vec<u8>, String> {
    let mut header = b"BTCCKP05".to_vec();
    header.extend_from_slice(&step.to_big_endian());
    header.extend_from_slice(&target);
    header.extend_from_slice(&fingerprint_bits.to_le_bytes());
    header.extend_from_slice(&u32::try_from(device).map_err(|_| "GPU index does not fit in the checkpoint format")?.to_le_bytes());
    header.extend_from_slice(&2u32.to_le_bytes());
    header.extend_from_slice(&u64::try_from(plan.mini_count).map_err(|_| "Mini count does not fit in the checkpoint format")?.to_le_bytes());
    header.extend_from_slice(&plan.mini_start.to_big_endian());
    header.extend_from_slice(&plan.mini_delta.to_big_endian());
    header.extend_from_slice(&plan.max_steps.to_big_endian());
    Ok(header)
}

fn selected_device(requested_device: Option<usize>) -> Result<usize, String> {
    let count = CudaContext::device_count().map_err(|e| format!("Cannot get the number of CUDA devices: {e:?}"))?;
    if count <= 0 { return Err("No CUDA device found".to_owned()); }
    let device = requested_device.unwrap_or(0);
    if device >= count as usize {
        return Err(format!("CUDA device {device} is unavailable; detected {count} device(s)"));
    }
    Ok(device)
}

fn mini_slot_bytes(fingerprint_bits: u32) -> usize {
    fingerprint_bits as usize / 8 + 8
}

fn make_mini_plan(start: U256, end: U256, step: U256, mini_count: usize) -> Result<MiniPlan, String> {
    if mini_count < 2 {
        return Err("At least 2 minis are required".to_owned());
    }
    let range_len = end - start + U256::one();
    let span = range_len - U256::one();
    let mini_delta = span / U256::from(mini_count - 1);
    let covered_span = mini_delta * U256::from(mini_count - 1);
    let trailing_gap = span - covered_span;
    let max_gap = trailing_gap.max(mini_delta);
    let max_steps = max_gap / step + U256::from(256u64);
    Ok(MiniPlan { mini_count, mini_start: start, mini_delta, trailing_gap, max_steps })
}

pub fn search(target: [u8; 33], step_pubkey: [u8; 33], step: U256, mini_count: usize, range_start: U256, range_end: U256, requested_device: Option<usize>, fingerprint_bits: u32, checkpoint_path: &str) -> Result<Option<GpuSearchResult>, String> {
    install_interrupt_handlers();
    println!("\n[DEBUG] Entering cuda_backend::search...");
    println!("[*] CUDA tuning: warp-batch={}, batch={} steps, block={} threads, max workers={}", WARP_BATCH_INVERSION, BATCH_SIZE, THREADS_PER_BLOCK, MAX_WORKERS);
    let device = selected_device(requested_device)?;
    let plan = make_mini_plan(range_start, range_end, step, mini_count)?;
    let mini_capacity = plan.mini_count.checked_mul(2).ok_or_else(|| "GPU mini table is too large".to_owned())?;
    let bloom_word_count = plan.mini_count.checked_mul(10).ok_or_else(|| "Bloom filter is too large".to_owned())?.div_ceil(64).checked_next_power_of_two().ok_or_else(|| "Bloom filter is too large".to_owned())? as u64;
    let slot_bytes = mini_slot_bytes(fingerprint_bits);
    println!("[GPU {device}] Mini table: {} minis, fingerprint={fingerprint_bits} bits, capacity {mini_capacity} ({:.1} MB VRAM), delta={}, trailing gap={}, max giant steps={}", plan.mini_count, mini_capacity as f64 * slot_bytes as f64 / 1024.0 / 1024.0, plan.mini_delta, plan.trailing_gap, plan.max_steps);

    // AI/maintainer note: this free edition intentionally supports one GPU only. Keep multi-GPU distribution in the paid edition.
    // Embed the cubin produced by nvcc at build time; do no runtime compilation.
    println!("[*] Loading CUDA kernel compiled during the build...");
    let ptx = Ptx::from_binary(KERNEL_CUBIN.to_vec());

    let header = make_checkpoint_header(target, step, fingerprint_bits, plan, device)?;
    let resuming = PathBuf::from(checkpoint_path).exists();
    let checkpoint = Arc::new(Mutex::new(Checkpoint::open(checkpoint_path, header, 2)?));
    println!("[*] {} checkpoint: {}", if resuming { "Resuming" } else { "New" }, checkpoint_path);
    let mut tasks = Vec::with_capacity(2);
    for (checkpoint_id, direction) in [1, -1].into_iter().enumerate() {
        let completed = checkpoint.lock().unwrap().completed[checkpoint_id];
        if completed > plan.max_steps { return Err("Checkpoint contains invalid giant-step progress".to_owned()); }
        if completed < plan.max_steps {
            tasks.push(Task { direction, start_step: completed, count: plan.max_steps - completed, checkpoint_id, checkpoint_base: completed });
        }
    }
    if tasks.is_empty() {
        println!("[*] Checkpoint indicates that the giant-step search is already complete.");
        return Ok(None);
    }
    let shared = Arc::new(SharedState { tested: AtomicU64::new(0), stop: AtomicBool::new(false), search_started: Mutex::new(None), result: Mutex::new(None), checkpoint: Arc::clone(&checkpoint) });
    let total_f64 = plan.max_steps.saturating_mul(U256::from(2u8))
        .to_string().parse::<f64>().unwrap_or(f64::MAX);
    let status_shared = Arc::clone(&shared);
    let status_thread = thread::spawn(move || {
        while !status_shared.stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(1));
            let started = *status_shared.search_started.lock().unwrap();
            let Some(started) = started else { continue; };
            let tested = status_shared.tested.load(Ordering::Relaxed);
            let elapsed = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            let speed = tested as f64 / elapsed;
            let equivalent_rate = format_bruteforce_rate(speed * mini_count as f64);
            let percent = (tested as f64 / total_f64 * 100.0).min(100.0);
            let eta = if speed > 0.0 { format_duration((total_f64 - tested as f64).max(0.0) / speed) } else { "--:--:--".to_owned() };
            print!("\r[CUDA GPU {device}] progress={percent:6.2}% | giant steps={tested} | equiv. brute-force={equivalent_rate} | ETA={eta}");
            let _ = io::stdout().flush();
        }
        println!();
    });

    let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(1);
    let worker_shared = Arc::clone(&shared);
    let worker = thread::spawn(move || {
        let result = search_device(target, step_pubkey, step, tasks, receiver, mini_capacity, bloom_word_count, plan.mini_start, plan.mini_delta, fingerprint_bits, device, ptx, worker_shared);
        if let Err(error) = &result {
            eprintln!("\n[!] CUDA ERROR on device {device}: {error}");
        }
        result
    });

    let build_started = Instant::now();
    let producer_shared = Arc::clone(&shared);
    let producer = thread::spawn(move || -> Result<(), String> {
        println!("[*] GPU {device}: generating the mini table, chunk={MINI_GENERATE_CHUNK}...");
        let secp = Secp256k1::new();
        let start_secret = SecretKey::from_byte_array(plan.mini_start.to_big_endian())
            .map_err(|error| format!("GPU {device}: invalid starting mini scalar: {error}"))?;
        let delta_secret = SecretKey::from_byte_array(plan.mini_delta.to_big_endian())
            .map_err(|error| format!("GPU {device}: invalid mini delta scalar: {error}"))?;
        let mut point = PublicKey::from_secret_key(&secp, &start_secret);
        let delta_point = PublicKey::from_secret_key(&secp, &delta_secret);
        let chunk_stride = scaled_point(&secp, delta_point, U256::from(MINI_GENERATE_CHUNK))?;
        let total_chunks = plan.mini_count.div_ceil(MINI_GENERATE_CHUNK);
        let mut loaded = 0usize;
        for first_chunk in (0..total_chunks).step_by(MINI_BUILD_BATCH) {
            if INTERRUPTED.load(Ordering::Relaxed) || producer_shared.stop.load(Ordering::Relaxed) { break; }
            let chunk_count = (total_chunks - first_chunk).min(MINI_BUILD_BATCH);
            let mut entries = Vec::with_capacity(chunk_count * MINI_SEED_BYTES);
            for offset in 0..chunk_count {
                let index = (first_chunk + offset) * MINI_GENERATE_CHUNK;
                let uncompressed = point.serialize_uncompressed();
                let mut row = [0u8; MINI_SEED_BYTES];
                row[..64].copy_from_slice(&uncompressed[1..]);
                row[64..].copy_from_slice(&(index as u64).to_le_bytes());
                entries.extend_from_slice(&row);
                if first_chunk + offset + 1 < total_chunks {
                    point = point.combine(&chunk_stride)
                        .map_err(|error| format!("GPU {device}: mini chunk stride failed at chunk {}: {error}", first_chunk + offset + 1))?;
                }
            }
            if sender.send(entries).is_err() {
                return Err("GPU mini loader stopped unexpectedly".to_owned());
            }
            let mini_batch_count = (chunk_count * MINI_GENERATE_CHUNK).min(plan.mini_count - loaded);
            loaded += mini_batch_count;
            if loaded == plan.mini_count || loaded / 5_000_000 > (loaded - mini_batch_count) / 5_000_000 {
                println!("[GPU {device}] Loaded minis: {loaded}/{}", plan.mini_count);
            }
        }
        Ok(())
    });

    let producer_result = match producer.join() {
        Ok(result) => result,
        Err(_) => Err("Mini producer thread panicked".to_owned()),
    };
    if INTERRUPTED.load(Ordering::Relaxed) { shared.stop.store(true, Ordering::Relaxed); }
    if INTERRUPTED.load(Ordering::Relaxed) {
        println!("[!] Mini table generation was interrupted; it will be rebuilt when resuming.");
    } else if producer_result.is_ok() {
        println!("[+] GPU mini points generated and inserted into the table in {:.2}s", build_started.elapsed().as_secs_f64());
    }
    if producer_result.is_ok() && !INTERRUPTED.load(Ordering::Relaxed) {
        *shared.search_started.lock().unwrap() = Some(Instant::now());
    } else {
        shared.stop.store(true, Ordering::Relaxed);
    }

    let worker_result = match worker.join() {
        Ok(result) => result,
        Err(_) => Err("CUDA worker thread panicked".to_owned()),
    };
    shared.stop.store(true, Ordering::Relaxed);
    let _ = status_thread.join();
    checkpoint.lock().unwrap().save(true)?;
    producer_result?;
    worker_result?;
    if INTERRUPTED.load(Ordering::Relaxed) { return Err(format!("Search interrupted; progress was saved to checkpoint `{checkpoint_path}`. Rerun with the same parameters to resume.")); }
    let result = shared.result.lock().unwrap().as_ref().copied();
    let elapsed_seconds = shared.search_started.lock().unwrap().map(|started| started.elapsed().as_secs_f64()).unwrap_or_default().max(f64::MIN_POSITIVE);
    Ok(result.map(|private_key| GpuSearchResult { private_key, tested: shared.tested.load(Ordering::Relaxed), elapsed_seconds, device }))
}

fn search_device(target: [u8; 33], step_pubkey: [u8; 33], step: U256, tasks: Vec<Task>, mini_batches: Receiver<Vec<u8>>, mini_capacity: usize, bloom_word_count: u64, mini_start: U256, mini_delta: U256, fingerprint_bits: u32, device: usize, ptx: Ptx, shared: Arc<SharedState>) -> Result<(), String> {
    println!("\n[DEBUG] GPU {}: Starting search_device...", device);
    let ctx = CudaContext::new(device).map_err(|e| format!("GPU {device}: context error: {e:?}"))?;
    println!("\n[DEBUG] GPU {}: Context created successfully.", device);
    let stream = ctx.default_stream();
    let module = ctx.load_module(ptx).map_err(|e| format!("GPU {device}: module error: {e:?}"))?;
    let function = module.load_function("secp256k1_search_kernel").map_err(|e| format!("GPU {device}: kernel error: {e:?}"))?;
    let insert_function = module.load_function("insert_minis_kernel").map_err(|e| format!("GPU {device}: insert kernel error: {e:?}"))?;
    println!("\n[DEBUG] GPU {}: Function loaded successfully.", device);
    let worker_result_words = MAX_WORKERS.checked_mul(12).ok_or_else(|| format!("GPU {device}: result buffer size overflow"))?;
    let mut any_candidate_device = stream.alloc_zeros::<u32>(1).map_err(|e| format!("GPU {device}: candidate flag allocation error: {e:?}"))?;
    let mut worker_results_device = stream.alloc_zeros::<u32>(worker_result_words).map_err(|e| format!("GPU {device}: result buffer allocation error: {e:?}"))?;
    let secp = Secp256k1::new();
    let target_point = PublicKey::from_slice(&target).map_err(|e| format!("Invalid target public key: {e}"))?;
    let step_point = PublicKey::from_slice(&step_pubkey).map_err(|e| format!("Invalid step public key: {e}"))?;
    let mini_delta_secret = SecretKey::from_byte_array(mini_delta.to_big_endian())
        .map_err(|e| format!("GPU {device}: invalid mini delta scalar: {e}"))?;
    let mini_delta_point = PublicKey::from_secret_key(&secp, &mini_delta_secret);
    let mini_delta_xy = point_xy(&mini_delta_point);
    let mini_delta_xy_device = stream.clone_htod(&mini_delta_xy).map_err(|e| format!("GPU {device}: mini delta point copy error: {e:?}"))?;
    let step_scalar_device = stream.clone_htod(&to_limbs(step)).map_err(|e| format!("GPU {device}: step scalar copy error: {e:?}"))?;
    let fingerprint_bytes = fingerprint_bits / 8;
    let fingerprint_table_bytes = mini_capacity.checked_mul(fingerprint_bytes as usize).ok_or_else(|| format!("GPU {device}: fingerprint table size overflow"))?;
    let index_table_bytes = mini_capacity.checked_mul(8).ok_or_else(|| format!("GPU {device}: mini index table size overflow"))?;
    let table_bytes = fingerprint_table_bytes.checked_add(index_table_bytes).ok_or_else(|| format!("GPU {device}: mini table size exceeds addressable memory"))?;
    let bloom_bytes = usize::try_from(bloom_word_count).ok().and_then(|words| words.checked_mul(8)).ok_or_else(|| format!("GPU {device}: Bloom filter size exceeds addressable memory"))?;
    let build_batch_bytes = MINI_BUILD_BATCH.checked_mul(MINI_SEED_BYTES).ok_or_else(|| format!("GPU {device}: temporary mini seed batch exceeds addressable memory"))?;
    let search_buffers_bytes = MAX_WORKERS.checked_mul(64 + 32).ok_or_else(|| format!("GPU {device}: search buffers exceed addressable memory"))?;
    // Mini upload batches and giant-search scratch buffers are used in
    // separate phases, so only the larger one contributes to peak memory.
    let phase_buffers_bytes = build_batch_bytes.max(search_buffers_bytes);
    let required_bytes = table_bytes.checked_add(bloom_bytes)
        .and_then(|n| n.checked_add(phase_buffers_bytes))
        .and_then(|n| n.checked_add(VRAM_SAFETY_RESERVE))
        .ok_or_else(|| format!("GPU {device}: total VRAM estimate exceeds addressable memory"))?;
    let (free_bytes, total_bytes) = ctx.mem_get_info().map_err(|e| format!("GPU {device}: cannot query VRAM: {e:?}"))?;
    println!("[GPU {device}] VRAM estimate: {fingerprint_bits}-bit fingerprint table {:.1} MiB + Bloom filter {:.1} MiB + maximum temporary buffers {:.1} MiB + reserve {:.0} MiB = {:.2} GiB | {:.2}/{:.2} GiB free", table_bytes as f64 / 1024.0_f64.powi(2), bloom_bytes as f64 / 1024.0_f64.powi(2), phase_buffers_bytes as f64 / 1024.0_f64.powi(2), VRAM_SAFETY_RESERVE as f64 / 1024.0_f64.powi(2), required_bytes as f64 / 1024.0_f64.powi(3), free_bytes as f64 / 1024.0_f64.powi(3), total_bytes as f64 / 1024.0_f64.powi(3));
    if free_bytes < required_bytes {
        return Err(format!("GPU {device}: not enough free VRAM for the mini table and Bloom filter (need {:.2} GiB, have {:.2} GiB free)", required_bytes as f64 / 1024.0_f64.powi(3), free_bytes as f64 / 1024.0_f64.powi(3)));
    }
    println!("[GPU {device}] Allocating and clearing the VRAM table...");
    let mut fingerprints_device = stream.alloc_zeros::<u8>(fingerprint_table_bytes).map_err(|e| format!("GPU {device}: fingerprint table allocation error: {e:?}"))?;
    let mut indices_device = stream.alloc_zeros::<u64>(mini_capacity).map_err(|e| format!("GPU {device}: mini index table allocation error: {e:?}"))?;
    println!("[GPU {device}] VRAM table ready; waiting for mini batches.");
    let mut bloom_device = stream.alloc_zeros::<u64>(bloom_word_count as usize).map_err(|e| format!("GPU {device}: Bloom filter allocation error: {e:?}"))?;
    let mini_capacity_arg = mini_capacity as u64;
    let mini_count_arg = (mini_capacity / 2) as u64;
    let mini_chunk_size = MINI_GENERATE_CHUNK as u32;
    while let Ok(batch) = mini_batches.recv() {
        let entry_count = (batch.len() / MINI_SEED_BYTES) as u64;
        let batch_device = stream.clone_htod(&batch).map_err(|e| format!("GPU {device}: mini batch copy error: {e:?}"))?;
        let config = LaunchConfig { grid_dim: (((entry_count + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut launch = stream.launch_builder(&insert_function);
        launch.arg(&mut fingerprints_device); launch.arg(&mut indices_device); launch.arg(&mini_capacity_arg); launch.arg(&mut bloom_device); launch.arg(&bloom_word_count); launch.arg(&fingerprint_bytes); launch.arg(&mini_delta_xy_device); launch.arg(&mini_chunk_size); launch.arg(&mini_count_arg); launch.arg(&batch_device); launch.arg(&entry_count);
        unsafe { launch.launch(config) }.map_err(|e| format!("GPU {device}: mini insertion error: {e:?}"))?;
        stream.synchronize().map_err(|e| format!("GPU {device}: mini insertion synchronization error: {e:?}"))?;
    }
    let mini_start_device = stream.clone_htod(&to_limbs(mini_start)).map_err(|e| format!("GPU {device}: mini start copy error: {e:?}"))?;
    let mini_delta_device = stream.clone_htod(&to_limbs(mini_delta)).map_err(|e| format!("GPU {device}: mini delta copy error: {e:?}"))?;
    println!("\n[DEBUG] GPU {}: Data copied to GPU.", device);
    for task in tasks {
        let mut current = task.start_step;
        let task_end = task.start_step + task.count;
        let directed_step_point = if task.direction < 0 { step_point.negate(&secp) } else { step_point };
        let directed_step_xy = point_xy(&directed_step_point);
        let step_xy_device = stream.clone_htod(&directed_step_xy).map_err(|e| format!("GPU {device}: directed step copy error: {e:?}"))?;
        let mut rejected_indices = Vec::<u64>::new();
        let mut rejected_indices_device = stream.clone_htod(&[0u64]).map_err(|e| format!("GPU {device}: rejected-index buffer allocation error: {e:?}"))?;
        let mut completed_batches = 0u32;
        let mut rejected_fingerprint_matches = 0u64;
        'restart_batch_group: while current < task_end && !shared.stop.load(Ordering::Relaxed) && !INTERRUPTED.load(Ordering::Relaxed) {
        let first_count = (task_end - current).min(U256::from(BATCH_SIZE)).low_u64();
        let worker_count = ((first_count + CANDIDATES_PER_WORKER - 1) / CANDIDATES_PER_WORKER).min(MAX_WORKERS as u64).max(1);
        let chunk_size = (first_count + worker_count - 1) / worker_count;
        println!("[DEBUG] GPU {device}: preparing {worker_count} starting points, chunk={chunk_size}, direction={}", task.direction);
        let base_point = add_scaled_point(&secp, target_point, directed_step_point, current)?;
        let stride_index = U256::from(chunk_size);
        let stride_point = scaled_point(&secp, directed_step_point, stride_index)?;
        let stride_adjustment = mul_mod_group(stride_index, step);
        let mut worker_points = Vec::with_capacity(worker_count as usize * 64);
        let mut worker_adjustments = Vec::with_capacity(worker_count as usize * 8);
        let mut worker_point = base_point;
        let mut worker_adjustment = mul_mod_group(current, step);
        for worker in 0..worker_count {
            let uncompressed = worker_point.serialize_uncompressed();
            worker_points.extend_from_slice(&uncompressed[1..]);
            worker_adjustments.extend_from_slice(&to_limbs(worker_adjustment));
            if worker + 1 < worker_count {
                worker_point = worker_point.combine(&stride_point)
                    .map_err(|e| format!("GPU {device}: worker stride point addition failed: {e}"))?;
                worker_adjustment = add_mod_group(worker_adjustment, stride_adjustment);
            }
        }
        let mut worker_points_device = stream.clone_htod(&worker_points).map_err(|e| format!("GPU {device}: worker points copy error: {e:?}"))?;
        let mut worker_adjustments_device = stream.clone_htod(&worker_adjustments).map_err(|e| format!("GPU {device}: worker adjustments copy error: {e:?}"))?;
        println!("[DEBUG] GPU {device}: starting points ready; blocks={}", (worker_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        let batch_gap_index = U256::from(BATCH_SIZE - chunk_size + 1);
        let batch_gap_point = if batch_gap_index.is_zero() { None } else { Some(scaled_point(&secp, directed_step_point, batch_gap_index)?) };
        let batch_gap_xy = batch_gap_point.as_ref().map(point_xy).unwrap_or([0u8; 64]);
        let batch_gap_scalar = mul_mod_group(batch_gap_index, step);
        let batch_gap_scalar_limbs = to_limbs(batch_gap_scalar);
        let batch_gap_xy_device = stream.clone_htod(&batch_gap_xy).map_err(|e| format!("GPU {device}: batch gap point copy error: {e:?}"))?;
        let batch_gap_scalar_device = stream.clone_htod(&batch_gap_scalar_limbs).map_err(|e| format!("GPU {device}: batch gap scalar copy error: {e:?}"))?;
        let threads = THREADS_PER_BLOCK;
        let blocks = (worker_count + threads - 1) / threads;
        let mut first_batch = 1u32;
        while current < task_end && !shared.stop.load(Ordering::Relaxed) && !INTERRUPTED.load(Ordering::Relaxed) {
            let remaining = task_end - current;
            let count = if remaining > U256::from(BATCH_SIZE) { BATCH_SIZE } else { remaining.low_u64() };
            stream.memset_zeros(&mut any_candidate_device).map_err(|e| format!("GPU {device}: candidate flag reset error: {e:?}"))?;
            stream.memset_zeros(&mut worker_results_device).map_err(|e| format!("GPU {device}: worker result reset error: {e:?}"))?;
            if first_batch != 0 { println!("[DEBUG] GPU {device}: starting first batch, count={count}"); }
            let config = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 };
            let mut launch = stream.launch_builder(&function);
            let bloom_enabled = 1u32;
            let rejected_count = rejected_indices.len() as u64;
            launch.arg(&mut worker_points_device); launch.arg(&step_xy_device); launch.arg(&mut worker_adjustments_device); launch.arg(&step_scalar_device); launch.arg(&count); launch.arg(&worker_count); launch.arg(&chunk_size); launch.arg(&fingerprints_device); launch.arg(&indices_device); launch.arg(&mini_capacity_arg); launch.arg(&fingerprint_bytes); launch.arg(&mut bloom_device); launch.arg(&bloom_word_count); launch.arg(&bloom_enabled); launch.arg(&mini_start_device); launch.arg(&mini_delta_device); launch.arg(&task.direction); launch.arg(&first_batch); launch.arg(&batch_gap_xy_device); launch.arg(&batch_gap_scalar_device); launch.arg(&rejected_indices_device); launch.arg(&rejected_count); launch.arg(&mut any_candidate_device); launch.arg(&mut worker_results_device);
            unsafe { launch.launch(config) }.map_err(|e| format!("GPU {device}: launch error: {e:?}"))?;
            first_batch = 0;
            let mut earliest: Option<(u64, U256, u64)> = None;
            let has_candidate = stream.clone_dtoh(&any_candidate_device).map_err(|e| format!("GPU {device}: candidate flag copy error: {e:?}"))?.first().copied().unwrap_or(0) != 0;
            if has_candidate {
                let result = stream.clone_dtoh(&worker_results_device).map_err(|e| format!("GPU {device}: worker results copy error: {e:?}"))?;
                for worker in 0..worker_count as usize {
                    let row = &result[worker * 12..worker * 12 + 12];
                    if row[0] == 0 { continue; }
                    let offset = worker as u64 * chunk_size + row[9] as u64;
                    let mini_index = row[10] as u64 | ((row[11] as u64) << 32);
                    let candidate = (offset, from_limbs(&row[1..9]), mini_index);
                    if earliest.as_ref().is_none_or(|current| candidate.0 < current.0) { earliest = Some(candidate); }
                }
            }
            if let Some((offset, private_key, mini_index)) = earliest {
                let candidate_valid = SecretKey::from_byte_array(private_key.to_big_endian()).ok()
                    .is_some_and(|secret| PublicKey::from_secret_key(&secp, &secret).serialize() == target);
                if candidate_valid {
                    if !shared.stop.swap(true, Ordering::Relaxed) { *shared.result.lock().unwrap() = Some(private_key); }
                    return Ok(());
                }
                rejected_fingerprint_matches += 1;
                if rejected_fingerprint_matches == 1 || rejected_fingerprint_matches % 1000 == 0 {
                    println!("[GPU {device}] Rejected false fingerprint matches: {rejected_fingerprint_matches}; continuing search.");
                }
                if shared.stop.load(Ordering::Relaxed) { return Ok(()); }
                if offset >= count { return Err(format!("GPU {device}: fingerprint match is outside the active batch")); }
                if offset != 0 {
                    shared.tested.fetch_add(offset, Ordering::Relaxed);
                    current += U256::from(offset);
                    rejected_indices.clear();
                }
                rejected_indices.push(mini_index);
                rejected_indices_device = stream.clone_htod(&rejected_indices).map_err(|e| format!("GPU {device}: rejected-index upload error: {e:?}"))?;
                let completed = task.checkpoint_base + (current - task.start_step);
                shared.checkpoint.lock().unwrap().update(task.checkpoint_id, completed, false)?;
                continue 'restart_batch_group;
            }
            shared.tested.fetch_add(count, Ordering::Relaxed);
            current += U256::from(count);
            rejected_indices.clear();
            completed_batches += 1;
            let interrupted = INTERRUPTED.load(Ordering::Relaxed);
            let task_done = current >= task_end;
            let completed = task.checkpoint_base + (current - task.start_step);
            shared.checkpoint.lock().unwrap().update(task.checkpoint_id, completed, interrupted || task_done || completed_batches % 16 == 0)?;
            if interrupted {
                shared.stop.store(true, Ordering::Relaxed);
                break;
            }
        }
        }
    }
    Ok(())
}

fn to_limbs(value: U256) -> [u32; 8] {
    let bytes = value.to_big_endian();
    let mut limbs = [0u32; 8];
    for (index, limb) in limbs.iter_mut().enumerate() { let offset = 32 - (index + 1) * 4; *limb = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()); }
    limbs
}

fn point_xy(point: &PublicKey) -> [u8; 64] {
    let uncompressed = point.serialize_uncompressed();
    let mut xy = [0u8; 64];
    xy.copy_from_slice(&uncompressed[1..]);
    xy
}

fn scaled_point(
    secp: &Secp256k1<secp256k1::All>,
    base: PublicKey,
    index: U256,
) -> Result<PublicKey, String> {
    let scalar = index % GROUP_ORDER;
    if scalar.is_zero() {
        return Err("A zero scalar produces the point at infinity".to_owned());
    }
    let tweak = Scalar::from_be_bytes(scalar.to_big_endian())
        .map_err(|error| format!("Invalid CUDA point multiplier: {error}"))?;
    base.mul_tweak(secp, &tweak)
        .map_err(|error| format!("CUDA point multiplication failed: {error}"))
}

fn add_scaled_point(
    secp: &Secp256k1<secp256k1::All>,
    target: PublicKey,
    base: PublicKey,
    index: U256,
) -> Result<PublicKey, String> {
    let scalar = index % GROUP_ORDER;
    if scalar.is_zero() {
        return Ok(target);
    }
    target.combine(&scaled_point(secp, base, scalar)?)
        .map_err(|error| format!("CUDA initial point addition failed: {error}"))
}

fn mul_mod_group(a: U256, b: U256) -> U256 {
    let product = (U512::from(a) * U512::from(b)) % U512::from(GROUP_ORDER);
    let bytes = product.to_big_endian();
    U256::from_big_endian(&bytes[32..])
}

fn add_mod_group(a: U256, b: U256) -> U256 {
    let sum = (U512::from(a) + U512::from(b)) % U512::from(GROUP_ORDER);
    let bytes = sum.to_big_endian();
    U256::from_big_endian(&bytes[32..])
}

fn from_limbs(limbs: &[u32]) -> U256 {
    let mut bytes = [0u8; 32];
    for (index, limb) in limbs.iter().take(8).enumerate() { let offset = 32 - (index + 1) * 4; bytes[offset..offset + 4].copy_from_slice(&limb.to_be_bytes()); }
    U256::from_big_endian(&bytes)
}

fn format_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", seconds / 3600, (seconds / 60) % 60, seconds % 60)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trip_and_configuration_mismatch_are_detected() {
        let path = std::env::temp_dir().join(format!("btc-solver-test-{}.checkpoint", std::process::id()));
        let _ = fs::remove_file(&path);
        let mut checkpoint = Checkpoint::open(path.to_str().unwrap(), b"test-header".to_vec(), 2).unwrap();
        checkpoint.update(0, U256::from(17), true).unwrap();
        checkpoint.update(1, U256::from(29), true).unwrap();
        drop(checkpoint);

        let loaded = Checkpoint::open(path.to_str().unwrap(), b"test-header".to_vec(), 2).unwrap();
        assert_eq!(loaded.completed, [U256::from(17), U256::from(29)]);
        assert!(Checkpoint::open(path.to_str().unwrap(), b"different-header".to_vec(), 2).is_err());
        fs::remove_file(path).unwrap();
    }
}
