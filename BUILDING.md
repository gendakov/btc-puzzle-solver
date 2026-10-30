# Building and installation

See [README.md](README.md) for the project overview and command-line reference. This guide covers build requirements, CUDA tuning, and runtime details.

This working tree is the free edition. CUDA searches use one GPU per run. The `max`, `--sector`, and `--sector-size-bits` options remain visible in help, but are paid features: using them prints the EUR 30 payment instructions and stops before searching. See [README.md](README.md) for the BTC, LTC, and ETH addresses and order instructions.

Build from the repository root. Cargo downloads dependencies on the first build unless they are already cached locally.

## System requirements

- Linux x86_64
- Rust stable toolchain with Cargo
- NVIDIA driver and CUDA Toolkit, including `nvcc`
- An NVIDIA GPU supported by the installed driver

The CUDA build targets `sm_86` (RTX 3060 Ti) by default. To target another GPU architecture, change `-arch=sm_86` in `build.rs` to the appropriate compute capability.

## Build

```bash
cargo build --release --locked
```

The build script compiles `cuda/secp256k1_search.cu` into a cubin with `nvcc` and embeds it in the executable. A CUDA toolkit and `nvcc` are required even when you intend to use CPU mode. No prebuilt binary or CUDA library from the source computer is required.

To compile the experimental warp batch-inversion kernel:

```bash
CUDA_WARP_BATCH=1 cargo build --release --locked
```

The CUDA search launch can be tuned at build time. Defaults are 1,048,576 giant steps per batch, 128 threads per block, and at most 65,536 workers. Supported settings:

```bash
CUDA_BATCH_SIZE=4194304 cargo build --release --locked
CUDA_THREADS_PER_BLOCK=256 cargo build --release --locked
CUDA_MAX_WORKERS=131072 cargo build --release --locked
CUDA_WARP_BATCH=1 CUDA_BATCH_SIZE=4194304 CUDA_THREADS_PER_BLOCK=256 cargo build --release --locked
```

`CUDA_BATCH_SIZE` accepts powers of two from 1,048,576 through 16,777,216. `CUDA_THREADS_PER_BLOCK` accepts 64, 128, 256, or 512. `CUDA_MAX_WORKERS` accepts 16,384, 32,768, 65,536, or 131,072. The selected values are printed when CUDA search starts. The progress display reports an estimated brute-force-equivalent candidate rate: aggregate giant-step rate multiplied by the average number of minis per GPU. This is a comparison metric, not the rate of literal scalar-key derivations. Use a separate `--checkpoint` path for each benchmark run.

## Running the solver

```bash
./target/release/btc_puzzle_solver --help
./target/release/btc_puzzle_solver 49 1000000 --gpu
./target/release/btc_puzzle_solver 140 max 100 --gpu -o found_private_key.txt
./target/release/btc_puzzle_solver 49 100000000 --workers 8
./target/release/btc_puzzle_solver 49 1000000 --gpu --checkpoint puzzle49.chk
```

**Paid feature:** to search one of `N` equal parts of a puzzle's private-key range, add `--sector K/N`, where `K` is between `1` and `N`. To process a consecutive range, use `--sector A-B/N`; for example, `3-5/10` searches sectors `3/10`, `4/10`, and `5/10` in order. Each sector gets its own checkpoint, and the next sector starts after the previous one finishes without a match. A verified key stops the sequence. The requested number of minis is generated across each sector's range, so all minis are concentrated in that part. For example, `1/1000` selects the first of 1,000 parts, while `1000/10000` selects part 1,000 of 10,000. In this free edition these commands print payment information and do not start a search:

```bash
./target/release/btc_puzzle_solver 140 max 100 --gpu --sector 1/1000
./target/release/btc_puzzle_solver 140 max 100 --gpu --sector 1000/10000
./target/release/btc_puzzle_solver 70 max --gpu --sector 3-5/10
```

**Paid feature:** to split the range into fixed-size chunks of `2^B` keys, use `--sector-size-bits B` and select chunk numbers with `--sector`. For Puzzle #140, chunks of `2^70` keys can be selected like this. In this free edition the command prints payment information and does not start a search:

```bash
./target/release/btc_puzzle_solver 140 max --gpu --sector 1 --sector-size-bits 70
```

This searches one chunk containing `2^70` keys, equal to twice the key range of Puzzle #70. Chunk numbers are 1-based and counted from the beginning of the puzzle range. The final chunk is shortened if the range ends before it is full. Each chunk gets its own checkpoint; processing stops when a key is found. The startup summary prints the exact key count for the selected chunk.

Each sector gets a separate default checkpoint file that includes both `K` and `N`. For a multi-sector run, an explicit `--checkpoint FILE` is also given a sector-specific suffix for each sector.

CUDA fingerprint size can be selected with `--fp N` (or the long alias `--fingerprint-bits N`), where `N` is 8, 16, 24, ... up to 192. The default is 192 bits. Shorter fingerprints use less VRAM per hash-table slot, but raise the chance that an unrelated point is reported as a match. In the paid edition, `max` recalculates capacity from the selected `--fp`, so shorter fingerprints can fit more minis in the same VRAM. At 8 bits, random collisions are expected very frequently for large tables. Every fingerprint candidate is checked by deriving its public key on the CPU. If validation fails, CUDA excludes that mini index at the current giant point and continues the same search, including checking other minis with the same short fingerprint. Use 192 bits for the lowest false-match rate and CPU validation overhead.
The false-match rate depends on Bloom-filter passes, hash-table probe lengths, and the selected bit count. Very short fingerprints can require many CPU validations, although those failed validations no longer stop the search.

Example using an 8-bit fingerprint:

```bash
./target/release/btc_puzzle_solver 140 max 100 --gpu --fp 8
```


Use `--clear` to delete all solver checkpoint/progress files in the current folder before starting a new run. It can also be used by itself to clear the files and exit:

```bash
./target/release/btc_puzzle_solver --clear
./target/release/btc_puzzle_solver 140 max 100 --gpu --sector 1/1000 --clear
```

## GPU memory

- In CUDA mode, the CPU computes a starting public point for each chunk of 256 minis and sends those seeds in bounded batches. The selected GPU generates the points in its chunks and inserts them directly into VRAM, avoiding transfer of every mini public key. The fingerprint bytes and aligned 64-bit mini indices use separate VRAM arrays, so each 8-bit reduction saves one byte per table entry. Table capacity is twice the mini count, without power-of-two rounding. A reported match is still checked against the full public key before the key is accepted.
- CUDA searches use one selected GPU per run. GPU 0 is selected by default; `--gpu-device N` selects another device.
- **Paid feature:** `max` selects the mini capacity for one GPU, including its search/build buffers and safety reserve. Actual capacity depends on free VRAM at startup. The free edition prints payment information and exits when `max` is requested.
- A blocked Bloom filter is built in VRAM during insertion and rejects most misses before exact hash lookup.
- CUDA stores each mini as its compressed public key and 64-bit sequence index; the private scalar is reconstructed only on a match.
- The CPU producer creates chunk seeds and the selected GPU generates its elliptic-curve point sequence; one host producer thread feeds the GPU. CPU mode uses `--workers N` for mini generation.
- CUDA giant-step coverage includes both the regular mini spacing and any remainder after the last mini. Checkpoints created by earlier builds are rejected when their search extent differs; use `--clear` before restarting an old run with the corrected coverage.
- `--benchmark` measures the CPU lookup table.
- `-o FILE` (or `--output FILE`) writes the verified private key as 64 lowercase hexadecimal digits and a newline. The file is created only after a match is verified and an existing file with that name is overwritten.
- CUDA search writes progress to an automatic checkpoint in the current directory. `--checkpoint FILE` selects another path. Pressing Ctrl+C or sending SIGTERM lets the current GPU batch finish and saves its progress; rerun with the same arguments and device set to resume. Mini shards are regenerated before the saved giant-step ranges continue. Delete the checkpoint to start that search from the beginning. CPU mode does not use this checkpoint.

The puzzle data file `btc_puzzles_160_with_pubkey.json` is included in the repository.
