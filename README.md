# Bitcoin Puzzle Solver

A Rust command-line solver for the public Bitcoin Puzzle challenge ranges. It searches for a private scalar whose secp256k1 public key matches a puzzle's known public key. It supports a parallel CPU search and a CUDA search for NVIDIA GPUs.

The included data file is `btc_puzzles_160_with_pubkey.json`. Puzzle ranges use inclusive start and end values. The solver needs a known public key, so entries whose `pubkey` is `null` cannot currently be searched.

## How it works

The solver uses a baby-step giant-step style search. It creates a table of public points for evenly spaced private scalars, then walks the target point by multiples of the selected step and looks for a table match. Any candidate private key is derived back to a public key and checked before it is reported as found.

- **CPU mode** builds the lookup table in system RAM and searches it with Rayon workers.
- **CUDA mode** builds a mini table on one selected NVIDIA GPU and searches it with a CUDA kernel. GPU 0 is selected by default; `--gpu-device N` selects another single device.
- **Paid options** include automatic VRAM capacity (`max`) and sector/chunk searches. They remain visible in `--help`; this free build displays the manual BTC/LTC/ETH payment instructions instead of running them.

This software is intended for the published puzzle ranges in the bundled data. Do not use it to search for keys to wallets you do not own or have permission to test.

## Requirements

- Linux x86_64
- Rust stable and Cargo
- CUDA Toolkit with `nvcc` available at build time
- For GPU searches: a compatible NVIDIA GPU and driver

`build.rs` compiles the CUDA kernel for `sm_86` by default, even when you plan to run CPU mode. For a different GPU architecture, adjust the `-arch=sm_86` argument in `build.rs` before building. See [BUILDING.md](BUILDING.md) for build tuning and additional setup details.

## Build

```bash
cargo build --release --locked
```

The executable is `target/release/btc_puzzle_solver`.

## Quick start

```bash
# Show all command-line options
./target/release/btc_puzzle_solver --help

# Search puzzle 49 on CPU with one million minis
./target/release/btc_puzzle_solver 49 1000000

# Search puzzle 49 on CUDA with one million minis
./target/release/btc_puzzle_solver 49 1000000 --gpu

# Select one GPU device explicitly
./target/release/btc_puzzle_solver 49 1000000 --gpu --gpu-device 0
```

The program expects to be run from the project directory because it loads `btc_puzzles_160_with_pubkey.json` by that relative path.

## Command-line reference

```text
btc_puzzle_solver <puzzle> <num_minis|max> [step] [options]
```

| Argument or option | Description |
| --- | --- |
| `<puzzle>` | Puzzle number from the bundled JSON data. |
| `<num_minis>` | Number of evenly spaced mini points; must be at least 2. |
| `max` | **Paid:** CUDA only; automatically calculate mini capacity from free VRAM. In the free build this displays the manual payment instructions and stops. |
| `[step]` | Giant-step size, in the range's scalar units; defaults to `1`. |
| `--gpu` | Use CUDA instead of CPU search. |
| `--gpu-device N` | Use only CUDA device index `N`; GPU 0 is selected by default. CUDA uses one GPU in this version. |
| `--workers N` | CPU worker count. The value is clamped to the available CPU parallelism. CUDA uses one mini producer. |
| `--sector K/N` | **Paid:** search sector `K` from `N` equal parts; sectors are 1-based. The free build displays the manual payment instructions and stops. |
| `--sector A-B/N` | **Paid:** search sectors `A` through `B` in order, stopping if a key is found. |
| `--sector A-B --sector-size-bits B` | **Paid:** search fixed-size chunks of `2^B` keys, numbered from the start of the puzzle range. The last chunk may be shorter. |
| `--fp N`, `--fingerprint-bits N` | CUDA fingerprint size from 8 to 192 bits in increments of 8; default is 192. |
| `--checkpoint FILE` | Set the CUDA checkpoint path. By default a path is generated from the puzzle, mini count, sector, and step. |
| `--clear` | Remove checkpoint/progress files in the current directory before the search. By itself, clears them and exits. |
| `-o FILE`, `--output FILE` | Write a verified private key as 64 lowercase hexadecimal digits and a newline. An existing file is overwritten. |
| `--show-minis` | Print the first 10 generated mini private scalars. |
| `--benchmark` | Benchmark the CPU lookup table. This option has no effect in CUDA mode. |
| `--profile FILE` | Write total mini-generation time, search time, and worker count after the run. |
| `-h`, `--help` | Show command-line help. |

`--sector` and `--sector-size-bits` are mutually exclusive modes in the paid version. `--sector-size-bits` requires `--sector`; accepted bit counts are 0 through 255. For equal sectors, the paid version processes each requested sector sequentially, with its own checkpoint. For fixed-size chunks, each chunk also gets a distinct checkpoint. An explicit checkpoint path receives a sector/chunk suffix when multiple ranges are requested.

### Paid version and payment

The paid version, **BTC Puzzle Solver Pro**, costs **EUR 30** as a one-time payment. Send the equivalent amount in BTC, LTC, or ETH to the corresponding address:

| Currency | Payment address |
| --- | --- |
| BTC | `bc1qv8dlms8qw7v6lx7p2h9p84xd2yyvrckj08xlm4` |
| LTC | `LUM12iDExStPQexVr5XJfY8kXPmwfe2PBi` |
| ETH | `0x21C65E8a759b7d04FaC964d832cE4d70B4C141a` |

The product name is `BTC Puzzle Solver Pro`. Standard BTC and LTC address transfers do not provide a reliable order memo. Email the transaction ID, product name, and delivery email to **gendakov@gmail.com**. Ethereum wallets may offer a transaction data field, but transaction data is public and permanent, so do not put your email address there. After payment is verified, the product will be sent to the email address you provide. Check the network and address carefully before sending; cryptocurrency transfers are generally irreversible.

The free program prints these payment details when a paid-only option is requested. The paid options remain unavailable in the free build.

## CUDA notes

- This version uses one GPU device per run. GPU 0 is selected by default; use `--gpu-device N` to select another device.
- In the paid version, the `max` capacity calculation includes the table, Bloom filter, search/build buffers, and safety reserves. Actual capacity depends on free VRAM when the program starts.
- Short fingerprints reduce table memory, but increase unrelated fingerprint matches and CPU validation work. The full public key is checked before accepting a candidate. Use 192 bits for the lowest false-match and validation rate.
- The reported brute-force-equivalent rate is an estimate: giant-step rate multiplied by the number of minis on the GPU. It is not a count of literal private-key derivations.
- CUDA checkpoint files are tied to the target puzzle, step, fingerprint size, mini layout, and selected device. Keep the same arguments and device to resume. Ctrl+C or SIGTERM lets the active GPU batch finish and saves progress; the mini table is rebuilt on resume. CPU mode does not use checkpoints.

For CUDA build-time tuning options and detailed memory behavior, see [BUILDING.md](BUILDING.md).

## Puzzle data

The bundled JSON records puzzle number, bit length, inclusive scalar range, address, and a compressed public key when available. Its metadata lists the data sources. Since the solver searches by public key, puzzles without a known `pubkey` are not currently supported.

## License

This free edition is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
