# Microbenchmarking gather instructions on Intel CPUs

## Build

```bash
git clone https://github.com/quim0/gather-microbenchmark.git
cd gather-microbenchmark
cmake -S . -B build
cmake --build build -j
```

Then, the binary is generated in the `bin` directory.

```bash
./bin/gathermicrobench
```