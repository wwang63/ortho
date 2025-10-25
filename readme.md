# Orthogonal DNA Library generator

This repository consists of 4 main files. 

1. orthogonal.py defines all general functions used. 
2. duplex_generator.py acts as a template for a script to generate a duplex orthogonal library. 
3. single_strand_generator.py acts as a template for a script to generate a single stranded orthogonal library. 
4. interactive.py acts as a barebone interactive python terminal line program to generate orthogonal libraries. 

The attached pdf ortho.pdf describes the methods used in generating a duplex library.

There may well be bugs! Please email me at jspaul@caltech.edu to report any bugs (or if you have any suggestions :) )!

## Runtime environment

The generator scripts automatically configure Python's ``multiprocessing`` module to use the
``forkserver`` start method when it is supported (for example on most Linux distributions).
Platforms that do not expose ``forkserver``—notably macOS—fall back to ``spawn`` and continue
running without additional setup, so no manual start-method tweaks are required before
launching the pipelines.

## Similarity benchmark

To quantify the speed-up of the NumPy-backed similarity matrix builder, run the benchmark
driver:

```
python benchmarks/similarity_benchmark.py --size 150 --length 20 --duplex
```

The script compares the optimised implementation against a naive Python baseline, verifies
that they produce identical results, and prints a timing table alongside the average
per-run durations and resulting speed-up factor.
