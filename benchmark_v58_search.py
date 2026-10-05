#!/usr/bin/env python3
"""Measure v58 host-side search-model overhead on an existing checkpoint."""
import argparse
import time
import main

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default="minimal_gp_checkpoint.npz")
ap.add_argument("--linkage-elites", type=int, default=96)
ap.add_argument("--linkage-loci", type=int, default=128)
ap.add_argument("--linkage-modules", type=int, default=96)
ap.add_argument("--linkage-max-module", type=int, default=16)
ap.add_argument("--qd-bins", type=int, default=6)
a = ap.parse_args()

sampler = main.CorpusSampler([b"benchmark corpus " * 64])
t0 = time.perf_counter()
gen, pop, _best, _history, _cfg, hof = main.load_checkpoint(a.checkpoint, sampler)
load_s = time.perf_counter() - t0
pop.sort(key=lambda g: g.fitness, reverse=True)

link = main.SparseLinkageModel()
t0 = time.perf_counter()
link.refresh(pop, gen, 1, a.linkage_elites, a.linkage_loci, a.linkage_modules, a.linkage_max_module)
link_s = time.perf_counter() - t0

t0 = time.perf_counter()
grid = main.build_qd_grid(pop, a.qd_bins)
qd_s = time.perf_counter() - t0

print(f"checkpoint generation={gen} population={len(pop)} rules={len(pop[0].rules) if pop else 0} hof={len(hof)} load={load_s:.3f}s")
print(f"linkage modules={len(link.modules)} loci={link.loci} build={link_s:.6f}s")
print(f"qd cells={len(grid)} build={qd_s:.6f}s")
