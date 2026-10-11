"""Exact parallel C++ CPU trajectory backend for Lens Spearman GP.

No torch required. Compiles the shipped C++ source on first use and caches the
shared object outside the source tree. Set CXX to choose a compiler.
"""
import ctypes
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np


@dataclass
class PackedInputs:
    tokens: np.ndarray
    lengths: np.ndarray


def prepare_inputs(inputs):
    length = len(inputs)
    width = max(1, max((len(x) for x in inputs), default=0))
    raw = np.zeros((length, width), dtype=np.int32)
    lengths = np.fromiter((len(x) for x in inputs), dtype=np.int32, count=length)
    for i, seq in enumerate(inputs):
        if len(seq):
            raw[i, :len(seq)] = seq
    return PackedInputs(raw, lengths)


def pack_rules(genome):
    meta = np.zeros((len(genome.rules), 4), dtype=np.int32)
    flat = []
    for i, rule in enumerate(genome.rules):
        meta[i, 0] = len(flat)
        meta[i, 1] = len(rule.pattern)
        flat.extend(rule.pattern)
        meta[i, 2] = len(flat)
        meta[i, 3] = len(rule.replacement)
        flat.extend(rule.replacement)
    return meta, np.asarray(flat if flat else [0], dtype=np.int32)


_LIB = None
_RULE_CACHE = OrderedDict()
_RULE_CACHE_LIMIT = 96
CPU_THREADS_DEFAULT = max(1, int(os.environ.get('REPLACER_CPU_THREADS') or min(8, os.cpu_count() or 1)))

def _cached_pack(genome):
    """Per-process bounded, identity-checked pack; no unbounded HoF ownership."""
    key = id(genome)
    cached = _RULE_CACHE.get(key)
    if cached is not None:
        owner, rule_refs, arrays = cached
        if owner is genome and len(rule_refs) == len(genome.rules) and all(
            a is b for a, b in zip(rule_refs, genome.rules)
        ):
            _RULE_CACHE.move_to_end(key)
            return arrays
    arrays = pack_rules(genome)
    _RULE_CACHE[key] = (genome, tuple(genome.rules), arrays)
    _RULE_CACHE.move_to_end(key)
    if len(_RULE_CACHE) > _RULE_CACHE_LIMIT:
        _RULE_CACHE.popitem(last=False)
    return arrays


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    cpp = Path(__file__).with_name('replacer_native_cpu.cpp')
    compiler = os.environ.get('CXX') or ('clang++' if sys.platform == 'darwin' else 'g++')
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-shared']
    flags += ['-pthread']  # portable C++17 std::thread pool on macOS and Linux
    digest = hashlib.sha256(cpp.read_bytes() + (' '.join([compiler]+flags)).encode()).hexdigest()[:16]
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    target = Path(tempfile.gettempdir()) / f'lens_native_cpu_{digest}{suffix}'
    if not target.exists():
        temp = target.with_name(target.name + f'.{os.getpid()}.tmp')
        command = [compiler, *flags, str(cpp), '-o', str(temp)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(temp, target)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f'CPU native compilation failed: {exc}\n'
                               f'{getattr(exc, "stderr", "")}\nCommand: {command}') from exc
        finally:
            temp.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(target))
    nd = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    nf = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    nd64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.replacer_cpu_batch.argtypes = [nd, nd, ctypes.c_int, ctypes.c_int,
                                       nd, nd, nd, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, nf, nd64,
                                       nd, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.replacer_cpu_batch.restype = ctypes.c_int
    lib.replacer_cpu_prefix_batch.argtypes = [nd, nd, ctypes.c_int, ctypes.c_int,
                                             nd, nd, nd, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, nd, nd, ctypes.c_int]
    lib.replacer_cpu_prefix_batch.restype = ctypes.c_int
    lib.replacer_cpu_infer_create.argtypes = [nd, nd, nd, nd, nd64,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.replacer_cpu_infer_create.restype = ctypes.c_void_p
    lib.replacer_cpu_infer_eval.argtypes = [ctypes.c_void_p, nd, nd, ctypes.c_int, ctypes.c_int, nd64]
    lib.replacer_cpu_infer_eval.restype = ctypes.c_int
    lib.replacer_cpu_outer_eval.argtypes = [ctypes.c_void_p, nd, nd,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            nf, nd64, nd]
    lib.replacer_cpu_outer_eval.restype = ctypes.c_int
    lib.replacer_cpu_infer_destroy.argtypes = [ctypes.c_void_p]
    lib.replacer_cpu_infer_destroy.restype = None
    _LIB = lib
    return lib


def trajectory_features_native(inputs, genome, max_output, hidden_rewrite_phases=2,
                               cpu_threads=None, return_states=True, prepared_rules=None):
    raw = prepare_inputs(inputs) if not isinstance(inputs, PackedInputs) else inputs
    if prepared_rules is None:
        prepared_rules = _cached_pack(genome)
    meta, tokens = prepared_rules
    n = len(raw.lengths)
    n_rules = len(genome.rules)
    features = np.zeros((n, n_rules), dtype=np.float32)
    baselines = np.zeros(n, dtype=np.float64)
    stats = np.zeros((n, 4), dtype=np.int32)
    states = np.zeros((n, max(raw.tokens.shape[1], int(max_output))), dtype=np.int32) if return_states else None
    lengths = np.zeros(n, dtype=np.int32) if return_states else None
    emb = np.asarray(genome.embedding, dtype=np.int32)
    # Each trajectory is independent and has its own hidden-state arena.
    threads = max(1, int(CPU_THREADS_DEFAULT if cpu_threads is None else cpu_threads))
    ret = _load_lib().replacer_cpu_batch(raw.tokens, raw.lengths, n, raw.tokens.shape[1],
                                         emb, meta, tokens, n_rules, int(max_output),
                                         int(hidden_rewrite_phases), threads,
                                         features, baselines, stats,
                                         lengths.ctypes.data if return_states else None,
                                         states.ctypes.data if return_states else None,
                                         states.shape[1] if return_states else 0)
    if ret:
        raise RuntimeError(f'CPU native evaluator failed with error {ret}')
    results = [states[i,:lengths[i]].tolist() for i in range(n)] if return_states else []
    return features, baselines, stats, results


def prefix_states_native(inputs, genome, stop_before_rule, max_output,
                         hidden_rewrite_phases=2, cpu_threads=None):
    """Exact first-sweep visible states before the selected rule boundary.

    The full original rule program defines hidden-phase scheduling, even
    though this interpreter stops at a prefix. These are *internal token IDs*;
    never feed them through the input Embedding LUT again.
    """
    raw = prepare_inputs(inputs) if not isinstance(inputs,PackedInputs) else inputs
    meta,tokens = _cached_pack(genome)
    n=len(raw.lengths)
    stride=max(raw.tokens.shape[1],int(max_output))
    out=np.zeros((n,stride),dtype=np.int32)
    lengths=np.zeros(n,dtype=np.int32)
    ret=_load_lib().replacer_cpu_prefix_batch(
        raw.tokens,raw.lengths,n,raw.tokens.shape[1],
        np.asarray(genome.embedding,dtype=np.int32),meta,tokens,len(genome.rules),
        int(max_output),int(hidden_rewrite_phases),
        max(1,int(CPU_THREADS_DEFAULT if cpu_threads is None else cpu_threads)),
        int(stop_before_rule),lengths,out,stride)
    if ret:
        raise RuntimeError(f'CPU native prefix snapshot error {ret}')
    return [tuple(out[i,:lengths[i]].tolist()) for i in range(n)]


class NativeInferenceContext:
    """One compiled C++ model ensemble shared across inference GA rounds.

    Both candidate trajectory execution and fitted readout reduction run in
    C++; a single native job queue spans every (model,candidate) pair.
    Keep the context alive while candidates evolve; call close() when done.
    """
    def __init__(self, genomes, max_output, hidden_rewrite_phases=2, cpu_threads=None):
        if not genomes:
            raise ValueError('native inference context requires at least one genome')
        self._ptr = None  # Safe even if construction fails before __del__.
        self.model_count = len(genomes)
        # Exact identity duplicates need one native model slot, not two.
        # The ensemble may contain both population and Hall-of-Fame references.
        # Avoid costly whole-genome fingerprinting on the common unique path.
        by_identity = {}
        unique_genomes = []
        positions = []
        for genome in genomes:
            key = id(genome)
            idx = by_identity.get(key)
            if idx is None:
                idx = len(unique_genomes)
                unique_genomes.append(genome)
                by_identity[key] = idx
            positions.append(idx)
        self._model_row_map = (np.asarray(positions, dtype=np.intp)
                               if len(unique_genomes) != len(genomes) else None)
        self._native_model_count = len(unique_genomes)
        self._native_feature_width = max(len(g.rules) for g in unique_genomes)
        offsets = [0]
        meta_parts = []
        token_parts = []
        emb_parts = []
        weight_parts = []
        token_offset = 0
        for genome in unique_genomes:
            meta, tokens = _cached_pack(genome)
            local_meta = meta.copy()
            local_meta[:, (0, 2)] += token_offset
            meta_parts.append(local_meta)
            token_parts.append(tokens)
            token_offset += len(tokens)
            nr = len(genome.rules)
            offsets.append(offsets[-1]+nr)
            emb_parts.append(np.asarray(genome.embedding[:256], dtype=np.int32))
            if len(genome.readout_weights) == nr:
                weights = np.asarray(genome.readout_weights, dtype=np.float64)
                # Match _prediction_scores_from_features: exclude tiny or NaN weights.
                weight_parts.append(np.where(np.abs(weights) > 1.0e-12, weights, 0.0))
            else:
                weight_parts.append(np.zeros(nr, dtype=np.float64))
        meta = np.ascontiguousarray(np.concatenate(meta_parts, axis=0), dtype=np.int32)
        tokens = np.ascontiguousarray(np.concatenate(token_parts), dtype=np.int32)
        weights = np.ascontiguousarray(np.concatenate(weight_parts), dtype=np.float64)
        embeddings = np.ascontiguousarray(np.concatenate(emb_parts), dtype=np.int32)
        offsets = np.ascontiguousarray(offsets, dtype=np.int32)
        self._lib = _load_lib()
        self._ptr = self._lib.replacer_cpu_infer_create(
            embeddings, offsets, meta, tokens, weights,
            self._native_model_count, int(max_output), int(hidden_rewrite_phases),
            max(1, int(CPU_THREADS_DEFAULT if cpu_threads is None else cpu_threads)),
        )
        if not self._ptr:
            raise RuntimeError('unable to construct native inference C++ ensemble')

    def score(self, candidates):
        if not self._ptr:
            raise RuntimeError('inference context is closed')
        packed = candidates if isinstance(candidates, PackedInputs) else prepare_inputs(candidates)
        scores = np.empty((self._native_model_count, len(packed.lengths)), dtype=np.float64)
        if scores.size:
            rc = self._lib.replacer_cpu_infer_eval(
                self._ptr, packed.tokens, packed.lengths,
                len(packed.lengths), packed.tokens.shape[1], scores,
            )
            if rc:
                raise RuntimeError(f'native inference evaluation failed: {rc}')
        return scores if self._model_row_map is None else scores[self._model_row_map]

    def features(self, inputs):
        """Evaluate the complete model x trajectory block with one native dispatch.

        Output arrays are model-major. Python owns them and the C++ workers
        finish before returning; no platform-specific or thread-local objects leak.
        """
        if not self._ptr:
            raise RuntimeError('native population context is closed')
        packed = inputs if isinstance(inputs, PackedInputs) else prepare_inputs(inputs)
        n = len(packed.lengths)
        width = self._native_feature_width
        features = np.zeros((self._native_model_count, n, width), dtype=np.float32)
        baselines = np.zeros((self._native_model_count, n), dtype=np.float64)
        stats = np.zeros((self._native_model_count, n, 4), dtype=np.int32)
        if n:
            rc = self._lib.replacer_cpu_outer_eval(
                self._ptr, packed.tokens, packed.lengths, n, packed.tokens.shape[1],
                width, features, baselines, stats)
            if rc:
                raise RuntimeError(f'native population evaluation failed: {rc}')
        if self._model_row_map is not None:
            features = features[self._model_row_map]
            baselines = baselines[self._model_row_map]
            stats = stats[self._model_row_map]
        return features, baselines, stats

    def close(self):
        if self._ptr:
            self._lib.replacer_cpu_infer_destroy(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):
        if getattr(self, '_ptr', None):
            self.close()
