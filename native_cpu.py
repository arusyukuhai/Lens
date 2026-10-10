"""Optional locally compiled C++17 evaluator; falls back to Python if unavailable.

Build uses a local compiler and /tmp cache, not the network. No persistent
model-specific cache or artificial recurrent-history limit is introduced.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

_LIBRARY = None
_ERROR = None
_PROGRESS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int)


def _get_library():
    global _LIBRARY, _ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    if _ERROR is not None:
        raise RuntimeError(_ERROR)
    source = Path(__file__).with_name('replacer_native.cpp')
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        _ERROR = 'C++ compiler not installed'
        raise RuntimeError(_ERROR)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    cache = Path(tempfile.gettempdir()) / 'lens_ar_native'
    cache.mkdir(exist_ok=True)
    target = cache / ('replacer_' + digest + suffix)
    if not target.exists():
        tmp = target.with_suffix(target.suffix + f'.{os.getpid()}.tmp')
        cmd = [compiler, '-std=c++17', '-O3', '-DNDEBUG', '-fPIC',
               '-dynamiclib' if sys.platform == 'darwin' else '-shared',
               str(source), '-o', str(tmp), '-pthread']
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            os.replace(tmp,target)
        except (OSError, subprocess.CalledProcessError) as e:
            if tmp.exists(): tmp.unlink()
            detail = getattr(e,'stderr',None) or str(e)
            _ERROR = f'Native C++ build failed: {detail}'
            raise RuntimeError(_ERROR) from e
    lib=ctypes.CDLL(str(target))
    arr16 = np.ctypeslib.ndpointer(dtype=np.int16,flags='C_CONTIGUOUS')
    arr32 = np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    arr8 = np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.lens_eval_cpu.argtypes=[arr16,arr16,arr32,arr32,arr8,
                                 ctypes.c_int,ctypes.c_int,ctypes.c_int,arr8,arr32,
                                 ctypes.c_int,arr32,ctypes.c_int]
    lib.lens_eval_cpu.restype=ctypes.c_int
    lib.lens_eval_cpu_progress.argtypes=lib.lens_eval_cpu.argtypes + [_PROGRESS_CALLBACK]
    lib.lens_eval_cpu_progress.restype=ctypes.c_int
    lib.lens_trace_cpu.argtypes=[arr16,arr16,arr32,arr32,arr8,
        ctypes.c_int,arr8,arr32,ctypes.c_int,ctypes.c_int,ctypes.c_int,
        ctypes.c_int,arr8,arr8,arr32,arr32,arr32]
    lib.lens_trace_cpu.restype=ctypes.c_int
    _LIBRARY=lib
    return lib


def evaluate_cpu(genomes, texts, rules: int, workers: int = 0,
                 on_genome_done=None) -> np.ndarray:
    """Evaluate genomes concurrently; optionally report each fully scored genome.

    The native worker threads call on_genome_done(1) after all texts of a
    genome finish.  Progress is observational and does not alter GA state.
    """
    if not genomes or not texts:
        return np.zeros((len(genomes),len(texts)),dtype=np.int32)
    lib=_get_library()
    G=len(genomes); T=len(texts)
    cap=max(1,max(max(sum(len(r.pattern) for r in g.rules),
                        sum(len(r.replacement) for r in g.rules)) for g in genomes))
    pat=np.zeros((G,cap),dtype=np.int16)
    rep=np.zeros_like(pat)
    po=np.zeros((G,rules+1),dtype=np.int32)
    ro=np.zeros_like(po)
    luts=np.empty((G,256),dtype=np.uint8)
    for gi,g in enumerate(genomes):
        if len(g.rules)!=rules:
            raise ValueError('All genomes must have the same rule count')
        luts[gi]=np.asarray(g.embedding,dtype=np.uint8)
        pc=rc=0
        for i,r in enumerate(g.rules):
            pn=len(r.pattern); qn=len(r.replacement)
            po[gi,i]=pc; ro[gi,i]=rc
            pat[gi,pc:pc+pn]=r.pattern
            rep[gi,rc:rc+qn]=r.replacement
            pc+=pn;rc+=qn
        po[gi,rules]=pc; ro[gi,rules]=rc
    buf=bytearray(); offsets=[0]
    for line in texts:
        buf.extend(bytes(line)); offsets.append(len(buf))
    packed=np.frombuffer(buf,dtype=np.uint8).copy()
    offs=np.asarray(offsets,dtype=np.int32)
    out=np.zeros((G,T),dtype=np.int32)
    args=(pat,rep,po,ro,luts,G,rules,cap,packed,offs,T,out,workers)
    if on_genome_done is None:
        err=lib.lens_eval_cpu(*args)
    else:
        # ctypes releases the GIL for the native call, and temporarily obtains
        # it for each callback. tqdm throttles the expensive terminal refresh.
        callback=_PROGRESS_CALLBACK(on_genome_done)
        err=lib.lens_eval_cpu_progress(*args, callback)
    if err in (-3, -4):
        raise MemoryError('Replacer state exceeded available/representable memory; '
                          'no state was silently truncated')
    if err:
        raise RuntimeError(f'Native CPU evaluator returned {err}')
    return out


def trace_cpu(genome, texts, rule_index: int, samples: int = 8,
              max_bytes: int = 512):
    """Return (real pre-rule windows, final-sweep windows, per-text hit counts).

    Uses one native evaluation of the unchanged genome; all recurrent states
    remain unbounded. Only observational snapshots are length-limited.
    """
    if not 0 <= rule_index < len(genome.rules):
        raise ValueError('invalid rule index')
    if samples < 1 or max_bytes < 1:
        raise ValueError('samples and max_bytes must be positive')
    lib=_get_library()
    R=len(genome.rules); T=len(texts)
    pp=[v for r in genome.rules for v in r.pattern]
    qq=[v for r in genome.rules for v in r.replacement]
    p=np.asarray(pp, dtype=np.int16)
    q=np.asarray(qq, dtype=np.int16)
    po=np.zeros(R+1,dtype=np.int32)
    ro=np.zeros(R+1,dtype=np.int32)
    for i,r in enumerate(genome.rules):
        po[i+1]=po[i]+len(r.pattern)
        ro[i+1]=ro[i]+len(r.replacement)
    lut=np.asarray(genome.embedding,dtype=np.uint8)
    packed=np.frombuffer(b''.join(bytes(t) for t in texts),dtype=np.uint8).copy()
    offsets=np.zeros(T+1,dtype=np.int32)
    for i,t in enumerate(texts):
        offsets[i+1]=offsets[i]+len(t)
    before=np.zeros((T,samples,max_bytes),dtype=np.uint8)
    after=np.zeros_like(before)
    bl=np.zeros((T,samples),dtype=np.int32)
    al=np.zeros_like(bl)
    scores=np.zeros(T,dtype=np.int32)
    err=lib.lens_trace_cpu(p,q,po,ro,lut,R,packed,offsets,T,
                           rule_index,samples,max_bytes,before,after,bl,al,scores)
    if err:
        raise RuntimeError(f'native trace failed: {err}')
    pre=[bytes(before[i,j,:bl[i,j]]) for i in range(T) for j in range(samples)
         if bl[i,j]>0]
    post=[bytes(after[i,j,:al[i,j]]) for i in range(T) for j in range(samples)
          if al[i,j]>0]
    return pre,post,scores
