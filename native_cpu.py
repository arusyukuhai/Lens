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
                                 ctypes.c_int,ctypes.c_int,arr8,arr32,
                                 ctypes.c_int,arr32,ctypes.c_int]
    lib.lens_eval_cpu.restype=ctypes.c_int
    lib.lens_eval_cpu_progress.argtypes=lib.lens_eval_cpu.argtypes + [_PROGRESS_CALLBACK]
    lib.lens_eval_cpu_progress.restype=ctypes.c_int
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
    G=len(genomes); T=len(texts); cap=rules*65
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
            if pn>64 or qn>64: raise ValueError('Rule length exceeds 64')
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
    args=(pat,rep,po,ro,luts,G,rules,packed,offs,T,out,workers)
    if on_genome_done is None:
        err=lib.lens_eval_cpu(*args)
    else:
        # ctypes releases the GIL for the native call, and temporarily obtains
        # it for each callback. tqdm throttles the expensive terminal refresh.
        callback=_PROGRESS_CALLBACK(on_genome_done)
        err=lib.lens_eval_cpu_progress(*args, callback)
    if err:
        raise RuntimeError(f'Native CPU evaluator returned {err}')
    return out
