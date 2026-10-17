"""Lens serial trainer with the OLD exact Replace semantics.

Two deliberately distinct training engines:
  --mode scalable  : batched GPU/MPS C2 *approximation* of old Replace
  --mode exact-small: exact joint-law differentiable serial Replace, finite supports

Both apply R_0, R_1, ..., R_(N-1) in order and train a SwiGLU
head on firing counts divided by the original input length.

The scalable mode's hard inference uses the exact old deterministic Replace.
It is NOT mathematically the full joint pushforward during soft training.
"""
import argparse
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--mode',choices=('scalable','exact-small'),default='scalable')
    p.add_argument('-h','--help',action='store_true')
    args,rest=p.parse_known_args()
    if args.help:
        if args.mode=='exact-small':
            from train_exact_small import main as exact_main
            sys.argv=[sys.argv[0],'--help']
            return exact_main()
        from lens_gpu.train import main as gpu_main
        sys.argv=[sys.argv[0],'--help']
        return gpu_main()
    sys.argv=[sys.argv[0],*rest]
    if args.mode=='exact-small':
        from train_exact_small import main
    else:
        from lens_gpu.train import main
    return main()


if __name__=='__main__':
    main()
