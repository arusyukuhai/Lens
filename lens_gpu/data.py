"""Document split + exact-count nested corruption, batched over documents/levels."""
import hashlib
import random
from pathlib import Path
import torch


def load_corpus(path,chunk_len=2000,max_chunks=65536):
    chunks=[]; current=bytearray()
    with open(path,'rb') as stream:
        for line in stream:
            line=line.rstrip(b'\r\n')
            if line==b'===SPLIT===':
                if len(current)>=3: chunks.append(bytes(current))
                current=bytearray()
                if max_chunks and len(chunks)>=max_chunks: break
            elif len(current)<chunk_len:
                current.extend((line+b'\n')[:chunk_len-len(current)])
        else:
            if len(current)>=3: chunks.append(bytes(current))
    if len(chunks)<6: raise ValueError('need >=6 documents separated by ===SPLIT===; no within-document validation split')
    return chunks


def split_corpus(chunks,seed=1729,valid=8,test=8):
    # Exact duplicate documents cannot leak across partitions.
    unique=list(dict.fromkeys(chunks)); order=list(range(len(unique)))
    random.Random(seed).shuffle(order)
    if valid<1 or test<1 or len(unique)<=valid+test:
        raise ValueError('need distinct train documents in addition to --valid and --test')
    train=[unique[i] for i in order[:-(valid+test)]]
    val=[unique[i] for i in order[-(valid+test):-test]]
    held=[unique[i] for i in order[-test:]]
    digest=lambda rows: [hashlib.sha256(c).hexdigest() for c in rows]
    manifest={'seed':seed,'train_sha256':digest(train),'valid_sha256':digest(val),'test_sha256':digest(held)}
    return train,val,held,manifest


def alphabet_of(chunks):
    values=sorted(set(b for c in chunks for b in c))
    if len(values)<2: raise ValueError('corruption needs >=2 bytes in training alphabet')
    return torch.tensor(values,dtype=torch.long)


def diffusion_batch(chunks,alphabet,levels,generator,pin=False,capacity_factor=1.):
    """CPU vectorized generation; same original has one shared nested path.
    Distinct random interior counts, mandatory clean/fully corrupt endpoints.
    Out-of-training-alphabet bytes get any train-alphabet replacement.
    Every changed byte is different; target is actual unchanged fraction.
    """
    n=len(chunks); length=max(len(c) for c in chunks)
    lengths=torch.tensor([len(c) for c in chunks],dtype=torch.long)
    levels=min(levels,int(lengths.min())+1)
    if levels<2: raise ValueError('levels >=2 required')
    clean=torch.zeros(n,length,dtype=torch.long)
    for i,c in enumerate(chunks): clean[i,:len(c)]=torch.tensor(list(c))
    pos=torch.arange(length)[None,:]
    valid=pos<lengths[:,None]
    keys=torch.rand(n,length,generator=generator)+2*(~valid)
    rank=keys.argsort(-1).argsort(-1)
    lookup=torch.full((256,),-1,dtype=torch.long)
    lookup[alphabet]=torch.arange(len(alphabet))
    current=lookup[clean]
    choice=torch.randint(len(alphabet)-1,(n,length),generator=generator)
    choice=choice+(choice>=current).long()*(current>=0).long()
    # For unseen byte, use full alphabet, still always differs from original.
    all_choice=torch.randint(len(alphabet),(n,length),generator=generator)
    choice=torch.where(current>=0,choice,all_choice)
    replacement=alphabet[choice]
    candidates=torch.arange(1,length)[None,:].expand(n,-1)
    keys=torch.rand(n,length-1,generator=generator)+2*(candidates>=lengths[:,None])
    interior=candidates.gather(1,keys.argsort(-1)[:,:levels-2])
    counts=torch.cat((torch.zeros(n,1,dtype=torch.long),interior,lengths[:,None]),1).sort(-1).values
    changed=(rank[:,None,:]<counts[:,:,None])&valid[:,None,:]
    noisy=torch.where(changed,replacement[:,None,:],clean[:,None,:]).reshape(n*levels,length)
    if capacity_factor<1.: raise ValueError('capacity_factor must be >=1')
    import math
    capacity=max(length,math.ceil(length*capacity_factor))
    if capacity>length:
        noisy=torch.nn.functional.pad(noisy,(0,capacity-length))
    lens=lengths[:,None].expand(n,levels).reshape(-1)
    target=(1-counts.float()/lengths[:,None]).reshape(-1)
    if pin: noisy,lens,target=noisy.pin_memory(),lens.pin_memory(),target.pin_memory()
    return noisy,lens,target


def synthetic(path,n=64,length=256):
    """Small artificial code-like corpus; not a real-world benchmark."""
    rng=random.Random(81); docs=[]
    for i in range(n):
        lines=[f'# document {i}\n']
        while sum(map(len,lines))<length:
            name=rng.choice(['value','index','total','result','offset','buffer'])
            lines.append(f'if {name} > {rng.randrange(100)}: return {name} + {rng.randrange(10)}\n')
        docs.append(''.join(lines)[:length])
    Path(path).write_text('\n===SPLIT===\n'.join(docs))
