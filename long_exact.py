"""Exact specialization for fixed byte inputs and fixed supervised targets.

Enumerates rule support ONLY, shares matches across replacements, compares
outputs via spans, and caches accepting rule pairs. Training is exact log-space
sequence likelihood on the declared finite rule family; it is NOT a general
fast differentiable marginal-output layer for uncertain long inputs.
"""
from dataclasses import dataclass
from itertools import product
import math
import torch
from torch import nn


@dataclass(frozen=True)
class MatchPlan:
    text: bytes
    matches: tuple  # (start, end, ((capture_start,capture_end),...))
    capture_totals: tuple
    consumed: int
    max_output: int

    def size(self, replacement):
        n = len(self.capture_totals)
        size = len(self.text) - self.consumed
        for c in replacement:
            if c >= 0:
                size += len(self.matches)
            elif n:
                r = -c-1
                size += self.capture_totals[r if r < n else 0]
        return size

    def pieces(self, replacement):
        pos = 0
        for start, end, caps in self.matches:
            yield self.text[pos:start]
            for c in replacement:
                if c >= 0:
                    yield bytes((c,))
                elif caps:
                    r = -c-1
                    a,b = caps[r if r < len(caps) else 0]
                    yield self.text[a:b]
            pos = end
        yield self.text[pos:]

    def render(self, replacement):
        replacement = validate_replacement(replacement)
        if not self.matches or self.size(replacement) > self.max_output:
            return self.text
        return b''.join(self.pieces(replacement))

    def equals(self, target, replacement):
        # Callers compiling a bank have already validated replacement opcodes.
        if not self.matches or self.size(replacement) > self.max_output:
            return self.text == target
        if self.size(replacement) != len(target):
            return False
        cursor = 0
        for piece in self.pieces(replacement):
            if not target.startswith(piece, cursor):
                return False
            cursor += len(piece)
        return cursor == len(target)


def validate_replacement(replacement):
    r = tuple(replacement)
    if len(r) > 64 or any(not -15 <= c < 256 for c in r):
        raise ValueError('unsupported replacement')
    return r


class BytePattern:
    """Native bytes.find, deterministic lazy delimiters, monotone captures."""
    def __init__(self, pattern, *, min_wildcard_len=0, max_output=1_048_576):
        self.pattern = tuple(pattern)
        if min_wildcard_len < 0 or max_output < 0 or any(c >= 256 for c in pattern):
            raise ValueError('invalid pattern or limit')
        self.minimum, self.max_output = min_wildcard_len, max_output
        parts = [bytearray()]
        for c in self.pattern:
            if c < 0:
                parts.append(bytearray())
            else:
                parts[-1].append(c)
        self.parts = tuple(bytes(p) for p in parts)
        self.n = len(parts)-1
        self.identity = not self.pattern or self.n > 16 or not any(self.parts)

    def match_plan(self, text):
        x = bytes(text)
        if self.identity:
            return MatchPlan(x,(),(),0,self.max_output)
        pos, consumed = 0, 0
        matches, totals = [], [0]*self.n
        while pos < len(x):
            start = x.find(self.parts[0],pos) if self.parts[0] else pos
            if start < 0:
                break
            end, caps, ok = start+len(self.parts[0]), [], True
            for w,literal in enumerate(self.parts[1:]):
                if literal:
                    found = x.find(literal,end+self.minimum)
                    if found < 0:
                        ok = False
                        break
                    caps.append((end,found))
                    end=found+len(literal)
                else:
                    stop=len(x) if w==self.n-1 else end
                    caps.append((end,stop))
                    end=stop
            if not ok:
                # Each greedy delimiter position is monotone in the start.
                # If this chain cannot finish, a later start cannot finish it.
                break
            matches.append((start,end,tuple(caps)))
            consumed += end-start
            for i,(a,b) in enumerate(caps):
                totals[i] += b-a
            pos=end
        return MatchPlan(x,tuple(matches),tuple(totals),consumed,self.max_output)

    def apply(self, text, replacement):
        return self.match_plan(text).render(replacement)


class RuleBank(nn.Module):
    def __init__(self, supports):
        super().__init__()
        self.supports=tuple(tuple(row) for row in supports)
        for row in self.supports:
            if not row or len(set(row)) != len(row) or any(not 0 <= c < 271 for c in row):
                raise ValueError('invalid fixed support')
        self.states=[]
        self.groups=[]
        for k in range(len(supports)+1):
            choices=list(product(*(range(len(r)) for r in supports[:k])))
            name=f'local_indices_{k}'
            self.register_buffer(name,torch.tensor(choices,dtype=torch.long).reshape(len(choices),k))
            self.groups.append(name)
            for choice in choices:
                self.states.append(tuple(supports[i][v] if supports[i][v]<256 else
                                         -(supports[i][v]-256)-1 for i,v in enumerate(choice)))
        self.states=tuple(self.states)

    def log_weights(self, sequence):
        if sequence.supports != self.supports:
            raise ValueError('model support must equal the compiled bank support')
        if self.local_indices_0.device != sequence.length_logits.device:
            raise ValueError('move bank and sequence to the same device')
        lengths=sequence.length_logits.log_softmax(-1)
        rows=[z.log_softmax(-1) for z in sequence.symbol_logits]
        result=[]
        for k,name in enumerate(self.groups):
            idx=getattr(self,name)
            w=lengths[k].expand(idx.shape[0])
            for i in range(k):
                w=w+rows[i][idx[:,i]]
            result.append(w)
        return torch.cat(result)


class LongTextLikelihood(nn.Module):
    def __init__(self, examples, a_supports, b_supports, *, min_wildcard_len=0,
                 max_output=1_048_576, max_rule_pairs=1_000_000):
        super().__init__()
        if not examples or len(b_supports)>64:
            raise ValueError('nonempty examples and <=64 replacement slots required')
        def count(rows):
            total, prefix=1,1
            for row in rows:
                prefix*=len(row); total+=prefix
            return total
        pairs=count(a_supports)*count(b_supports)
        if pairs>max_rule_pairs:
            raise ValueError(f'{pairs} rule pairs exceed the explicit limit')
        self.a,self.b=RuleBank(a_supports),RuleBank(b_supports)
        for b in self.b.states:
            validate_replacement(b)
        matchers=[BytePattern(a,min_wildcard_len=min_wildcard_len,max_output=max_output)
                  for a in self.a.states]
        unique,group_ids,accepted_counts={},[],[]
        for x,y in examples:
            x,y=bytes(x),bytes(y)
            accepted=[]
            for ai,matcher in enumerate(matchers):
                plan=matcher.match_plan(x)
                for bi,b in enumerate(self.b.states):
                    if plan.equals(y,b):
                        accepted.append((ai,bi))
            if not accepted:
                raise ValueError('target not reachable in fixed candidate family')
            key=tuple(accepted)
            if key not in unique:
                unique[key]=len(unique)
                idx=unique[key]
                self.register_buffer(f'accepted_{idx}',torch.tensor(key,dtype=torch.long))
            group_ids.append(unique[key])
            accepted_counts.append(len(key))
        self.group_count=len(unique)
        self.register_buffer('group_ids',torch.tensor(group_ids,dtype=torch.long))
        self.stats=dict(examples=len(examples),pattern_states=len(self.a.states),
                        replacement_states=len(self.b.states),rule_pairs=pairs,
                        unique_acceptance_sets=len(unique),accepted_pairs_per_example=accepted_counts,
                        input_bytes=sum(len(x) for x,_ in examples))

    def forward(self, A, B):
        la,lb=self.a.log_weights(A),self.b.log_weights(B)
        values=[]
        for i in range(self.group_count):
            idx=getattr(self,f'accepted_{i}')
            values.append(-torch.logsumexp(la[idx[:,0]]+lb[idx[:,1]],dim=0))
        return torch.stack(values)[self.group_ids].mean()

    def functional_nll(self, a_logits, a_length, b_logits, b_length):
        """Functional interface for derivative checks, with no parameter mutation."""
        class Values:
            pass
        A,B=Values(),Values()
        A.supports=self.a.supports; A.symbol_logits=a_logits; A.length_logits=a_length
        B.supports=self.b.supports; B.symbol_logits=b_logits; B.length_logits=b_length
        return self(A,B)
