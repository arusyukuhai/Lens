// Portable C++17 reference-equivalent CPU accelerator for Lens's recurrent GA.
// Built locally at first use; no network, binary downloads or fixed context size.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <limits>
#include <vector>

namespace {
using token = unsigned char;
struct Rule {
    const int16_t *p = nullptr, *q = nullptr;
    int pn = 0, qn = 0, wc = 0;
    int anchor_kind = 0, anchor_value = -1;
    bool always_identity = false;
};
struct Capture { int start = 0, length = 0; };

// Per-worker epoch stamps: avoid clearing 256 KiB for every example.
// One exact pair/byte membership table is reused across all subsequent rollouts.
struct ThreadMarks {
    std::array<uint32_t,256> bytes{};
    std::array<uint32_t,65536> pairs{};
    uint32_t epoch = 0;
    uint32_t next_epoch() {
        if (++epoch == 0) {
            bytes.fill(0); pairs.fill(0);
            epoch = 1;
        }
        return epoch;
    }
};

// Safe static rule simplification. An identity rule has no externally visible
// effect even when it matches: no firing counts or other side effects exist.
inline bool always_identity(const Rule &r) {
    if (r.pn != r.qn || r.pn <= 0 || r.wc > 15) return false;
    int capture_index = 0;
    for (int i=0; i<r.pn; ++i) {
        if (r.p[i] >= 0) {
            if (r.q[i] != r.p[i]) return false;
        } else {
            ++capture_index;
            if (r.q[i] != -capture_index) return false;
        }
    }
    return true;
}

inline bool match_at(const token *s, int n, int start, const Rule &r,
                     Capture (&caps)[16], int &end) {
    int i = 0, pos = start, c = 0;
    while (i < r.pn && r.p[i] >= 0) {
        if (pos >= n || s[pos] != r.p[i]) return false;
        ++i; ++pos;
    }
    while (i < r.pn) {
        if (r.p[i] != -1 || c >= 16) return false;
        ++i;
        int begin = pos, lit = i;
        while (i < r.pn && r.p[i] >= 0) ++i;
        int llen = i - lit;
        caps[c].start = begin;
        if (llen == 0) {
            if (i == r.pn) { caps[c].length = n - pos; pos = n; }
            else caps[c].length = 0;
        } else {
            int found = -1;
            for (int k = pos; k + llen <= n; ++k) {
                if (s[k] != r.p[lit]) continue;
                int z = 1;
                while (z < llen && s[k+z] == r.p[lit+z]) ++z;
                if (z == llen) { found = k; break; }
            }
            if (found < 0) return false;
            caps[c].length = found - pos;
            pos = found + llen;
        }
        ++c;
    }
    end = pos;
    return pos > start;
}

struct Model {
    std::vector<Rule> rules;
    std::array<token, 256> lut;
    std::array<token, 256> inverse;
    std::array<std::array<token, 256>, 4> transforms;
};

Model compile_model(const int16_t *pat, const int16_t *rep,
                    const int32_t *po, const int32_t *ro,
                    const token *raw_lut, int R,
                    const std::array<uint32_t, 256>& byte_frequencies,
                    const std::array<uint32_t, 65536>& pair_frequencies) {
    Model model;
    model.rules.reserve(R);
    for (int i = 0; i < 256; ++i) {
        model.lut[i] = raw_lut[i];
        model.inverse[raw_lut[i]] = static_cast<token>(i);
    }
    for (int i = 0; i < 256; ++i) {
        int z = model.lut[i];
        model.transforms[0][i] = model.inverse[(z + 1) & 255];
        model.transforms[1][i] = model.inverse[(z - 1) & 255];
        model.transforms[2][i] = model.inverse[(z * 2) & 255];
        model.transforms[3][i] = model.inverse[z / 2];
    }
    for (int i = 0; i < R; ++i) {
        Rule r;
        r.p = pat + po[i]; r.pn = po[i+1] - po[i];
        r.q = rep + ro[i]; r.qn = ro[i+1] - ro[i];
        for (int k = 0; k < r.pn; ++k) if (r.p[k] == -1) ++r.wc;
        r.always_identity = always_identity(r);
        // Prefer the rarest necessary adjacent literal pair, not the first.
        // A zero-occurrence pair proves a rule cannot fire for these examples
        // until other rules create it; the dynamic table is still authoritative.
        uint32_t min_count = std::numeric_limits<uint32_t>::max();
        for (int k=0;k+1<r.pn;++k) {
            if (r.p[k] >= 0 && r.p[k+1] >= 0) {
                int key=(r.p[k]<<8)|r.p[k+1];
                if (pair_frequencies[key] < min_count) {
                    min_count=pair_frequencies[key];
                    r.anchor_kind=2; r.anchor_value=key;
                }
            }
        }
        if (r.anchor_kind == 0) {
            for (int k=0;k<r.pn;++k) if (r.p[k]>=0 && byte_frequencies[r.p[k]] < min_count) {
                min_count=byte_frequencies[r.p[k]];
                r.anchor_kind=1; r.anchor_value=r.p[k];
            }
        }
        model.rules.push_back(r);
    }
    return model;
}

int run(const Model &m, const token *txt, int L) {
    if (L < 2) return 0;
    // State never expands during a sweep. Teacher forcing adds one byte at each
    // step; thus the actual input length + 1 is sufficient, with no truncation.
    std::vector<token> buffer_a(static_cast<size_t>(L+1));
    std::vector<token> buffer_b(static_cast<size_t>(L+1));
    auto *a = &buffer_a, *b = &buffer_b;
    (*a)[0] = txt[0]; (*a)[1] = 0;
    int n = 2, correct = 0;
    static thread_local ThreadMarks marks;
    uint32_t epoch = marks.epoch;
    bool dirty = true;
    for (int t = 1; t < L; ++t) {
        for (const Rule &r : m.rules) {
            if (n == 0 || r.pn == 0 || r.always_identity) continue;
            if (dirty) {
                epoch=marks.next_epoch();
                const token *cur = a->data();
                for (int j = 0; j < n; ++j) marks.bytes[cur[j]] = epoch;
                for (int j = 0; j + 1 < n; ++j)
                    marks.pairs[(int(cur[j])<<8) | cur[j+1]] = epoch;
                dirty = false;
            }
            if (r.anchor_kind == 1 && marks.bytes[r.anchor_value] != epoch) continue;
            if (r.anchor_kind == 2 && marks.pairs[r.anchor_value] != epoch) continue;
            const token *src = a->data();
            token *dst = b->data();
            int scan = 0, prev = 0, out = 0;
            bool matched = false;
            Capture captures[16];
            while (scan < n) {
                int ss = -1, finish = -1;
                if (r.p[0] >= 0) {
                    for (int k = scan; k < n; ++k) {
                        if (src[k] != r.p[0]) continue;
                        int f = 0;
                        if (match_at(src,n,k,r,captures,f)) {ss=k; finish=f; break;}
                    }
                } else {
                    int f = 0;
                    if (match_at(src,n,scan,r,captures,f)) {ss=scan; finish=f;}
                }
                if (ss < 0) break;
                matched = true;
                if (ss > prev) { std::memcpy(dst+out,src+prev,size_t(ss-prev)); out += ss-prev; }
                int emitted = 0, allowance = finish-ss;
                for (int j = 0; j < r.qn; ++j) {
                    int op = r.q[j];
                    if (op >= 0) {
                        if (emitted < allowance) {dst[out++] = static_cast<token>(op); ++emitted;}
                        continue;
                    }
                    if (r.wc <= 0) continue;
                    int kind = 0, ci = 0;
                    if (op >= -15) {kind=0; ci=-op-1;}
                    else if (op >= -31) {kind=2; ci=-op-16;}
                    else if (op >= -47) {kind=1; ci=-op-32;}
                    else if (op >= -111) {int off=-op-48; kind=3+off/16; ci=off%16;}
                    else continue;
                    if (ci >= r.wc) ci=0;
                    const Capture &cap = captures[ci];
                    int begin=cap.start, length=cap.length;
                    if (emitted + length > allowance) continue;
                    if (kind == 0) {
                        if (length) std::memcpy(dst+out,src+begin,size_t(length));
                    } else if (kind == 1) {
                        for (int k=0; k<length; ++k) dst[out+k]=src[begin+length-k-1];
                    } else if (kind == 2) {
                        if (length) std::memcpy(dst+out,src+begin,size_t(length));
                        std::sort(dst+out,dst+out+length,[&](token u,token v){return m.lut[u]<m.lut[v];});
                    } else {
                        const auto &map = m.transforms[kind-3];
                        for (int k=0; k<length; ++k) dst[out+k]=map[src[begin+k]];
                    }
                    out += length; emitted += length;
                }
                prev = scan = finish;
            }
            if (!matched) continue;
            if (n > prev) {std::memcpy(dst+out,src+prev,size_t(n-prev)); out+=n-prev;}
            bool changed = out != n || std::memcmp(src,dst,size_t(out)) != 0;
            n = out;
            std::swap(a,b);
            dirty = changed;
        }
        if (n==0) {(*a)[0]=0; n=1;}
        if ((*a)[n-1] == txt[t]) ++correct;
        (*a)[n-1] = txt[t];
        (*a)[n++] = 0;
        dirty = true;
    }
    return correct;
}
} // namespace

// Input arrays: G genomes, R rules, fixed token capacity R*65 per genome,
// concatenated sample records with offsets. out is [G, T] int32.
static int lens_eval_cpu_impl(
    const int16_t *patterns, const int16_t *replacements,
    const int32_t *po, const int32_t *ro, const uint8_t *lut,
    int G, int R, const uint8_t *texts, const int32_t *offsets,
    int T, int32_t *out, int workers, void (*on_genome_done)(int)) {
    if (!patterns || !replacements || !po || !ro || !lut || !texts || !offsets || !out || G<0 || R<0 || T<0) return -1;
    if (G==0 || T==0) return 0;
    // Corpus-wide raw frequencies only choose sound anchor positions.
    // A candidate with a chosen anchor still checks current-state membership.
    std::array<uint32_t,256> freq_bytes{};
    std::array<uint32_t,65536> freq_pairs{};
    for (int s=0;s<T;++s) {
        for (int k=offsets[s];k<offsets[s+1];++k) {
            const int b=texts[k];
            ++freq_bytes[b];
            if (k+1<offsets[s+1]) ++freq_pairs[(b<<8)|texts[k+1]];
        }
    }
    const int stride = R*65;
    const int offset_stride = R+1;
    std::atomic<int> cursor{0};
    auto task = [&](){
        for (;;) {
            int g = cursor.fetch_add(1,std::memory_order_relaxed);
            if (g >= G) break;
            Model m = compile_model(patterns+g*stride,replacements+g*stride,
                                    po+g*offset_stride,ro+g*offset_stride,
                                    lut+g*256,R,freq_bytes,freq_pairs);
            for (int s=0; s<T; ++s) {
                int st=offsets[s], en=offsets[s+1];
                out[g*T+s] = run(m,texts+st,en-st);
            }
            // The callback runs only after the complete genome was evaluated.
            // It cannot alter fitness or scheduling; nullptr is the fast path.
            if (on_genome_done) on_genome_done(1);
        }
    };
    int nthreads = workers > 0 ? workers : int(std::thread::hardware_concurrency());
    nthreads = std::max(1,std::min(nthreads,G));
    std::vector<std::thread> jobs;
    jobs.reserve(std::max(0,nthreads-1));
    for(int j=1;j<nthreads;++j) jobs.emplace_back(task);
    task();
    for(auto &job:jobs) job.join();
    return 0;
}

// Preserve the legacy C ABI for existing callers.
extern "C" int lens_eval_cpu(
    const int16_t *patterns, const int16_t *replacements,
    const int32_t *po, const int32_t *ro, const uint8_t *lut,
    int G, int R, const uint8_t *texts, const int32_t *offsets,
    int T, int32_t *out, int workers) {
    return lens_eval_cpu_impl(patterns, replacements, po, ro, lut, G, R,
                              texts, offsets, T, out, workers, nullptr);
}

// Progress ticks are per finished genome, including across native worker threads.
extern "C" int lens_eval_cpu_progress(
    const int16_t *patterns, const int16_t *replacements,
    const int32_t *po, const int32_t *ro, const uint8_t *lut,
    int G, int R, const uint8_t *texts, const int32_t *offsets,
    int T, int32_t *out, int workers, void (*on_genome_done)(int)) {
    return lens_eval_cpu_impl(patterns, replacements, po, ro, lut, G, R,
                              texts, offsets, T, out, workers, on_genome_done);
}
