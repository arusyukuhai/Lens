// Replacer exact CPU trajectory evaluator. C++17 / OpenMP optional.
// The Python reference in main.py is the behavioral specification.
#include <algorithm>
#include <atomic>
#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
using I = int32_t;
using V = std::vector<I>;
struct R { V a,b; I anchor=-1,anchor2=-1; };
struct Memory {
    std::deque<V> left, right;
    int cap, left_tokens=0, right_tokens=0;
    explicit Memory(int n):cap(std::max(1,n)){}
    void insert(std::deque<V>& bank, int& total, V item, bool front) {
        if ((int)item.size()>cap) item.resize(cap);
        while(!bank.empty() && (bank.size()>=2048 || total+(int)item.size()>cap)) {
            total-=int(front ? bank.back().size() : bank.front().size());
            if(front) bank.pop_back(); else bank.pop_front();
        }
        total+=int(item.size());
        if(front) bank.push_front(std::move(item)); else bank.push_back(std::move(item));
    }
    void pair(V a,V b,bool front){
        insert(left,left_tokens,std::move(a),front);
        insert(right,right_tokens,std::move(b),front);
    }
    V exec(I op,bool& changed){
        int idx=-op-128;
        if(idx<0 || idx>=24)return {};
        int action=idx%3; int q=idx/3;
        bool right_sel=q&1, second=(q>>1)&1, bottom=(q>>2)&1;
        auto& bank=right_sel?right:left;
        if(bank.empty())return {};
        int offset=(second && bank.size()>=2)?1:0;
        int at=bottom?int(bank.size())-1-offset:offset;
        V value=bank[at];
        if(action){
            (right_sel ? right_tokens : left_tokens)-=int(bank[at].size());
            bank.erase(bank.begin()+at);changed=true;
        }
        return action==2?V():value;
    }
    std::string key(const V& state)const{
        std::string key;
        auto add=[&](I x){key.append(reinterpret_cast<const char*>(&x),sizeof(I));};
        add((I)state.size());
        if(!state.empty())key.append((const char*)state.data(), state.size()*sizeof(I));
        auto append_bank=[&](const std::deque<V>& bank){
            add((I)bank.size());
            for(auto& v:bank){add((I)v.size()); if(!v.empty())key.append((const char*)v.data(),v.size()*sizeof(I));}
        };
        append_bank(left);append_bank(right);
        return key;
    }
};
struct Capture { int pos,len; };
// Captures are bounded to 16. Inline storage avoids a heap allocation on
// every attempted rule match, including unsuccessful literal probes.
struct Captures {
    std::array<Capture,16> buf{};
    uint8_t count=0;
    void clear(){count=0;}
    size_t size()const{return count;}
    void push_back(Capture c){buf[count++]=c;}
    const Capture& operator[](size_t i)const{return buf[i];}
};
static bool match_at(const V& st,const V& pat,int start,int& finish,Captures& caps){
    int n=st.size(),p=0,pos=start;
    caps.clear();
    while(p<(int)pat.size() && pat[p]>=0){
        if(pos>=n || st[pos]!=pat[p])return false;
        ++p;++pos;
    }
    while(p<(int)pat.size()){
        if(pat[p]>=0 || caps.size()>=16)return false;
        ++p;
        int cs=pos, lb=p;
        while(p<(int)pat.size() && pat[p]>=0)++p;
        int length=p-lb;
        if(length==0){
            if(p>=(int)pat.size()){caps.push_back({cs,n-pos});pos=n;}
            else caps.push_back({cs,0});
            continue;
        }
        int found=-1;
        for(int cand=pos;cand+length<=n;++cand){
            if(st[cand]==pat[lb] && std::equal(pat.begin()+lb,pat.begin()+p,st.begin()+cand)){
                found=cand;break;
            }
        }
        if(found<0)return false;
        caps.push_back({cs,found-pos});
        pos=found+length;
    }
    finish=pos;
    return true;
}
static V slice(const V& st,Capture c){return V(st.begin()+c.pos,st.begin()+c.pos+c.len);}
static bool capture_memory(const V& st,const V& pat,const Captures& caps,Memory& mem){
    V top[2],bottom[2]; int ti=0,bi=0,ci=0;
    for(auto t:pat){
        if(t>=0)continue;
        if(ci>=(int)caps.size())break;
        if(t==-2 && ti<2)top[ti++]=slice(st,caps[ci]);
        else if(t==-3 && bi<2)bottom[bi++]=slice(st,caps[ci]);
        ++ci;
    }
    if(ti)mem.pair(top[0],ti>=2?top[1]:V(),true);
    if(bi)mem.pair(bottom[0],bi>=2?bottom[1]:V(),false);
    return ti||bi;
}
static V emit(const V& st,const V& rep,const Captures& caps,Memory& mem,int room,bool& mc){
    V out;
    out.reserve(std::min(room,(int)rep.size()+16));
    int wc=caps.size();
    for(I t:rep){
        if(t>=0){if((int)out.size()<room)out.push_back(t);continue;}
        if(t<=-128 && t>=-151){
            bool changed=false;V vals=mem.exec(t,changed);mc|=changed;
            if(vals.size()<=size_t(room-(int)out.size()))out.insert(out.end(),vals.begin(),vals.end());
            continue;
        }
        if(!wc)continue;
        int kind,ci;
        if(t>=-15){kind=1;ci=-t-1;}
        else if(t>=-31){kind=7;ci=-t-16;}
        else if(t>=-47){kind=2;ci=-t-32;}
        else if(t>=-111){int off=-t-48; kind=3+off/16;ci=off%16;}
        else continue;
        if(ci>=wc)ci=0;
        // Python's -0/negative index edge case isn't reachable for supported -1..-111.
        if(ci<0)continue;
        V vals=slice(st,caps[ci]);
        if(kind==2)std::reverse(vals.begin(),vals.end());
        else if(kind==7)std::sort(vals.begin(),vals.end());
        else if(kind==3){for(I& v:vals){if(v>=0 && v<512) v=(v+1)&511;}}
        else if(kind==4){for(I& v:vals){if(v>=0 && v<512) v=(v+511)&511;}}
        else if(kind==5){for(I& v:vals){if(v>=0 && v<512) v=(v*2)&511;}}
        else if(kind==6){for(I& v:vals){if(v<512) v=v/2;}}
        if(vals.size()<=size_t(room-(int)out.size()))out.insert(out.end(),vals.begin(),vals.end());
    }
    return out;
}
static bool replace_once(V& st,const R& rule,int max_output,Memory& mem,bool& mc,bool& of){
    auto& pat=rule.a;
    if(pat.empty() || st.empty())return false;
    // Exact no-false-negative necessary-token filter, refreshed lazily whenever
    // the visible state mutates. (Caller handles fast presence bitset.)
    V out;
    int prev=0,scan=0,finish=0;
    bool matched=false;
    Captures caps;
    while(scan<(int)st.size()){
        int pos=-1;
        if(pat[0]>=0){
            for(int c=scan;c<(int)st.size();++c){
                if(st[c]!=pat[0])continue;
                if(match_at(st,pat,c,finish,caps)){pos=c;break;}
            }
        }else{
            if(match_at(st,pat,scan,finish,caps))pos=scan;
        }
        if(pos<0)break;
        if(!matched){out.reserve(st.size());matched=true;}
        mc |= capture_memory(st,pat,caps,mem);
        out.insert(out.end(),st.begin()+prev,st.begin()+pos);
        bool changed=false;
        V vals=emit(st,rule.b,caps,mem,std::max(0,finish-pos),changed);
        mc|=changed;
        out.insert(out.end(),vals.begin(),vals.end());
        if((int)out.size()>max_output){of=true;return true;}
        prev=finish;scan=finish;
        // valid rules make progress. Defensive guard for malformed old checkpoints.
        if(finish<=pos)break;
    }
    if(!matched)return false;
    out.insert(out.end(),st.begin()+prev,st.end());
    if((int)out.size()>max_output){of=true;return true;}
    st.swap(out);
    return true;
}
static bool hidden_phase(V& st,const Memory& mem,int max_output,bool& of){
    const size_t n=std::min(mem.left.size(),mem.right.size());
    if(n==0)return false; // Most trajectories never capture hidden memory.
    bool changed=false;
    // Only copy when at least one mapping can possibly execute.
    bool viable=false;
    for(size_t k=0;k<n;++k){
        const V &a=mem.left[k], &b=mem.right[k];
        if(!a.empty() && !b.empty() && b.size()<=a.size() &&
           a.size()<=st.size() && a!=b){viable=true;break;}
    }
    if(!viable)return false;
    const V original=st;
    for(size_t k=0;k<n;++k){
        const V &needle=mem.left[k], &rep=mem.right[k];
        if(needle.empty() || rep.empty() || rep.size()>needle.size() || needle.size()>st.size() || needle==rep)continue;
        V out;out.reserve(st.size());
        int i=0,nn=st.size(),len=needle.size();
        while(i<=nn-len){
            if(st[i]==needle[0] && std::equal(needle.begin(),needle.end(),st.begin()+i)){
                out.insert(out.end(),rep.begin(),rep.end());
                changed=true;i+=len;
            }else{out.push_back(st[i]);++i;}
            if((int)out.size()>max_output){of=true;st=original;return changed;}
        }
        out.insert(out.end(),st.begin()+i,st.end());
        if((int)out.size()>max_output){of=true;st=original;return changed;}
        st.swap(out);
    }
    return changed;
}
static std::vector<R> parse_rules(const I* rule_meta,const I* rule_tokens,int rule_count){
    std::vector<R> rules(rule_count);
    for(int i=0;i<rule_count;++i){
        int p=rule_meta[i*4],pn=rule_meta[i*4+1],r=rule_meta[i*4+2],rn=rule_meta[i*4+3];
        if(p<0||pn<0||r<0||rn<0)return {};
        rules[i].a.assign(rule_tokens+p,rule_tokens+p+pn);
        rules[i].b.assign(rule_tokens+r,rule_tokens+r+rn);
        for(I t:rules[i].a)if(t>=0){rules[i].anchor=t;break;}
        for(size_t j=0;j+1<rules[i].a.size();++j){
            if(rules[i].a[j]>=0 && rules[i].a[j]<512 &&
               rules[i].a[j+1]>=0 && rules[i].a[j+1]<512){rules[i].anchor2=int(j);break;}
        }
    }
    return rules;
}

// A per-model inverted index of necessary literal tokens. It is *sound*:
// a rule can only fire if its first literal occurs in the current state.
// Rules without a literal, or with a literal outside the 0..512 fast domain,
// remain unconditional. This replaces a full loop over inactive rules with
// visits to only potentially firing rules and hidden phase boundaries.
struct RulePlan {
    std::array<std::vector<int>,513> postings;
    std::vector<uint64_t> unconditional;
    int count=0;
    bool use_index=false;
    explicit RulePlan(const std::vector<R>& rules):count(int(rules.size())) {
        unconditional.assign((rules.size()+63)/64,0);
        for(int i=0;i<count;++i) {
            I a=rules[i].anchor;
            if(a>=0 && a<=512)postings[a].push_back(i);
            else unconditional[i>>6]|=uint64_t(1)<<(i&63);
        }
        int distinct=0;
        for(const auto& list:postings)distinct+=!list.empty();
        // For concentrated, common alphabets an inverted index is a slowdown.
        // Keep the cheaper original token-presence scan for such models.
        use_index=(count>=1024 && distinct>=96);
    }
    bool activate(const V& state,std::vector<uint64_t>& mask)const {
        mask=unconditional;
        bool present[513]={};
        for(I token:state) {
            if(token<0 || token>512 || present[token])continue;
            present[token]=true;
            for(int ri:postings[token])mask[ri>>6]|=uint64_t(1)<<(ri&63);
        }
        int active_count=0;
        for(uint64_t word:mask)active_count+=__builtin_popcountll(word);
        // A dense index is not worth searching: a sequential scan is cheaper.
        return active_count>int(count*0.60);
    }
    static int next(const std::vector<uint64_t>& mask,int from,int count) {
        int w=from>>6;
        if(w>=int(mask.size()))return count;
        uint64_t v=mask[w] & (UINT64_MAX<<(from&63));
        while(true){
            if(v)return std::min(count,(w<<6)+int(__builtin_ctzll(v)));
            if(++w>=int(mask.size()))return count;
            v=mask[w];
        }
    }
};

// Shared exact interpreter for outer trajectories and fused inference.
// Both entrypoints call this function so hidden-memory and wildcard semantics
// cannot silently diverge between Spearman and the inference GA.
static void evaluate_one(const I* input_tokens, I input_length, int input_stride,
                         const I* embedding, const std::vector<R>& rules, const RulePlan& plan,
                         const std::vector<int>& phase_ends, int max_output,
                         float* features, double& baseline, I* stats,
                         I* final_lengths, I* final_tokens, int final_stride,
                         const double* readout=nullptr, double* prediction=nullptr,
                         int stop_before_rule=-1) {
    const int rule_count=(int)rules.size();

        const int rawlen=std::max(0,std::min(input_stride,(int)input_length));
        V st;st.reserve(rawlen);
        for(int k=0;k<rawlen;++k){I t=input_tokens[k];st.push_back(t>=0&&t<256?embedding[t]:512);}
        Memory mem(rawlen);
        std::unordered_set<std::string> seen;
        seen.insert(mem.key(st));
        const int limit=std::max(1,(int)std::ceil(2.0*std::sqrt((double)std::max(1,rawlen))));
        // Fixed 4096-bit pair Bloom filter, without per-token heap allocations.
        // False positives only cost a matcher call; false negatives are impossible.
        std::array<uint64_t,64> adjacent_bloom{};
        bool pairs_dirty=true;
        auto pair_hash=[](uint32_t key)->uint32_t{
            key^=key>>16;key*=0x7feb352dU;
            key^=key>>15;key*=0x846ca68bU;
            return key^(key>>16);
        };
        auto pair_bits=[](uint32_t hash){
            return std::array<uint32_t,2>{hash & 4095u, (hash>>12)&4095u};
        };
        auto have_pair=[&](const R& r){
            if(r.anchor2<0)return true;
            if(pairs_dirty){
                adjacent_bloom.fill(0);
                for(size_t k=1;k<st.size();++k){
                    int a=st[k-1],b=st[k];
                    if(a>=0 && a<512 && b>=0 && b<512){
                        auto bpos=pair_bits(pair_hash(uint32_t((a<<9)|b)));
                        adjacent_bloom[bpos[0]>>6]|=uint64_t(1)<<(bpos[0]&63u);
                        adjacent_bloom[bpos[1]>>6]|=uint64_t(1)<<(bpos[1]&63u);
                    }
                }
                pairs_dirty=false;
            }
            int off=r.anchor2;
            auto bpos=pair_bits(pair_hash(uint32_t((r.a[off]<<9)|r.a[off+1])));
            return (adjacent_bloom[bpos[0]>>6] & (uint64_t(1)<<(bpos[0]&63u))) &&
                   (adjacent_bloom[bpos[1]>>6] & (uint64_t(1)<<(bpos[1]&63u)));
        };
        std::vector<uint64_t> active;
        bool active_dirty=true,dense=false;
        std::array<uint64_t,8> literal_presence{};
        bool literal_dirty=true;
        for(int rnd=0;rnd<limit;++rnd){
            bool rewrote=false,mc=false,overflow=false;
            bool snapshot_reached=false;
            int phase_idx=0,ri=0;
            while(ri<rule_count){
                // Snapshot immediately before the chosen rule, including hidden
                // rewrite boundaries crossed by the prefix.  Only round zero.
                if (rnd==0 && stop_before_rule>=0 && ri>=stop_before_rule &&
                    (phase_idx>=int(phase_ends.size()) ||
                     phase_ends[phase_idx]>stop_before_rule)) {
                    snapshot_reached=true; break;
                }
                if(active_dirty){
                    if(plan.use_index)dense=plan.activate(st,active);
                    else {active.assign((rule_count+63)/64,UINT64_MAX);dense=true;}
                    active_dirty=false;
                }
                int next=dense ? ri : RulePlan::next(active,ri,rule_count);
                if (rnd==0 && stop_before_rule>=0) next=std::min(next,stop_before_rule);
                // Hidden sweep boundaries must execute exactly when passed,
                // even if no rule between here and the boundary can fire.
                int boundary=(phase_idx<(int)phase_ends.size()) ? phase_ends[phase_idx] : rule_count;
                if(boundary<=next && boundary>ri){
                    ri=boundary;
                }else if(next>ri){
                    ri=next;
                }
                if(ri==boundary && phase_idx<(int)phase_ends.size()){
                    bool ch=hidden_phase(st,mem,max_output,overflow);
                    ++phase_idx;
                    if(ch){rewrote=true;pairs_dirty=true;active_dirty=plan.use_index;literal_dirty=true;}
                    if(overflow){baseline-=1.0;stats[2]=1;break;}
                    continue;
                }
                if(ri>=rule_count)break;
                const R& rule=rules[ri];
                bool matched=false;
                // Pair filter and 1-token filter both have zero false negatives.
                bool may_fire=bool(active[ri>>6] & (uint64_t(1)<<(ri&63)));
                if(may_fire && !plan.use_index && rule.anchor>=0 && rule.anchor<512){
                    if(literal_dirty){
                        literal_presence.fill(0);
                        for(I tok:st)if(tok>=0 && tok<512)
                            literal_presence[tok>>6]|=uint64_t(1)<<(tok&63);
                        literal_dirty=false;
                    }
                    may_fire=bool(literal_presence[rule.anchor>>6] & (uint64_t(1)<<(rule.anchor&63)));
                }
                if(may_fire && have_pair(rule))
                    matched=replace_once(st,rule,max_output,mem,mc,overflow);
                if(matched){
                    if(features)features[ri]+=1.0f;
                    if(readout && prediction)*prediction+=readout[ri];
                    stats[1]++;rewrote=true;
                    pairs_dirty=true;active_dirty=plan.use_index;literal_dirty=true;
                }
                if(overflow){baseline-=1.0;stats[2]=1;break;}
                ++ri;
                while(phase_idx<(int)phase_ends.size() && ri>=phase_ends[phase_idx]){
                    bool ch=hidden_phase(st,mem,max_output,overflow);
                    if(ch){rewrote=true;pairs_dirty=true;active_dirty=plan.use_index;literal_dirty=true;}
                    if(overflow){baseline-=1.0;stats[2]=1;break;}
                    ++phase_idx;
                }
                if(overflow)break;
            }
            stats[0]=rnd+1;
            if(snapshot_reached) break;
            if(overflow || (!rewrote && !mc))break;
            auto key=mem.key(st);
            if(!seen.insert(std::move(key)).second){stats[3]=1;break;}
        }
        if(final_lengths)*final_lengths=(I)st.size();
        if(final_tokens && st.size()<=(size_t)final_stride)
            std::copy(st.begin(),st.end(),final_tokens);
}

extern "C" int replacer_cpu_batch(
    const I* input_tokens,const I* input_lengths,int sample_count,int input_stride,
    const I* embedding,const I* rule_meta,const I* rule_tokens,int rule_count,
    int max_output,int phases,int threads,
    float* features,double* baselines,I* stats,I* final_lengths,I* final_tokens, int final_stride) {
    if(!input_tokens || !input_lengths || !embedding || !rule_meta || !rule_tokens ||
       !features || !baselines || !stats || sample_count<0 || input_stride<1 || rule_count<0)return -1;
    auto rules=parse_rules(rule_meta,rule_tokens,rule_count);
    if((int)rules.size()!=rule_count)return -2;
    RulePlan plan(rules);
    int phase_count=std::max(1,std::min(phases,std::max(1,rule_count)));
    std::vector<int> phase_ends;
    for(int k=1;k<=phase_count && rule_count>0;++k)phase_ends.push_back((k*rule_count+phase_count-1)/phase_count);
    std::atomic<int> next_job{0};
    auto worker=[&](){
    for(int j;(j=next_job.fetch_add(1,std::memory_order_relaxed))<sample_count;){
        evaluate_one(input_tokens+j*input_stride,input_lengths[j],input_stride,
                     embedding,rules,plan,phase_ends,max_output,
                     features+j*rule_count,baselines[j],stats+j*4,
                     final_lengths?final_lengths+j:nullptr,
                     final_tokens?final_tokens+j*final_stride:nullptr,final_stride);
    }
    };
    int workers=std::max(1,std::min(threads,sample_count));
    std::vector<std::thread> pool;pool.reserve(std::max(0,workers-1));
    for(int t=1;t<workers;++t)pool.emplace_back(worker);
    worker();
    for(auto& t:pool)t.join();
    return 0;
}

// Native first-sweep prefix snapshots for stage-aware symbolic synthesis.
// Unlike running a truncated genome, this preserves the full genome's hidden
// phase boundaries and avoids a Python O(samples * rules * length) interpreter.
extern "C" int replacer_cpu_prefix_batch(
    const I* input_tokens,const I* input_lengths,int sample_count,int input_stride,
    const I* embedding,const I* rule_meta,const I* rule_tokens,int rule_count,
    int max_output,int phases,int threads,int stop_before_rule,
    I* out_lengths,I* out_tokens,int out_stride) {
    if(!input_tokens || !input_lengths || !embedding || !rule_meta || !rule_tokens ||
       !out_lengths || !out_tokens || sample_count<0 || input_stride<1 ||
       stop_before_rule<0 || stop_before_rule>=rule_count || out_stride<max_output)
        return -1;
    auto rules=parse_rules(rule_meta,rule_tokens,rule_count);
    if((int)rules.size()!=rule_count)return -2;
    RulePlan plan(rules);
    int phase_count=std::max(1,std::min(phases,std::max(1,rule_count)));
    std::vector<int> phase_ends;
    for(int k=1;k<=phase_count && rule_count>0;++k)
        phase_ends.push_back((k*rule_count+phase_count-1)/phase_count);
    std::atomic<int> next_job{0};
    auto worker=[&](){
        for(int j;(j=next_job.fetch_add(1,std::memory_order_relaxed))<sample_count;){
            double baseline=0.; I stats[4]={0,0,0,0};
            evaluate_one(input_tokens+j*input_stride,input_lengths[j],input_stride,
                         embedding,rules,plan,phase_ends,max_output,
                         nullptr,baseline,stats,out_lengths+j,
                         out_tokens+j*out_stride,out_stride,
                         nullptr,nullptr,stop_before_rule);
        }
    };
    int workers=std::max(1,std::min(threads,sample_count));
    std::vector<std::thread> pool;pool.reserve(std::max(0,workers-1));
    for(int t=1;t<workers;++t)pool.emplace_back(worker);
    worker();
    for(auto& t:pool)t.join();
    return 0;
}

// Inference context owns packed C++ models across all inner GA generations.
// This avoids Python per-genome dispatch and temporary worker pools per model.
// Long-lived worker group: a GA inference round uses the same immutable models
// for many candidate populations. Reusing its threads avoids fork/join overhead
// on every round without sharing hidden memory or state between trajectories.
struct InferenceContext {
    std::vector<std::vector<R>> models;
    std::vector<std::unique_ptr<RulePlan>> plans;
    std::vector<std::vector<int>> phase_ends;
    std::vector<V> embeddings;
    std::vector<std::vector<double>> readouts;
    int max_output=0, threads=1;
    const I* batch_tokens=nullptr;
    const I* batch_lengths=nullptr;
    int sample_count=0, stride=0, jobs=0;
    double* scores=nullptr;
    float* outer_features=nullptr;
    double* outer_baselines=nullptr;
    I* outer_stats=nullptr;
    int outer_feature_width=0;
    std::atomic<int> next{0};
    std::mutex gate;
    std::condition_variable start, done;
    int epoch=0, pending=0;
    bool stopping=false;
    std::vector<std::thread> workers;

    void process() {
        for(int job;(job=next.fetch_add(1,std::memory_order_relaxed))<jobs;){
            int m=job/sample_count, j=job%sample_count;
            double baseline=0.0,pred=0.0;
            I local_stats[4]={0,0,0,0};
            I* job_stats=outer_stats ? outer_stats+size_t(job)*4 : local_stats;
            float* job_features=outer_features ? outer_features+size_t(job)*outer_feature_width : nullptr;
            evaluate_one(batch_tokens+j*stride,batch_lengths[j],stride,
                         embeddings[m].data(),models[m],*plans[m],phase_ends[m],max_output,
                         job_features,baseline,job_stats,nullptr,nullptr,0,
                         outer_features ? nullptr : readouts[m].data(),
                         outer_features ? nullptr : &pred);
            if(outer_baselines)outer_baselines[job]=baseline;
            else scores[job]=pred+baseline;
        }
    }
    void start_workers() {
        for(int k=1;k<threads;++k){
            workers.emplace_back([this] {
                int seen_epoch=0;
                while(true){
                    std::unique_lock<std::mutex> lock(gate);
                    start.wait(lock,[&]{return stopping || epoch!=seen_epoch;});
                    if(stopping)return;
                    seen_epoch=epoch;
                    lock.unlock();
                    process();
                    lock.lock();
                    if(--pending==0)done.notify_one();
                }
            });
        }
    }
    int eval(const I* tokens,const I* lengths,int samples,int width,double* out) {
        size_t n_jobs=models.size()*size_t(samples);
        if(n_jobs>size_t(std::numeric_limits<int>::max()))return -2;
        if(n_jobs==0)return 0;
        {
            std::lock_guard<std::mutex> lock(gate);
            batch_tokens=tokens;batch_lengths=lengths;
            sample_count=samples;stride=width;scores=out;
            outer_features=nullptr;outer_baselines=nullptr;outer_stats=nullptr;
            jobs=int(n_jobs);next.store(0,std::memory_order_relaxed);
            pending=int(workers.size());++epoch;
        }
        start.notify_all();
        process();
        if(!workers.empty()){
            std::unique_lock<std::mutex> lock(gate);
            done.wait(lock,[&]{return pending==0;});
        }
        return 0;
    }
    int eval_outer(const I* tokens,const I* lengths,int samples,int width,
                   int feature_width,float* features,double* baselines,I* stats){
        if(!tokens || !lengths || !features || !baselines || !stats || feature_width<=0 || samples<0 || width<=0)
            return -1;
        size_t n_jobs=models.size()*size_t(samples);
        if(n_jobs>size_t(std::numeric_limits<int>::max()))return -2;
        if(n_jobs==0)return 0;
        for(const auto& model:models)if(model.size()>size_t(feature_width))return -3;
        {
            std::lock_guard<std::mutex> lock(gate);
            batch_tokens=tokens;batch_lengths=lengths;sample_count=samples;stride=width;
            outer_features=features;outer_baselines=baselines;outer_stats=stats;
            outer_feature_width=feature_width;scores=nullptr;
            jobs=int(n_jobs);next.store(0,std::memory_order_relaxed);
            pending=int(workers.size());++epoch;
        }
        start.notify_all();process();
        if(!workers.empty()){
            std::unique_lock<std::mutex> lock(gate);
            done.wait(lock,[&]{return pending==0;});
        }
        return 0;
    }
    ~InferenceContext(){
        {
            std::lock_guard<std::mutex> lock(gate);
            stopping=true;
        }
        start.notify_all();
        for(auto& th:workers)if(th.joinable())th.join();
    }
};
extern "C" void* replacer_cpu_infer_create(
    const I* embeddings,const I* model_offsets,const I* meta,const I* tokens,
    const double* readout,int model_count,int max_output,int phases,int threads) {
    if(!embeddings||!model_offsets||!meta||!tokens||!readout||model_count<=0||max_output<=0)return nullptr;
    try {
        auto ctx=std::make_unique<InferenceContext>();
        ctx->max_output=max_output;ctx->threads=std::max(1,std::min(threads,64));
        ctx->models.reserve(model_count);
        ctx->plans.reserve(model_count);
        ctx->phase_ends.reserve(model_count);
        ctx->embeddings.reserve(model_count);
        ctx->readouts.reserve(model_count);
        for(int m=0;m<model_count;++m){
            int begin=model_offsets[m],end=model_offsets[m+1],nr=end-begin;
            if(begin<0||nr<0)return nullptr;
            auto rules=parse_rules(meta+begin*4,tokens,nr);
            if((int)rules.size()!=nr)return nullptr;
            ctx->models.push_back(std::move(rules));
            ctx->plans.emplace_back(std::make_unique<RulePlan>(ctx->models.back()));
            int p=std::max(1,std::min(phases,std::max(1,nr)));
            std::vector<int> ends;
            for(int k=1;k<=p && nr>0;++k)ends.push_back((k*nr+p-1)/p);
            ctx->phase_ends.push_back(std::move(ends));
            ctx->embeddings.emplace_back(embeddings+m*256,embeddings+(m+1)*256);
            ctx->readouts.emplace_back(readout+begin,readout+end);
        }
        ctx->start_workers();
        return ctx.release();
    }catch(...){return nullptr;}
}
extern "C" void replacer_cpu_infer_destroy(void* handle) {
    delete reinterpret_cast<InferenceContext*>(handle);
}
extern "C" int replacer_cpu_infer_eval(void* handle,
    const I* input_tokens,const I* input_lengths,int sample_count,int stride,double* scores){
    if(!handle||!input_tokens||!input_lengths||!scores||sample_count<0||stride<=0)return -1;
    auto& ctx=*reinterpret_cast<InferenceContext*>(handle);
    return ctx.eval(input_tokens,input_lengths,sample_count,stride,scores);
}

// Population evaluation shares the same fused model x trajectory worker pool.
// Outputs [model][sample][rule] allow readout training in Python as before.
extern "C" int replacer_cpu_outer_eval(void* handle,
    const I* input_tokens,const I* input_lengths,int sample_count,int stride,
    int feature_width,float* features,double* baselines,I* stats){
    if(!handle)return -1;
    return reinterpret_cast<InferenceContext*>(handle)->eval_outer(
        input_tokens,input_lengths,sample_count,stride,feature_width,features,baselines,stats);
}
