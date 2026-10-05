import importlib.util, sys, random
spec=importlib.util.spec_from_file_location('v56','/mnt/data/minimal_replacer_gp_embedding_plot_v56.py')
m=importlib.util.module_from_spec(spec); sys.modules['v56']=m; spec.loader.exec_module(m)

def mk(base):
    rs=[]
    for i in range(20):
        r=m.Rule([base+i, base+i+1, base+i+2],[base+i], pack_filter_key=((base+i)*m._GP_INDEX_TOKENS+(base+i+1))*m._GP_INDEX_TOKENS+(base+i+2), pack_anchor_offset=0)
        rs.append(r)
    return m.Genome(rs, embedding=m.default_embedding())
for seed in range(1000):
    random.seed(seed)
    p1,p2=mk(1),mk(101)
    c=m.crossover(p1,p2,0.0)
    ids1={id(x) for x in p1.rules}; ids2={id(x) for x in p2.rules}
    assert all(id(x) in ids1 or id(x) in ids2 for x in c.rules)
    assert any(id(x) in ids2 for x in c.rules)
print('PASS: 1000 same-embedding crossovers allocate no recoded donor Rules')
