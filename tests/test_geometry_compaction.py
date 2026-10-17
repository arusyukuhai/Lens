"""DP state count depends on pattern geometry, not maximum capture length."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lens_gpu.model import SerialRule


def test_unbounded_geometry():
    a=SerialRule(256,max_pattern=4,max_replacement=4,max_capture=1,max_stars=3)
    b=SerialRule(256,max_pattern=4,max_replacement=4,max_capture=10000,max_stars=3)
    assert len(a._geometries)==len(b._geometries)==26
    assert a.max_capture is None and b.max_capture is None
    assert all(len(parts)-1<=3 and any(parts) for _,_,parts in a._geometries)
    print('26 STAR-position geometries independent of capture length: PASS')


if __name__=='__main__':test_unbounded_geometry()
