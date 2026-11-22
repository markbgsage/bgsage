"""Prepare a private encoder variant from the released checkout."""
import argparse
import gzip
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


BASE_SHA256 = "e6575e158754cb75b9be403e80122972fbc669d31c5bcd9b3f6afeb484e3c458"


def release_source():
    frozen = gzip.decompress((HERE / "release-encoding.cpp.gz").read_bytes())
    if hashlib.sha256(frozen).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen pre-experiment CPU encoder checksum mismatch")
    return frozen.decode("utf-8").replace("\r\n", "\n")


def variant(kind):
    code=release_source()
    addition=(HERE/"masks.inc").read_text(encoding="utf-8")
    if kind == "baseline":
        dest = ROOT / "build/cpu_feature_masks/baseline"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "encoding.cpp").write_text(code, encoding="utf-8")
        return
    code=code.replace('#include <algorithm>','#include <algorithm>\n#ifdef _MSC_VER\n#include <intrin.h>\n#endif')
    marker='// Compute all extended features for one side'
    assert code.count(marker)==1
    code=code.replace(marker,addition+'\n'+marker)
    if kind in ("pips","both","reuse"):
        code=code.replace('    out_piploss = compute_piploss(board);','    // Computed alongside the shot geometries in the outer encoder.')
    i=code.index('    // Hitting shots (expensive')
    j=code.index('    // Compute all GNUbg features',i)
    left='true' if kind in ('shots','both','reuse') else 'false'
    right='true' if kind in ('pips','both','reuse') else 'false'
    old=code[i:j]
    lines=(f'    ShotCounts p_counts=mask_shots<{left},{right}>(shot_position(board));\n'
           f'    ShotCounts o_counts=mask_shots<{left},{right}>(shot_position(flipped));\n')
    if left=='true':
        lines+=('    int p_shots=p_counts.single, o_shots=o_counts.single;\n'
                '    int p_dbl_shots=p_counts.twice, o_dbl_shots=o_counts.twice;\n')
    else: lines+=old
    code=code[:i]+lines+'\n'+code[j:]
    if right=='true':
        k=code.index('    // === Player 1 features')
        code=code[:k]+('    p_piploss=board[25] ? compute_piploss(board) : p_counts.pips;\n'
                       '    o_piploss=flipped[25] ? compute_piploss(flipped) : o_counts.pips;\n\n')+code[k:]
    if kind=="reuse":
        a=code.index('    // Bits 0..23 are blocked points')
        b=code.index('    // Initialize escape tables once',a)
        code=code[:a]+'''    const Board flipped=flip(board);
    const auto player=shot_position(board), opponent=shot_position(flipped);
    const unsigned player_blocks=player.anchors, opp_blocks=opponent.anchors;
    const int p_opp_back=highest_hit_point(player.blots|player.blocked);
    const int o_opp_back=highest_hit_point(opponent.blots|opponent.blocked);

'''+code[b:]
        a=code.index('    auto exit_probs =',code.index('std::array<float, EXTENDED_CONTACT_INPUTS> compute_extended_contact_inputs('))
        b=code.index('    int p_shots=p_counts.single',a)
        code=code[:a]+'''    const int pc=blocked_home(player.blocked), oc=blocked_home(opponent.blocked);
    BarExitProbs exit_probs{pc*pc*(1.0f/36.0f),oc*oc*(1.0f/36.0f)};
    ForwardAnchorPoints fwd_anchors{first_front_anchor(player.anchors),first_front_anchor(opponent.anchors)};
    const int p_max_pt=highest_hit_point(player.men), o_max_pt=highest_hit_point(opponent.men);
    const int p_max_anch=highest_hit_point(player.anchors), o_max_anch=highest_hit_point(opponent.anchors);
    const int p_back_esc=p_max_pt<19 ? 36 : back_escapes(board);
    const int o_back_esc=o_max_pt<19 ? 36 : back_escapes(flipped);
    const auto p_counts=player.bar ? mask_shots<true,false>(player) : mask_shots<true,true>(player);
    const auto o_counts=opponent.bar ? mask_shots<true,false>(opponent) : mask_shots<true,true>(opponent);
'''+code[b:]
        code=code.replace('    out_backbone_val = backbone(board);',
                          '    out_backbone_val = backbone_from_anchors(board,player_blocks);')
    dest=ROOT/"build/cpu_feature_masks"/kind
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"encoding.cpp").write_text(code,encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("variants",nargs="+",choices=("baseline","shots","pips","both","reuse"))
    for tag in p.parse_args().variants: variant(tag)
