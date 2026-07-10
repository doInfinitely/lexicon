"""Do the learned operators behave like an algebra? Structural probes.

None of this is about storage. Each probe asks a question about language
that the operators can answer, and that a bag of embeddings cannot:

  translation   Is a relation a constant displacement (the word2vec premise,
                king-man+woman=queen), or genuinely word-dependent? Measured
                as the fraction of displacement variance explained by the
                relation's mean offset.
  involution    Did the inverse operator learn to actually invert? Round trip
                f_r-1(f_r(x)) should return x.
  determinism   Is the relation one-to-many? If many sources map into one
                target region, the forward map collapses; the inverse cannot
                be a function. Measured by target entropy / collision rate.
  commutation   Do operators commute? plural(past(x)) vs past(plural(x)).
                Commuting operators factor the lexicon into independent axes.
  identity      What does an operator do to words outside its domain?
                A well-behaved morphological operator should be near-identity
                on words it does not apply to.

!! CORRECTION. The headline "morphological operators are idempotent (0.936),
lexicographic operators are transitive (0.779)" was a max-vs-min cherry-pick.
Category MEANS are inflectional 0.918 vs lexicographic 0.844 -- the split is
real but far smaller. Worse, 0.779 is an IDEMPOTENCE score; nothing here
measures transitivity, so calling those operators "transitive" was an
unmeasured interpretation. And for a true involution cos(f(f(x)),f(x)) =
cos(x,f(x)), so probe_idempotence CANNOT distinguish an involution from a
projection -- it is uninformative for antonymy, the case that matters most.
"""
import json, collections
import torch
import torch.nn.functional as F

from lexicon.model import LexiconSpace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT = {"I": "inflectional", "D": "derivational",
       "E": "encyclopedic", "L": "lexicographic"}


def load(ckpt_path="real/lexicon_space.pt"):
    ck = torch.load(ckpt_path, weights_only=False)
    vocab = ck["vocab"]
    space = LexiconSpace(ck["relation_names"]).to(DEVICE)
    space.load_state_dict(ck["state_dict"])
    space.eval()
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    with torch.no_grad():
        table = F.normalize(space.adapter(P), dim=-1)
    return ck, space, vocab, {w: i for i, w in enumerate(vocab)}, table


def relation_pairs(vocab_set):
    rels = json.load(open("harbor/workspace/data/relations.json"))
    out = {}
    for r, meta in rels.items():
        pl = [(p["source"], p["target"]) for p in meta["pairs"]
              if p["source"] in vocab_set and p["target"] in vocab_set
              and p["source"] != p["target"]]
        if pl:
            out[r] = pl
    return out


@torch.no_grad()
def probe_translation(space, table, widx, pairs):
    """Fraction of displacement variance explained by the mean offset.
    1.0 => the relation IS a constant vector (pure word2vec translation).
    0.0 => the displacement depends entirely on the word."""
    rows = {}
    for r, pl in pairs.items():
        s = table[[widx[a] for a, _ in pl]]
        t = table[[widx[b] for _, b in pl]]
        d = t - s                                   # observed displacements
        mu = d.mean(0, keepdim=True)
        ss_tot = (d - d.mean(0, keepdim=True)).pow(2).sum()
        ss_mu = (d - mu).pow(2).sum()               # residual after mean offset
        explained = 1 - ss_mu / (d.pow(2).sum() + 1e-9)
        # cosine spread: how aligned are the displacement directions?
        dn = F.normalize(d, dim=-1)
        align = (dn @ F.normalize(mu, dim=-1).T).mean().item()
        rows[r] = {"offset_explains": round(explained.item(), 3),
                   "direction_alignment": round(align, 3),
                   "n": len(pl)}
    return rows


@torch.no_grad()
def probe_involution(space, table, widx, pairs):
    """Round trip: does the learned inverse invert the learned forward map?"""
    rows = {}
    for r, pl in pairs.items():
        if r + "_inv" not in space.ops.rel_index:
            continue
        x = table[[widx[a] for a, _ in pl]]
        fwd = F.normalize(space.ops.apply_named(r, x), dim=-1)
        back = F.normalize(space.ops.apply_named(r + "_inv", fwd), dim=-1)
        rows[r] = round(F.cosine_similarity(back, x, dim=-1).mean().item(), 3)
    return rows


@torch.no_grad()
def probe_determinism(space, table, widx, pairs, vocab):
    """One-to-many-ness. Apply the forward operator to every source; count how
    many distinct words the outputs decode to. collisions < 1 means several
    sources land on the same target -- the map is many-to-one, so its inverse
    cannot be a single-valued function."""
    rows = {}
    for r, pl in pairs.items():
        x = table[[widx[a] for a, _ in pl]]
        out = F.normalize(space.ops.apply_named(r, x), dim=-1)
        dec = (out @ table.T).argmax(1).tolist()
        distinct = len(set(dec))
        # how many distinct GOLD targets do these sources have?
        gold = len({b for _, b in pl})
        rows[r] = {"decoded_distinct": distinct, "gold_distinct": gold,
                   "n_sources": len(pl),
                   "fan_in": round(len(pl) / max(gold, 1), 2)}
    return rows


@torch.no_grad()
def probe_commutation(space, table, widx, pairs, top=8):
    """Do operators commute on a common domain?"""
    rels = sorted(pairs, key=lambda r: -len(pairs[r]))[:top]
    M = {}
    sample = table[torch.randperm(len(table), device=DEVICE)[:256]]
    for a in rels:
        for b in rels:
            if a >= b:
                continue
            ab = F.normalize(space.ops.apply_named(
                a, space.ops.apply_named(b, sample)), dim=-1)
            ba = F.normalize(space.ops.apply_named(
                b, space.ops.apply_named(a, sample)), dim=-1)
            M[(a, b)] = round(F.cosine_similarity(ab, ba, dim=-1).mean().item(), 3)
    return M


@torch.no_grad()
def probe_domain(space, table, widx, pairs):
    """How near-identity is an operator on its own sources vs. on random words?
    A morphological operator that 'knows its domain' should move its sources a
    lot and leave unrelated words comparatively alone."""
    rng = torch.randperm(len(table), device=DEVICE)[:256]
    rows = {}
    for r, pl in pairs.items():
        x = table[[widx[a] for a, _ in pl]]
        movement_in = 1 - F.cosine_similarity(
            F.normalize(space.ops.apply_named(r, x), dim=-1), x, dim=-1).mean()
        y = table[rng]
        movement_out = 1 - F.cosine_similarity(
            F.normalize(space.ops.apply_named(r, y), dim=-1), y, dim=-1).mean()
        rows[r] = {"moves_own_sources": round(movement_in.item(), 3),
                   "moves_random_words": round(movement_out.item(), 3),
                   "selectivity": round((movement_in / (movement_out + 1e-6)).item(), 2)}
    return rows


@torch.no_grad()
def probe_idempotence(space, table, widx, pairs):
    """Is f_r a projection onto a feature value rather than a group element?
    If so, applying it twice changes nothing: f(f(x)) == f(x). A translation,
    by contrast, keeps translating: pluralizing twice would double the offset.
    Compared against the null of applying it once, cos(f(x), x)."""
    rows = {}
    for r, pl in pairs.items():
        x = table[[widx[a] for a, _ in pl]]
        f1 = F.normalize(space.ops.apply_named(r, x), dim=-1)
        f2 = F.normalize(space.ops.apply_named(r, f1), dim=-1)
        rows[r] = {"idempotence": round(F.cosine_similarity(f2, f1, dim=-1).mean().item(), 3),
                   "moved_at_all": round(1 - F.cosine_similarity(f1, x, dim=-1).mean().item(), 3)}
    return rows


def main():
    ck, space, vocab, widx, table = load()
    pairs = relation_pairs(set(vocab))

    trans = probe_translation(space, table, widx, pairs)
    inv = probe_involution(space, table, widx, pairs)
    det = probe_determinism(space, table, widx, pairs, vocab)
    dom = probe_domain(space, table, widx, pairs)

    print("=" * 100)
    print("IS EACH RELATION A CONSTANT DISPLACEMENT?  (the word2vec premise)")
    print("=" * 100)
    print(f"{'relation':<34}{'cat':<16}{'offset explains':>16}"
          f"{'dir. align':>12}{'fan-in':>9}{'selectivity':>13}")
    for r in sorted(trans, key=lambda r: -trans[r]["offset_explains"]):
        print(f"{r:<34}{CAT[r[0]]:<16}{trans[r]['offset_explains']:>16.3f}"
              f"{trans[r]['direction_alignment']:>12.3f}"
              f"{det[r]['fan_in']:>9.2f}{dom[r]['selectivity']:>13.2f}")

    bycat = collections.defaultdict(list)
    for r, v in trans.items():
        bycat[CAT[r[0]]].append(v["offset_explains"])
    print("\nmean 'offset explains' by category:")
    for c, v in sorted(bycat.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {c:<18}{sum(v)/len(v):.3f}")

    print("\n" + "=" * 100)
    print("DO THE LEARNED INVERSES ACTUALLY INVERT?  cos(f_inv(f(x)), x)")
    print("=" * 100)
    iv = sorted(inv.items(), key=lambda kv: -kv[1])
    for r, v in iv[:5]:
        print(f"  best  {r:<34}{v:.3f}")
    for r, v in iv[-5:]:
        print(f"  worst {r:<34}{v:.3f}")
    print(f"  mean round-trip fidelity: {sum(inv.values())/len(inv):.3f}")

    print("\n" + "=" * 100)
    print("WHICH RELATIONS ARE ONE-TO-MANY?  (fan-in > 1 => inverse is not a function)")
    print("=" * 100)
    for r in sorted(det, key=lambda r: -det[r]["fan_in"])[:10]:
        d = det[r]
        print(f"  {r:<34} {d['n_sources']:>3} sources -> {d['gold_distinct']:>3} "
              f"distinct targets   fan-in {d['fan_in']:.2f}   "
              f"model decodes {d['decoded_distinct']:>3} distinct")

    print("\n" + "=" * 100)
    print("DO OPERATORS COMMUTE?  cos(f_a(f_b(x)), f_b(f_a(x)))")
    print("=" * 100)
    com = probe_commutation(space, table, widx, pairs)
    cs = sorted(com.items(), key=lambda kv: -kv[1])
    for (a, b), v in cs[:5]:
        print(f"  commute    {a:<30}{b:<30}{v:.3f}")
    for (a, b), v in cs[-5:]:
        print(f"  do NOT     {a:<30}{b:<30}{v:.3f}")

    print("\n" + "=" * 100)
    print("ARE OPERATORS PROJECTIONS?  f(f(x)) vs f(x)   [1.0 = idempotent]")
    print("=" * 100)
    idem = probe_idempotence(space, table, widx, pairs)
    im = sorted(idem.items(), key=lambda kv: -kv[1]["idempotence"])
    for r, v in im[:4]:
        print(f"  most idempotent  {r:<32}{v['idempotence']:.3f}  "
              f"(first application moved it {v['moved_at_all']:.3f})")
    for r, v in im[-4:]:
        print(f"  least            {r:<32}{v['idempotence']:.3f}  "
              f"(first application moved it {v['moved_at_all']:.3f})")
    print(f"  mean idempotence: {sum(v['idempotence'] for v in idem.values())/len(idem):.3f}")

    json.dump({"translation": trans, "involution": inv, "determinism": det,
               "domain": dom, "idempotence": idem,
               "commutation": {f"{a}|{b}": v for (a, b), v in com.items()}},
              open("real/algebra.json", "w"), indent=1)
    print("\nwrote real/algebra.json")


if __name__ == "__main__":
    main()
