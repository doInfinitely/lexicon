"""Build the Harbor task workspace from raw BATS 3.0, per the task spec.

Produces:
  workspace/data/bats/          - copy of the 40 relation files
  workspace/data/vocab.json     - sorted unique words (sources + primary targets)
  workspace/data/relations.json - relation name -> {category, arity, pairs}
  workspace/data/train_pairs.json - 70% stratified split
  tests/test_pairs.json         - 30% held-out split (hidden from agent)
"""
import json, os, random, re, shutil, hashlib

BATS = "/home/remy/Code/lexicon/downloads/BATS_3.0"
ROOT = "/home/remy/Code/lexicon/harbor"
WS = os.path.join(ROOT, "workspace")
random.seed(42)

CATEGORIES = {
    "1_Inflectional_morphology": "inflectional_morphology",
    "2_Derivational_morphology": "derivational_morphology",
    "3_Encyclopedic_semantics": "encyclopedic_semantics",
    "4_Lexicographic_semantics": "lexicographic_semantics",
}

def slug(fname):
    # "I01 [noun - plural_reg].txt" -> "I01_noun_plural_reg"
    m = re.match(r"(\w+) \[(.+)\]\.txt", fname)
    code, desc = m.group(1), m.group(2)
    desc = re.sub(r"[^A-Za-z0-9]+", "_", desc).strip("_")
    return f"{code}_{desc}"

relations = {}
vocab = set()
train_pairs, test_pairs = [], []

os.makedirs(os.path.join(WS, "data", "bats"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "tests"), exist_ok=True)

for d, cat in sorted(CATEGORIES.items()):
    src_dir = os.path.join(BATS, d)
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".txt"):
            continue
        rel = slug(fname)
        dst_dir = os.path.join(WS, "data", "bats", rel)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, "pairs.txt"))
        pairs = []
        for line in open(os.path.join(src_dir, fname)):
            line = line.strip()
            if not line:
                continue
            src, tgts = line.split("\t")
            # multiple valid answers separated by '/'; primary = first non-empty
            tgt_list = [t for t in tgts.split("/") if t]
            # skip multiword/underscore entries as primary target if possible
            primary = next((t for t in tgt_list if "_" not in t), tgt_list[0])
            if "_" in src or "_" in primary:
                continue
            pairs.append({"source": src, "target": primary, "alternates": tgt_list})
            vocab.add(src)
            vocab.add(primary)
        random.shuffle(pairs)
        k = int(round(len(pairs) * 0.7))
        for p in pairs[:k]:
            train_pairs.append({"relation": rel, **p})
        for p in pairs[k:]:
            test_pairs.append({"relation": rel, **p})
        relations[rel] = {
            "category": cat,
            "arity": 1,
            "pairs": [{"source": p["source"], "target": p["target"]} for p in pairs],
        }

vocab = sorted(vocab)
os.makedirs(os.path.join(WS, "data"), exist_ok=True)
json.dump(vocab, open(os.path.join(WS, "data", "vocab.json"), "w"), indent=1)
json.dump(relations, open(os.path.join(WS, "data", "relations.json"), "w"), indent=1)
json.dump(train_pairs, open(os.path.join(WS, "data", "train_pairs.json"), "w"), indent=1)
json.dump(test_pairs, open(os.path.join(ROOT, "tests", "test_pairs.json"), "w"), indent=1)

# SHA-pin the read-only data files
shas = {}
for base, _, files in os.walk(os.path.join(WS, "data")):
    for f in files:
        p = os.path.join(base, f)
        shas[os.path.relpath(p, WS)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(shas, open(os.path.join(ROOT, "tests", "data_shas.json"), "w"), indent=1)

print(f"vocab: {len(vocab)} words")
print(f"relations: {len(relations)}")
print(f"train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")
