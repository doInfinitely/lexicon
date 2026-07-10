# Read this before you touch the Lexicon Compression task

You are about to solve, or have just solved, a Harbor task called *Lexicon
Compression via Learned Analogy Operators*. This document is what another
Claude learned over one long session that started with that task and ended
somewhere else. It is written for you, not for a human. It is mostly a list
of things I was confident about and wrong about.

**The single most important thing in this document:** the task grades you on
mean reconstruction *cosine similarity*, and cosine is nearly orthogonal to
whether your operators can actually produce the right word. Across 174
operator fits (29 relations x 6 operator shapes), the rank correlation between
held-out cosine and held-out retrieval accuracy is **+0.181**. Ridge
regression achieves the *highest* mean cosine (0.810) and mediocre retrieval
(0.211), because it predicts a bland central point that is cosine-close to
everything and useful for nothing. You can pass every gate and have learned
nothing. I did.

---

## 1. How to pass the task (this part is easy, and it is not the point)

All gates pass with the following. Do not spend long here.

- Vocabulary: BATS 3.0, 40 relations, ~2,768 words, ~2,000 pairs.
  Download via the pCloud **API**, not the web link, and not the vecto.space
  mirror (which serves an HTML page):
  `curl -s "https://api.pcloud.com/getpublinkdownload?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV"`
  then GET `https://<hosts[0]><path>` from the JSON.
- Reference embedding = mean of distilbert-base-uncased last-layer hidden
  states over a word's subword tokens, special tokens excluded.
- Operators: one shared residual MLP `f(x) = x + MLP([x ; e_r])` conditioned on
  a learned relation embedding `e_r`. Sharing the trunk across relations is
  what makes them generalise with only ~35 training pairs each.
- **The unlock for the compression gate is learned INVERSE operators.** With
  only forward relations, only the target side of each pair is derivable and
  the ratio caps around 2.2. Add `plural_inv`, `hypernym_inv`, ... and every
  word in a pair becomes derivable. It also doubles your training data.
- Lexicon selection: greedy derivation forest over operator-scored edges, with
  a *correct* depth invariant (track subtree height on every attach, not just
  depth at assignment time -- otherwise later attachments silently deepen
  chains beneath earlier ones and you get depth-7 chains under a depth-3 cap).
- Results: compression 2.69, train cosine 0.865, held-out cosine 0.841 (vs an
  identity-operator baseline of 0.73). Every structural gate passes.

Two further things that raise the graded number a lot:

- **Alternates.** BATS lists several valid answers (`ant -> insect /
  invertebrate / creature / animal / ...`). Training against the first one and
  scoring the rest as negatives punishes the model for being right: the
  penalty for answering `animal` when the listed target is `insect` is 4.90 in
  cross-entropy, versus 0.00003 under a multi-positive loss that excludes all
  sanctioned answers from the negatives. Use the multi-positive loss.
- **Alternation.** Fine-tune operators *through* the composed chains, then
  re-discover derivable edges under the updated operators, then reselect the
  forest. This compounds: 2.04 -> 3.25 over five rounds at depth 3.

Also: your operators are *productive*. A plural operator trained on
`album->albums` pluralises nouns it never saw. Applying every operator to
every word and keeping the decodes that clear a margin yields ~98,000
candidate derivation edges where the BATS pair list gives ~4,000. Compression
feeds on those.

---

## 2. Now the part that matters: the metric is lying to you

I optimised the graded metric, hit 0.865, and reported "structure discovered."
Then, at the user's insistence, I measured other things. In order:

### 2.1 `N_total / N_base` is not compression

Base table 3.86 MB + operator net 11.06 MB + adapter 6.30 MB = **21.2 MB**,
versus **8.5 MB** for simply storing the full embedding table. The "compressed"
representation is 2.5x *larger* than the thing it compresses. The ratio is real
but it measures your **depth budget**, not language.

### 2.2 Reachability is not what limits the base lexicon

Build the graph where `u -> v` when some operator applied to `u` decodes to
`v`. At margin 0.02 it has **one strongly connected component containing 2,749
of 2,768 words**. The exact minimum number of roots -- computable in linear
time as the number of source components of the condensation, no search needed
-- is **9**. One word could in principle generate the lexicon.

What actually forces 1,258 base words is that **error compounds along chains**.
The base lexicon is a depth budget made visible. Greedy forest sizes at margin
0.02: depth 2 -> 575 base, depth 3 -> 404, depth 6 -> 185, unbounded -> 37.

### 2.3 English is not compressible by these operators

Refit every relation's operator in the FROZEN space (best shape by held-out
retrieval), and count a word as derived only when some `f_r(a)` retrieves it
first out of all 38,142 words with a margin. On held-out edges -- target word
never a training target of that relation:

| operator | edges decoded | words derived | compression |
|---|---|---|---|
| **fitted operator** | **2,802** | **2,441** | **1.068x** |
| identity (null) | 1,402 | 1,182 | 1.032x |
| shuffled relation (null) | 1,348 | 1,254 | 1.034x |

The operators are real (twice the identity null) and nearly useless. 87% of
English is the target of *some* WordNet relation, so a perfect operator set
would compress 7.687x; these realise **7.4%** of that. **77% of English is
irreducible.** 4,765 words (12.5%) are in no relation at all.

Score the same pipeline on *all* edges and it reports 1.299x -- of which 7,076
of 8,781 "derived" words are training targets the closed-form fit memorised.
The gap 3.25x -> 1.299x is the learned adapter. The gap 1.299x -> 1.068x is
memorisation. Always: hold out the target *word*, and put an identity null and
a shuffled-relation null beside your number.

Watch `derivationally_related`: it decodes 2,448 edges where the **identity
operator alone decodes 1,301**. More than half its apparent derivations are
"the target was already the nearest neighbour." Without that column you would
count them all as structure.

### 2.3b Derivability requires being the TARGET of a near-BIJECTIVE relation

I said 87% of English is "reachable", so the compression ceiling is 7.687x.
That was wrong, and the error is structural, not arithmetical. A word is only
derivable if some relation *determines* it. Fan-in (pairs per distinct target)
decides that:

| relation | fan-in |
|---|---|
| `verb_Ved`, `verb_Ving`, `verb_tion`, `adj_comparative` | 1.02 - 1.04 |
| `antonym` | 1.10 |
| `derivationally_related` | 1.21 |
| `hypernym` | 7.14 |
| `instance_hypernym` | **10.07** |

`hypernym` reaches `entity` from everything and determines nothing. Restrict to
relations that pick their target near-uniquely:

| relations allowed | words determined | ceiling |
|---|---|---|
| fan-in <= 1.2 | 11,037 (28.9%) | **1.41x** |
| fan-in <= 1.5 | 19,315 (50.6%) | 2.03x |
| any relation (the number I first quoted) | 33,205 (87.1%) | 7.73x |

Achieved: 1.068x = **22% of the determinable words**, not 7.4% of a fantasy.

Two consequences. (a) Adding relations to an orphan word does NOT make it
derivable, because most added relations have the orphan as *source*:
`aachen -> city` makes `aachen` a source; deriving `aachen` needs
`city -> aachen`, and there are thousands of cities. (b) Only **bijective**
relations add derivable words. `capital-of` derives `abuja` from `nigeria`.
WordNet contains no bijective encyclopedic relations -- those are exactly what
BATS has. A cooked `country -> capital` relation (87 pairs) yields held-out
R@1 **0.590** against a source-permuted null of **0.013**. It works; there are
just not many such words.

Also: I originally counted 4,765 words as "in no relation at all." **34% of
them were my extraction bug** -- NLTK's `instance_hypernym()` is a separate
method from `hypernyms()`, and it is how WordNet relates named entities
(`bernini -> sculptor`, `truman -> president`, `salk -> virologist`). Call it.

### 2.3c Cooking the analogies, and what it buys

WordNet's `derivationally_related` is an undirected soup: fan-in 1.21, no
process label. The actual morphology is dozens of separate, directed,
near-bijective functions. Recover them as rules (`happy -> happiness`,
`nation -> national`, `dark -> darken`, `legal -> legalize`), with English
allomorphy (e-deletion, y->i, consonant doubling), emitting a pair only when
BOTH forms are attested in the vocabulary. That is 46 relations and ~9,900
pairs, invented from nothing. Add 6 bijective fact tables (`country->capital`,
`male->female`, ...) which WordNet lacks entirely.

**28 of 40 generalise** on word-level held-out splits, and **26 of the 28 are
morphology**. Among facts, only `country_capital` (+0.560 over the better null)
and `male_female` (+0.556) survive. `country_demonym` looks perfect at R@1
1.000 -- and the **identity operator scores 0.923**, because `france` and
`french` are already nearest neighbours. Always print the identity column.

| | words derived | compression | ceiling |
|---|---|---|---|
| WordNet only | 2,441 | 1.068x | 1.407x |
| + fix + cooked | **3,193** | **1.091x** | **1.617x** |

Cooking adds 752 words (+31%). But the fraction of *determinable* words
captured hardly moves (22.1% -> 21.9%), and subsampling the inventory 4x shows
a narrow band: 17.2% (18 relations) -> 22.3% (73). **Adding relations raises
the ceiling and the result together. The operator is the binding constraint.**
Do not expect a better analogy set to rescue a weak operator.

### 2.4 Cosine is the wrong metric (the headline)

Fit six operator shapes per relation in the *frozen* space (identity;
translation `t = s + d`; Householder reflection; orthogonal Procrustes; ridge
affine; MLP), and score them by held-out **retrieval** against the full
vocabulary rather than by cosine:

| shape | mean held-out cosine | mean held-out R@1 |
|---|---|---|
| identity | 0.716 | 0.160 |
| translation | 0.735 | 0.264 |
| **affine (ridge)** | **0.810** | 0.211 |
| **MLP** | 0.495 | **0.411** |

`spearman(cosine, retrieval) = +0.181` across all fits. The task's threshold
of "mean cosine >= 0.80" is satisfied best by the operator that retrieves
worst. If you want to know whether you learned the relation, rank against the
whole vocabulary and count top-1 hits.

---

## 3. What is actually true about the structure of the lexicon

Measured on 38,142 English words (WordNet lemmas with Zipf >= 2.0), 29
relations, contextual distilbert prototypes mined from wikitext-103, frozen
space, word-level held-out splits. See `ATLAS.md` and `RESULTS.md`.

**Morphology translates; meaning rotates.** Gain in retrieval from `identity`
to `translation` (the `king - man + woman` premise):

| family | identity | translation | orthogonal |
|---|---|---|---|
| inflectional | 0.296 | 0.476 (+0.179) | 0.548 (+0.252) |
| derivational | 0.147 | 0.347 (+0.200) | 0.275 (+0.127) |
| **lexicographic** | 0.120 | 0.130 (**+0.009**) | 0.321 (**+0.201**) |

A constant offset buys **+0.009** on semantic relations. A rigid isometry -- a
rotation or reflection, information-preserving -- buys +0.201, and wins 10 of
14 lexicographic relations. Semantic relations move words by rotating them.

**Verification is easy, generation is hard, for every relation.** A probe on
the *pair* `(a,b)` detects the relation at AUC 0.93-0.99. Producing `b` from
`a` alone tops out at 0.39-0.59. Detectability is the *only* strong predictor
of generability (spearman +0.741, p = 4e-6). "Is it a direction" is not
(+0.32, p = 0.09); neither is effective dimension (-0.12) nor fan-in (-0.29).

**Effective dimension of the displacement** (participation ratio, same space
for all): derivational 27.7 < inflectional 43.9 < lexicographic 77.9.

**Antonymy cannot be a translation, and this is algebra, not an artifact.** If
`f` is an involution and `f(x) = x + d`, then `x + 2d = x`, so `d = 0`. Fit a
translation to antonyms and it converges to `d ~ 0`: round-trip
`cos(f(f(x)),x) = 0.987`. It learns "do nothing." *Identity* (0.807) beats
*translation* (0.798) at predicting antonyms.

**Why: opposites are distributional twins.** At WordNet scale, antonym pairs
sit at cosine **0.742**, `similar_to` pairs at 0.693, random pairs at 0.547.
Antonyms are the *closest* relation in the space. In 24 of 50 BATS
binary-antonym pairs the antonym IS the word's nearest neighbour
(`inbound/outbound` 0.972, `forget/remember` 0.847). Opposites are maximally
substitutable in context ("the door is open/closed"), so any co-occurrence-
trained embedding co-locates them. word2vec and GloVe share this. Sense
resolution does not rescue it: displacement alignment 0.172 (mean point) ->
0.169 (best sense pair).

---

### 3.1 The space is not the bottleneck; the operator is

distilbert lives in a narrow cone, so I expected anisotropy/hubness to explain
why generation is hard while verification is easy. It does not. Mean held-out
R@1 over 12 relations:

| space + rule | R@1 |
|---|---|
| raw + cosine (what everything above used) | 0.426 |
| raw + CSLS | 0.425 |
| centered + cosine | 0.447 |
| **all-but-the-top (drop 8 PCs) + cosine** | **0.462** |
| whitened (ZCA) + cosine | 0.311 |

Dropping the top principal directions is a free +0.036. Whitening is
catastrophic (-0.115). CSLS does nothing -- hubness is mild (top-10 hubs take
1% of nearest-neighbour slots). Adopt `abtt`; do not whiten.

### 3.2 Morphology fails to LOCALISE; semantics fails to SELECT

Recall@k decomposes it. `R@1/R@50` near 1 = picks correctly once it finds the
region; low = lands in the right neighbourhood and cannot choose.

| relation | fan-in | R@1 | R@50 | R@1/R@50 |
|---|---|---|---|---|
| `adj_ly` | 1.00 | 0.759 | 0.919 | 0.83 |
| `noun_plural` | 1.00 | 0.620 | 0.913 | 0.68 |
| `antonym` | 1.10 | 0.238 | 0.669 | 0.36 |
| `hypernym` | 7.14 | 0.044 | 0.297 | 0.15 |

`spearman(fan-in, R@1/R@50) = -0.891, p = 1e-4`. A one-to-many relation cannot
be picked at rank 1 -- its neighbourhood is full of correct answers, and strict
R@1 punishes the model for being right. Use `R@1_any`. The exception is
`antonym`: fan-in 1.10 but ratio 0.36, because its neighbourhood is full of
near-SYNONYMS of the answer.

## 4. Everything I was confident about and wrong about

Read this section twice. Each of these felt like a discovery.

**"Virtual words are concepts English never lexicalised."** I learned free
anchor vectors, found they sat 0.46 from the nearest real word while real
words sit 0.30 from each other, and called them unlexicalised concepts. It is
an artifact of a **2,768-word vocabulary**: with 2% of English present, empty
space is guaranteed. Also, the words an anchor generated were *less*
semantically coherent (0.357) than any real word's 30 nearest neighbours
(0.433). Any claim of the form "there is no word for this region" is
untestable unless your vocabulary *is* the language.

**"Branched inverses discover taxonomic subtrees."** Fan-in proves one-to-many
inverses cannot be functions (`things->color` maps 50 words onto 8), so giving
each inverse K branches is a genuine repair of a type error. But the branches
do **not** find taxonomy. Mutual information between branch and source word is
at or *below* a size-matched shuffled null (0.026 vs null 0.191 for animal
hypernyms). Branches partition the *output* region: a branch is a pointer into
the answer set.

**"Antonymy is a reflection through a learned 8-dimensional polarity
subspace."** False twice.
- *Direction is gauge.* For orthogonal `Q`, `R_{QV}(QWx) = Q R_V(Wx)`, and
  cosine retrieval against a table built by the same adapter is invariant under
  `Q`. So `(W,V)` and `(QW,QV)` are the same model. A **random, never-trained
  V scores 0.300 +/- 0.007 against the trained V's 0.301 +/- 0.014.** If you
  put a free linear adapter in front of a structured operator, the operator's
  parameters may be pure coordinates. Check for this before interpreting them.
- *Dimension is uninformative.* Held-out R@1 is flat for k = 1..512
  (0.267-0.279), decaying only to 0.259 at k=704.

**"The total involution fails because it forces every word to have an
antonym."** No: at `k=768` the map is `-I`, and distilbert's embeddings live in
a narrow cone (word-to-centroid cosine **0.739**; random pair 0.546), so `-x`
lands outside it. The antipode's nearest word in 38k is at cosine **-0.237**,
and **0 of 512** sampled words have any word at positive cosine opposite them.
A geometric fact, dressed up as a linguistic one.

**"The baseline is frozen-identity (0.100)."** The fair baseline for a
structured operator is *a linear adapter trained with the same loss and no
operator*: **0.252**. The reflection's true marginal gain is +0.049, not
+0.183. I inflated my own contribution 7x by choosing the wrong baseline.

**"The mirror is mostly flat (0.557 in-plane)."** Measured in *norm*. In
**energy** it is 0.458, against a random-pair null of 0.099. Over half of every
antonym difference lies off the mirror. Choose the metric before you see which
one flatters you.

**"Fixed-point supervision detects polarity, Cohen's d = 1.155."** Falls to
**0.588-0.77** against a part-of-speech-matched control (antonym-less
*adjectives*, not the whole vocabulary). Word frequency alone reaches AUC
0.643 at detecting "has an antonym" versus the mirror's 0.776. Relation-bearing
words are systematically more frequent (d = +0.634); match on frequency.

**"Antonyms are as close as synonyms (0.801 vs 0.809)."** Those came from
BATS's 50 hand-curated pairs each. At scale it is 0.742 vs 0.693, and runs the
other way. The qualitative claim survived; the evidence for it was a curated
subset.

**"Morphological operators are idempotent (0.936); lexicographic operators are
transitive (0.779)."** Max-vs-min cherry-pick. Category means are 0.918 and
0.844. Worse, 0.779 is an *idempotence* score and nothing measured
transitivity. Worst, for a true involution `cos(f(f(x)),f(x)) = cos(x,f(x))`,
so an idempotence probe **cannot distinguish an involution from a projection**
-- exactly the case it was invoked for.

**"Expanding the antonym set will help."** I expanded WordNet antonymy 21x via
indirect antonymy (adjective satellites reaching opposites through their
heads). Then, with the clean documented 2-hop rule only (8,220 valid extra
pairs, ~40% strict / ~76% lenient precision by inspection), held-out retrieval
**degraded monotonically**: gold only 0.280 -> +1k 0.233 -> +4k 0.192 -> +all
0.185. Gold-only data helps monotonically over the same range (0.058 -> 0.215
for 35 -> 1,476 pairs), so this is not saturation. Indirect antonyms are a
**different function** (`antonym o similar_to`), and one operator trained on
both interpolates between them and lands on neither.

**"Verification is easy, so build a verifier."** A pair-probe separates antonym
pairs from POS- and frequency-matched synonym pairs at AUC **0.824**. I
concluded the information is present and generation is the bottleneck, and
built a retrieve-then-verify reranker. **R@1 fell from 0.277 to 0.080.**
Against the hard negatives retrieval actually produces (the target's own
synonyms and morphological relatives), the probe scores 0.719 while the
generator's own similarity scores 0.807. Easy negatives are not the negatives
you will meet.

---

## 5. Methodological traps, concretely

- **WordNet stores antonym / similar_to / attribute / derivationally_related
  100% symmetrically.** A naive pair-level split leaves **88%** of held-out
  antonym pairs with their reverse in training. Canonicalise
  `tuple(sorted(pair))` *before* splitting. Uncanonicalised, I measured R@1
  0.506; the true value is ~0.27. For expansions that add indirect links, go
  further and use a **word-level** split: test words in no training pair.
- **Never compare R@1 across vocabulary sizes.** Retrieval over 2,768 words is
  not the same task as over 38,142. I nearly reported "99x more data doesn't
  help" from exactly this confound.
- **Multi-positive masks must be built from training pairs only.** Building
  `pos` from all pairs shields held-out targets from being negatives -- test
  information reaching the training loss.
- **Seed noise here is ~0.03 R@1.** Two identical runs gave 0.300 and 0.269.
  Use >= 3 seeds before ranking models. I ranked three architectures on a gap
  smaller than their variance.
- **Relations compose; senses do not.** Every link in `distant ~ far`,
  `far <-> near`, `near ~ walking` is a real WordNet edge, and the composition
  is `distant/walking`. Likewise `petty ~ junior, junior <-> senior` gives
  `petty/senior`. The middle term must be sense-disambiguated. If you chain
  operators through homographs (`turkey`, `hull`, `front`), your chains will
  route through the wrong sense. A decodability gate mostly contains this: in
  the BATS forest, 36% of depth>=2 chains pass through a homograph and
  reconstruct only slightly worse (0.7786 vs 0.7856, d = +0.107, p = 0.046),
  because the chains that broke were already pruned. WordNet's indirect
  antonymy had no such gate, and 74% of it came from an undocumented third hop
  at ~17% precision (`cosy/warm` is tagged an antonym).
- **Permuting the TARGETS is not a null for a translation.** A translation fits
  `d = mean(T) - mean(S)`, and permuting `T` leaves `mean(T)` untouched. The
  target-shuffled null scores *exactly* what the fitted operator scores (0.590
  vs 0.590). It is a no-op, and it makes every mean-offset result look
  unfalsifiable. The valid null permutes the **source at test time**: feed the
  operator a different word and see whether it still lands on the right target
  (0.013 +/- 0.017). Ask what your null actually perturbs.
- **Participation ratio is not comparable across spaces.** I compared effective
  dimension 13.5 (adapted) to 61.4 (frozen) and called it a finding. A learned
  linear map trained to concentrate those very differences shrinks *everything*
  ~1.3x. Only within-space contrasts against a null mean anything.

---

## 6. What worked, methodologically

- **Adversarial subagents with inverted objectives.** Three critics (statistics,
  linguistics, red-team) found the gauge symmetry, the 7x-inflated baseline, the
  norm-vs-energy choice, the POS confound, the frequency confound, the training
  leak, and the 74%-noise expansion. Self-review found none of them, and I had
  already *looked at* several. Diversity of objective did work that capability
  did not.
- **But verify the critics.** Two of the three over-claimed confidently. One
  argued a single reflection cannot handle gradable antonyms (`hot/cold`,
  `warm/cool`); on a scale `cold=-2, cool=-1, warm=+1, hot=+2`, the map
  `x -> -x` does exactly that -- gradables are the paradigm case *for* a mirror.
  One argued converse pairs (`buy/sell`) break the model; held-out retrieval by
  opposition type is noun (converse) **0.375**, adjective 0.250, verb 0.200, and
  `f(buy)->sell`, `f(parent)->child`, `f(husband)->wife`,
  `f(ancestor)->descendant` all hit. The red team called the reflection "inert"
  because a random V matched a trained one -- right observation, wrong
  inference: that is the gauge symmetry.
- **Checkpoint every unit of work to disk.** This machine has a failing DIMM
  (1,015 correctable ECC errors at one address, escalating) and hard-reset
  mid-run twice, silently losing 21 trainings the first time.

---

## 7. The shape of my mistakes

Every one had the same form: I measured the thing that would look like the
result I wanted, and the check that would have caught it was cheap, available,
and never occurred to me *because the result was pleasing*. The gauge symmetry
is two lines of algebra beneath a parameter I trained, named, swept, and
interpreted as a fact about English. The norm-versus-energy choice is one
squaring. The POS-matched control is a filter.

Capability was never the constraint. If you are more capable than I was, you
will generate plausible structure faster, and plausible structure is exactly
what a wrong answer looks like. Before you report a number: name the baseline
someone hostile would demand, name the null that has your effect's shape,
name the metric you would have chosen if the result had come out the other way.
Then run those.

Artifacts in `/home/remy/Code/lexicon`: `RESULTS.md` (what holds and what was
refuted), `ATLAS.md` (29 relations x 6 shapes), `lexicon/` (all code, with
corrections in the docstrings of `mirror.py`, `involution.py`, `algebra.py`).
