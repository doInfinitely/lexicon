# Antonymy in embedding space: what actually holds

Corrected after adversarial review. Where an earlier claim was refuted, the
refutation is stated rather than deleted, because the failure mode is the
interesting part.

Setup: 38,142-word English vocabulary (WordNet lemmas, zipf >= 2.0), contextual
distilbert prototypes mined from wikitext-103 (97.8% of words have >= 3 real
occurrences). 1,736 canonical direct WordNet antonym pairs. All retrieval is
rank-1 against the **full 38k vocabulary**, on a **word-level** held-out split
(held-out words appear in no training pair), with the InfoNCE positive mask
built from **training pairs only**.

## Survives

**1. A translation cannot be an involution.** If `f(x) = x + d` and
`f(f(x)) = x`, then `d = 0`. So antonymy — if it is an involution — provably
cannot be a constant offset, in any embedding space. Confirmed empirically:
fitting a translation to antonym pairs drives `d -> 0`, giving a round-trip
`cos(f(f(x)), x) = 0.987`. It learned "do nothing." Meanwhile `identity`
(0.807) *beats* `translation` (0.798) at predicting antonyms, and a full affine
map buys only +0.006 over identity. Capacity is not the constraint.

**2. The distributional hypothesis places opposites together.** Frozen
distilbert, WordNet scale: antonym pairs cos **0.742** (n=3469), `similar_to`
pairs 0.693, random pairs 0.547. Antonyms are the *closest* relation measured.
In 24/50 BATS binary-antonym pairs the antonym IS the word's nearest
neighbour. Opposites are maximally substitutable in context ("the door is
open/closed"), so any co-occurrence-trained embedding must co-locate them.
This is not a distilbert defect; word2vec and GloVe share it. Sense resolution
does not help: displacement alignment 0.172 (mean point) -> 0.169 (best sense
pair). Antonymy is not a direction at any resolution. Morphology is
(verb->Ving alignment 0.79).

**3. Involutive structure does real work.** Held-out R@1, 3 seeds:

| model | R@1 |
|---|---|
| frozen distilbert, identity operator | 0.100 |
| linear adapter alone, no reflection | 0.252 +/- 0.002 |
| linear adapter + exact reflection (k=8) | **0.301 +/- 0.014** |
| linear adapter + `f(x) = -x` | 0.008 |

The reflection adds +0.049 over a fair baseline. Pure negation fails utterly,
so it is the *involutive structure*, not sign-flipping, that helps.

**4. Fixed-point supervision generalizes.** An exact involution only *permits*
fixed points; it does not induce them. Without supervising them, `f` wanders
over the 35k antonym-less words and ends up nearer the identity on the very
words it trained on (Cohen's d = **-0.364**, backwards) — because antonyms sit
at cos 0.74, so mapping a word to its antonym is a *small* move. Supervising
"words with no antonym satisfy f(x) = x" makes polarity transfer to unseen
words, with correct flips: `passive->active`, `decrease->increase`,
`ventral->dorsal`, `illegitimate->legitimate`, `imprudent->prudent`.

**5. Data, not fine-tuning.** Fine-tuning on 35 pairs manufactures a beautiful
antonymy direction on the training set (alignment 0.85, R@1 1.000) that
evaporates on held-out pairs (0.33, R@1 0.267): it memorizes with 4.3M
parameters. Held-out R@1 vs training-pair count, fixed vocabulary and test set:
35 -> **0.058** (worse than the 0.100 identity baseline: training *wrecks* the
space), 100 -> 0.108, 300 -> 0.146, 1000 -> 0.177, 1476 -> 0.215. Monotone,
unsaturated. Direction alignment stays ~0.14 throughout: more data makes
antonymy *learnable*, never *linear*.

## Refuted

**"V is a learned 8-dimensional polarity subspace of English."** False twice
over.

*Its direction is gauge.* For any orthogonal `Q`, `R_{QV}(QWx) = Q * R_V(Wx)`,
and cosine retrieval against a table built by the same adapter is invariant
under `Q`. So `(W, V)` and `(QW, QV)` are the *same model*: with a free linear
adapter, `V` is unidentifiable. Confirmed: a random, never-trained `V` scores
0.300 +/- 0.007 against the trained `V`'s 0.301 +/- 0.014. The gauge-invariant
object is `span(W^T V)` in raw space; independent runs recover it only
partially (min principal angle 14 deg, mean 49; a random subspace gives 84.5).

*Its dimension is also uninformative.* Gold-only, 3 seeds, held-out R@1 by `k`:
1 -> 0.268, 2 -> 0.267, 4 -> 0.279, 8 -> 0.277, 16 -> 0.268, 32 -> 0.277,
64 -> 0.278, 128 -> 0.279, 256 -> 0.279, 384 -> 0.273, 512 -> 0.272,
640 -> 0.263, 704 -> 0.259, **768 -> 0.000**. Flat across three orders of
magnitude, then a cliff at exactly `k = 768`. An earlier reading of
`k=1 (0.243) >> k=8 (0.185)` came from the *corrupted* gold+indirect training
set. So "English antonymy needs ~1-8 polarity dimensions" is retired: the
measurement cannot distinguish 1 from 512.

*And the cliff is geometric, not linguistic.* At `k = 768` the reflection is
`f = -I`. distilbert embeddings occupy a narrow cone -- a word sits at cosine
**0.739** from the corpus centroid, two random words at **0.546** -- so `-x`
lands outside the cone entirely. Its nearest word in the whole 38k vocabulary
is at cosine **-0.237**, and **0 of 512** sampled words have any word at
positive cosine opposite them. `f = -I` maps every word into empty space. The
earlier explanation ("a total involution forces every word to have an
antonym") was a linguistic story told about an anisotropy artifact. The single
requirement on the reflection is that it fix a *nonzero* subspace, which is why
every `k < 768` performs identically.

**"The mirror is mostly flat" (0.557 in-plane).** Measured in *norm*, which
flatters it. In **energy**: antonym differences are 0.458 in-plane (random-pair
null 0.099; true chance 8/768 = 0.010). **Over half** of every antonym
difference lies off the mirror.

**"Polarity detection, Cohen's d = 1.155."** Falls to **0.588-0.77** against a
part-of-speech-matched control (antonym-less *adjectives*, not the whole
vocabulary). Word frequency alone reaches AUC 0.643 at detecting "has an
antonym" versus the mirror's 0.776, and `corr(cos(f(x),x), zipf) = -0.249`.
Real, but roughly 40% frequency artifact — and a random plane reproduces
d ~ 1.0, so it evidences no learned geometry.

**"Antonyms are as close as synonyms (0.801 vs 0.809)."** Those numbers are
from BATS's 50 hand-curated pairs each. At scale the gap is 0.742 vs 0.693 and
runs the other way. The qualitative claim survives; the evidence cited for it
was a curated subset.

**"Morphological operators are idempotent (0.936); lexicographic operators are
transitive (0.779)."** Max-vs-min cherry-pick. Category means: inflectional
0.918, lexicographic 0.844. And 0.779 is an *idempotence* score — nothing in
`algebra.py` measures transitivity. Worse, for a true involution
`cos(f(f(x)), f(x)) = cos(x, f(x))`, so the idempotence probe cannot
distinguish an involution from a projection precisely where antonymy is
concerned.

**"Indirect antonymy is antonym o similar_to, and polarity invariance recovers
it."** The decomposition is right; the remedy fails. 74% of the 35k "indirect"
pairs come from an undocumented *third* hop into the antonym's own satellites;
sampled precision is ~17% strict (`cosy/warm` is tagged an antonym). Training
on them *degrades* direct antonym retrieval, 0.263 -> 0.200. And imposing
`p(a) = p(b)` on synonyms does not make indirect antonymy emerge: held-out
indirect R@1 = **0.004**.

**"Exploding the antonym set will help."** It does not, even when the added
pairs are valid. Restricting to the documented two-hop rule plus POS-checked
morphological negation gives 8,220 extra pairs (7.2x), at ~40% strict / ~76%
lenient precision by inspection — 2.3x cleaner than the third hop. Held-out
R@1 on gold direct antonyms, word-level split, 2 seeds:

| train pairs | source | R@1 |
|---|---|---|
| 1,331 | gold only | **0.280 +/- 0.003** |
| 2,331 | gold + 1k clean | 0.233 +/- 0.003 |
| 5,331 | gold + 4k clean | 0.192 +/- 0.005 |
| 9,551 | gold + all clean (7.2x) | **0.185 +/- 0.008** |

Monotone *degradation*. Since gold-only data helps monotonically over the same
range (0.058 -> 0.215 from 35 -> 1,476 pairs), this is not saturation: indirect
antonyms are a **different function**, and training one operator on both makes
it interpolate between `antonym` and `antonym o similar_to`, landing on
neither. The earlier claim that the data curve is "unsaturated" holds only for
*direct* antonymy, where WordNet gives 1,736 pairs and no more.

**Why composition fails: the middle term must be sense-disambiguated.** Every
link in `distant ~ far`, `far <-> near`, `near ~ walking` is a real WordNet
edge, and the composition yields `distant/walking`. Likewise `petty ~ junior`,
`junior <-> senior` gives `petty/senior`, and `sleazy ~ inferior` gives
`sleazy/superior`. The satellite attaches to its head under one sense while the
head's antonym opposes another. Relations compose; senses do not. This is a
general constraint on chained derivations -- including the depth-3 operator
chains in the earlier lexicon work, which happily routed through homographs
(`turkey`, `hull`) that the sense-clustering had already flagged.

## The set of words: English is not compressible by these operators

The original premise -- a small base lexicon plus operators reconstructs the
rest -- was reported at 3.25x compression. That number was measured inside a
*learned adapter's space*, trained until the derivations decoded, and scored on
pairs the model had trained on. Refit in the FROZEN space, with the best shape
per relation chosen by held-out retrieval, a word counts as derived only if
some `f_r(a)` retrieves it first out of all 38,142 words with a margin:

| operator (held-out edges only) | edges decoded | words derived | compression |
|---|---|---|---|
| **fitted operator** | **2,802** | **2,441** | **1.068x** |
| identity (null) | 1,402 | 1,182 | 1.032x |
| shuffled relation (null) | 1,348 | 1,254 | 1.034x |

The operators are real -- twice the identity null on edges whose target word
was never a training target -- and nearly useless for compression.

Context. 33,180 of 38,142 words (87%) are the target of *some* WordNet
relation, so a perfect operator set would compress **7.687x**. The operators
realise **7.4%** of that. **77% of English is irreducible**, and 4,765 words
(12.5%) participate in no relation at all -- irreducible by definition, not by
failure of the operators.

Scored over *all* edges rather than held-out ones, the same pipeline reports
1.299x, of which 7,076 of 8,781 derived words are simply training targets the
closed-form fit memorised. The gap between 1.299x and 1.068x is memorisation;
the gap between 3.25x and 1.299x is the adapter.

Relations that actually derive words, in order: `derivationally_related`,
`verb_Ving`, `verb_Ved`, `adj_ly`, `antonym`, `adj_comparative`, `verb_er`,
`noun_plural`. Note that `derivationally_related` decodes 2,448 edges where
the *identity operator* alone decodes 1,301: more than half its apparent
derivations are "the target was already the nearest neighbour."

### Derivability requires being the target of a near-BIJECTIVE relation

Fan-in (pairs per distinct target) decides whether a relation can *determine*
its target. `hypernym` is 7.14 and `instance_hypernym` is 10.07 -- they reach
`entity` and `city` from everything and pin down nothing. The near-injective
relations are morphology (1.02-1.08) and antonymy (1.10).

| relations allowed | words determined | ceiling |
|---|---|---|
| fan-in <= 1.2 | 11,037 (28.9%) | **1.407x** |
| fan-in <= 1.5 | 19,315 (50.6%) | 2.03x |
| any relation | 33,205 (87.1%) | 7.73x |

So the 7.687x "ceiling" quoted above counts words no relation can select.

### Cooking the analogies WordNet lacks

Two fixes. (1) `instance_hypernym()` is a separate NLTK method from
`hypernyms()`; omitting it made 1,628 words (34% of the apparent orphans) look
unrelated (`bernini -> sculptor`, `truman -> president`). Fixed. It has fan-in
10.07, so it fixes the census, not the compression. (2) WordNet's
`derivationally_related` is an undirected soup (fan-in 1.21) with no process
label. Recovering the processes as 46 separate directed near-bijective
relations (`happy -> happiness`, `nation -> national`, `dark -> darken`), plus
6 bijective fact tables, gives 9,964 cooked pairs -- both forms required to be
attested in the vocabulary, nothing invented.

**28 of 40 cooked relations generalise** (lift > 0.10 over the better of the
identity and source-permuted nulls), and 26 of the 28 are morphology. Only
`country_capital` (+0.560) and `male_female` (+0.556) survive among the facts.
`country_demonym` scores R@1 **1.000 and identity scores 0.923** -- `france`
and `french` are already nearest neighbours, so the operator does nothing.

Result, same held-out-edges protocol:

| | words derived | compression | identity null | ceiling |
|---|---|---|---|---|
| WordNet only | 2,441 | 1.068x | 1,182 | 1.407x |
| **+ fix + cooked** | **3,193** | **1.091x** | 1,497 | **1.617x** |

Cooking adds 752 derived words (+31%); 28% of all derivations now come from a
cooked relation. But the fraction of the *determinable* words captured barely
moves: 22.1% -> 21.9%. Subsampling the relation inventory 4x shows this is not
a constant but a narrow band -- 17.2% (18 relations) -> 17.2% (36) -> 20.5%
(54) -> 22.3% (73). **Adding relations raises the ceiling and the result
together. The binding constraint is the operator, not the relation inventory.**

## Is the SPACE the bottleneck, or the operator? (the space is not)

distilbert's embeddings occupy a narrow cone (cos 0.739 to the centroid, 0.546
between random words). Anisotropy and hubness are known pathologies of
retrieval-through-a-linear-map, with known fixes. Mean held-out R@1 over 12
relations, best closed-form shape per cell:

| space + retrieval rule | mean R@1 | vs baseline |
|---|---|---|
| raw + cosine (used throughout this project) | 0.426 | -- |
| raw + CSLS | 0.425 | -0.001 |
| centered + cosine | 0.447 | +0.021 |
| **all-but-the-top (drop 8 PCs) + cosine** | **0.462** | **+0.036** |
| whitened (ZCA) + cosine | 0.311 | **-0.115** |

Removing the top principal directions is a real but small free win (+8%
relative); adopt it. Full whitening is catastrophic -- it amplifies
low-variance directions that carry no signal. CSLS does nothing because
hubness is mild here (the top 10 hubs take only 1% of nearest-neighbour
slots). **The verification/generation gap is not a retrieval artifact.**

## Localisation vs selection: what generation actually fails at

Recall@k in the abtt space decomposes the failure. `R@1/R@50` near 1 means the
operator picks correctly once it finds the region; low means it localises but
cannot select.

| relation | fan-in | R@1 | R@50 | R@1/R@50 |
|---|---|---|---|---|
| `deriv:adj_ly` | 1.00 | 0.759 | 0.919 | 0.83 |
| `cook:suf_ness` | 1.00 | 0.783 | 0.967 | 0.81 |
| `infl:noun_plural` | 1.00 | 0.620 | 0.913 | 0.68 |
| `lex:antonym` | 1.10 | 0.238 | 0.669 | **0.36** |
| `lex:member_meronym` | 1.31 | 0.190 | 0.499 | 0.38 |
| `lex:hypernym` | 7.14 | 0.044 | 0.297 | **0.15** |

Morphology is **localisation-limited**; semantics is **selection-limited**. And
`spearman(fan-in, R@1/R@50) = -0.891, p = 1e-4`: a one-to-many relation cannot
be picked at rank 1, because the neighbourhood is full of correct answers.
Strict R@1 punishes it for being right.

The exception is diagnostic: **`antonym` has fan-in 1.10 and ratio 0.36.** It
is near-bijective, so alternative correct answers cannot explain it. Its
neighbourhood is full of near-*synonyms* of the answer (`chilly` for `cold`) --
the co-location result, arriving from a third direction.

## The lemma vocabulary was the wrong object

Every compression number above was measured on WordNet lemmas -- and a lemma
list is what a lemmatiser leaves behind. `cats`, `dogs`, `books`, `children`,
`women` (Zipf 4.5-5.6) are **absent from it**. `infl:noun_plural` had 639 pairs
against 38,142 words. I measured "is English compressible" on a vocabulary from
which the compressible forms had already been deleted.

On a surface-form vocabulary (51,148 words, Zipf >= 2.5): `noun_plural` has
**8,382** pairs, `verb_3pSg` 8,341, `verb_Ved` 3,350, `verb_Ven` 3,320,
`verb_Ving` 3,106. Words determined by a fan-in <= 1.2 relation: 28.9% ->
**45.4%**. Ceiling: 1.617x -> **1.832x**.

This invalidates the compression claims (`1.068x`, `1.091x`, "77% of English is
irreducible") and nothing else. The atlas measures per-relation shape and
retrieval on pairs present in both vocabularies; adding `cats` changes how many
words `plural` can derive, not what shape `plural` is.

## Irregularity does not exist in embedding space

"Irregular" is a property of the STRING: `went` is not `go`+ed. An operator
never sees a string -- it sees a contextual vector built from sentences where
`went` means past-of-go. Fit each inflection operator on **regular pairs only**,
then test on irregulars it has never seen:

| relation | R@1 regular | R@1 irregular | identity null | source-permuted |
|---|---|---|---|---|
| `noun_plural` | 0.849 | 0.605 | 0.452 | 0.008 |
| `verb_Ved` | 0.719 | 0.614 | 0.089 | 0.003 |
| `verb_Ven` | 0.717 | 0.441 | 0.082 | 0.005 |

`be -> was` and `arise -> arose` are retrieved by an operator that never met an
irregular and cannot read a spelling. Three explanations for the residual gap
were tested and **all three failed**:

1. *Subword overlap.* `walked` contains the wordpiece `walk`, so maybe the
   operator exploits orthography. No: irregular pairs sharing **no** subword
   (`be/was`, `arise/arose`) score **0.632** on `verb_Ved`, HIGHER than
   irregulars that do share one (0.429), against an identity null of 0.083.
2. *Frequency.* No, and backwards: the no-subword irregulars are **more**
   frequent than the regulars (Zipf 4.16 vs 3.39), 100% corpus-attested.
3. *Orthographic baggage in the regular-trained operator.* No: fitting on
   irregulars alone does not beat fitting on regulars (0.629 vs 0.645), and
   fitting on **both** is best (0.677).

Fitting on both being best is the tell: regular and irregular pairs lie on the
**same displacement manifold**. There is one past-tense operator. `walk->walked`
and `be->was` are the same geometric move. The regular/irregular split is an
orthographic premise with no geometric correlate, and partitioning results by
it -- as an earlier version of `paradigm.py` did -- imports spelling into a
question about meaning.

## Is a paradigm one object?

Predict a held-out inflected form of a held-out paradigm. `z` is inferred from
the observed forms only and never sees the target.

| scheme | all paradigms |
|---|---|
| base (lemma embedding alone) | 0.503 |
| lexeme(1) | 0.280 |
| centroid(2) | 0.703 |
| lexeme(2) | 0.614 |
| centroid(3) | 0.746 |
| **lexeme(3)** | **0.771** |

Observing three forms predicts a fourth at 0.771 where the lemma alone gives
0.503. **The other forms carry information about the target that the lemma does
not**: `ran` is not a function of `run` alone, so a paradigm is more than a bag
of pairwise maps from its base. But the learned lexeme barely beats the
**centroid of the observed forms** (0.771 vs 0.746) and loses to it at k=1 and
k=2. The virtual word is, once again, an average.

## Where the critics were wrong

Adversarial review caught all of the above. It also over-claimed twice, so
critic findings were verified rather than accepted.

- A single reflection *can* handle gradable antonyms. On a scale
  `cold=-2, cool=-1, warm=+1, hot=+2`, the map `x -> -x` sends `hot->cold` and
  `warm->cool`. Gradables are the paradigm case *for* a mirror.
- Converse pairs (`buy/sell`) do not break it. Held-out R@1 by opposition type:
  noun (converse) **0.375**, adjective 0.250, verb 0.200. `f(buy)->sell`,
  `f(parent)->child`, `f(husband)->wife`, `f(ancestor)->descendant` all hit.
- The reflection is not "inert" because a random `V` matches a trained one.
  That is the gauge symmetry. Adapter+reflection (0.301) still beats
  adapter-alone (0.252).

## Known bugs, fixed

- `involution.load_antonyms` built the InfoNCE positive mask from *all* pairs,
  shielding 520 held-out antonyms from repulsion during training. Now built
  from training pairs only; `pos_eval` carries the full set to evaluation.
- WordNet stores antonym / similar_to / attribute / derivationally_related
  100% symmetrically. A naive pair-level split leaves 88% of held-out antonym
  pairs with their reverse in training. Canonicalize `tuple(sorted(pair))`
  *before* splitting. An uncanonicalized split reported R@1 0.506; the true
  value is ~0.27.
- Never compare R@1 across vocabulary sizes: retrieval over 2,768 words is not
  the same task as over 38,142.
- Single-run differences of ~0.03 here are seed noise. Use >= 3 seeds.
