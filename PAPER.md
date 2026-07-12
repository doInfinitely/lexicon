# Morphological embedding factorization helps language models in proportion to a language's morphology

**Draft — internal. Numbers are single-machine, treebank/subset scale, 2–3 seeds. Read the Limitations section before citing any figure.**

## Abstract

We test whether giving a language model an explicit morphological prior — factoring a
word's input embedding into a lemma embedding plus an operator for its inflection —
improves next-token prediction, measured in bits per character of the original text
(the only tokenizer-independent measure). In English the effect is negligible
(−0.011 bits/char). Across five morphologically richer languages it is 20–40× larger
(−0.19 to −0.42 bits/char), scaling with how much of the language is inflected. However,
a shuffled-root control shows the gain is largely due to the *low-rank pooling structure*
of tying rare word-forms to shared roots, not to the *correctness* of the morphological
grouping — the morphology-specific component is clearly present only in Turkish. We also
find that derivation (meaning-changing morphology) contributes nothing for a language
model in any functional form we tried, and that the "operator as a token" framing is
strictly worse than "operator as a pre-model affine map," because a token costs sequence
length and can only translate, while meaning-change needs rotation. The paper doubles as
a catalogue of measurement traps: nearly every strong effect we found was, on first
measurement, a confound in the direction of our hopes.

## 1. Setup

**Metric.** bits/char over the original characters, counting only characters inside the
scored window. Cross-entropy is *not* comparable across tokenizers (a lexeme stream
contains low-entropy operator tokens), so we never compare it.

**Baseline.** GPT-2 byte-level BPE, 16k vocab, with a byte fallback so no token is ever
`<unk>` (see §5.1 — the `<unk>` handling was our largest single measurement error).

**Trunk.** A small GPT (L6 d384, 10.65M non-embedding params) unless noted. Embedding
parameters are excluded from "model size": they are a lookup table, zero FLOPs.

**The factorization.** Instead of a free embedding per word, a word is

    e(word) = E_root + d_slot + U_slot (V_slot^T E_root)          (rank 32)

`E_root` is the lemma embedding; `d_slot` a per-inflection translation; `U V^T` a low-rank
correction (rotation). The identity slot is pinned (`d_0 = 0, U_0 = V_0 = 0`) to remove a
gauge freedom that otherwise makes the operator parameters unidentifiable. This precomputes
into an ordinary `[V, d]` table, so it costs **zero inference FLOPs** — it is a structured
prior on the embedding matrix, not a runtime component.

## 2. English: the prior barely helps, and derivation not at all

On wikitext-103, inflection-only factoring (7 operators: plural, gerund, 3sg, participle,
past, comparative, superlative) beats BPE, and the advantage *grows as data shrinks*:

| paragraphs | bpe | inflection-factored | Δ |
|---|---|---|---|
| 10,000 | 1.956 | 1.824 | −0.132 |
| 40,000 | 1.771 | 1.689 | −0.082 |
| 160,000 | 1.740 | 1.665 | −0.075 |

Protocol: byte fallback both sides, each arm trained to its own best held-out checkpoint.
At the small rung both arms genuinely converge (stop <5k steps), so the gain is a fair
best-vs-best comparison, not a truncation artifact — though at 10k it is partly *overfitting
resistance*, a legitimate sample-efficiency benefit, not pure representation.

**Derivation contributes nothing.** Adding derivational operators (`worker←work`,
`nation→national`, …) — as tokens or as affine maps, with rotation allowed, at zero
sequence-length cost, composed up to depth 4 — never beats inflection-only
(Δ = −0.001 ± 0.003). Seven inflectional operators do all the work; the 27 derivational
operators and the entire 29-relation "atlas" buy zero bits/char for a language model.
(The atlas's *geometric* claims about embedding space — orthogonal maps beat translation on
lexicographic relations, cosine is the wrong retrieval metric — survive their controls;
what dies is the inference from "English has this structure" to "an LM benefits from it.")

**Operator as matrix, not token.** A pre-model affine map beats an operator *token* and
beats a free embedding, and it survives its shuffled-root null. A pure translation
(`E_root + d_slot`, no rotation) *loses* to a free embedding; the rotation term is
necessary. This contradicts the atlas's "inflection is a translation" prediction *inside a
language model*: the bias is necessary but not sufficient.

## 3. Cross-linguistic: the prior scales with morphology

Universal Dependencies treebanks (gold lemma + features = root + operator, no analyzer or
sandhi confound), 3 seeds. `affine − free` is the factorization gain; `affine − shufroot`
(random-root pointing, matched capacity) isolates the morphology-specific component.

| language | % content-word inflected | affine − free | affine − shufroot |
|---|---|---|---|
| English | 30% | −0.011 | −0.007 |
| Spanish | 40% | −0.189 | −0.017 |
| German | 41% | −0.216 | +0.012 |
| Turkish | 66% | −0.359 | −0.044 |
| Russian | 71% | −0.419 | −0.016 |
| Finnish | 80% | −0.229 | −0.005 |

**Finding 1 (robust).** The factorization gain is 20–40× larger in every morphologically
rich language than in English. The English-vs-rich contrast is unambiguous; the ordering
is roughly monotonic in inflection rate (Finnish is a low outlier). English is close to the
*worst* language to demonstrate this on — a fact obscured for us for weeks by only running
English.

**Finding 2 (the honest caveat).** The morphology-*specific* component (`affine − shufroot`)
is clearly nonzero only in Turkish (−0.044). In German it is even slightly positive
(shuffled beats real). So most of the gain is the *low-rank pooling structure* of tying
sparse word-forms to shared roots — which helps regardless of whether the grouping is the
*correct* morphology. We therefore frame the result as **structured sparse embedding
factorization helps morphologically rich languages**, not "morphology is the mechanism."

**Finding 3.** Word-level tokenization *alone* (before factoring) loses to BPE in the
sparse agglutinative languages (Turkish +0.155, Russian +0.405, Finnish +0.218) because
word-forms are too rare; the factorization is what makes word-level viable there. In less
sparse languages (Spanish, German) word-level already wins.

## 4. What actually drives the English gain, ranked

1. **Decomposition volume** (0.073 bits/char): decomposing many *rare* words costs tokens
   and buys nothing — rare words carry no probability mass.
2. **Decomposition frequency profile** (0.031): spending the token budget on *frequent*
   decomposable forms, not rare ones, is where the win is.
3. **False decompositions** (0.009): `station←state`, `number←numb` from hand-written affix
   rules. Real but an order of magnitude smaller than (1)–(2).

## 5. Measurement traps (the real contribution may be this section)

Nearly every strong effect was, on first measurement, an artifact favoring our hypothesis.

**5.1 The `<unk>` cheat.** BPE mapped out-of-vocab pieces to a single frequent `<unk>`
token while bits/char credited the model with the swallowed word's characters. BPE `<unk>`ed
5.7% of tokens; the lexeme stream 2.9%. Two consequences: the "20% sequence-length penalty"
of word-level tokenization was mostly BPE's cheat (real ratio 1.02×, not 1.20×), and
"derivation actively hurts" was an artifact — the more complete dictionary escaped to
`<unk>` less, so it cheated less, so it *scored* worse. A byte fallback fixed both.

**5.2 Gauge freedom (four times).** A free linear map in front of a structured operator
makes the operator's parameters unidentifiable. A randomly-initialized untrained reflection
once scored identically to a trained one. Pin the identity slot.

**5.3 Nulls that were no-ops.** Shuffling the *targets* of a fitted translation is
permutation-invariant, so the "null" scored exactly what the fit scored. The valid null
permutes the source, or the root assignment.

**5.4 Confounded arms.** Changing dictionary + escape brackets + operator order in one run,
then attributing the delta to whichever had a story attached. Minimal one-variable contrasts
should come first.

**5.5 Strawman baselines.** Comparing a prototype-initialized embedding (std 0.02) against
PyTorch's default `N(0,1)`; then, after fixing that, leaving *positional* embeddings at
`N(0,1)` so word identity was 50× quieter than position.

**5.6 Protocol dependence.** The lexeme advantage moves with the learning-rate schedule
(−0.131 vs −0.075 on identical configs). Any bits→params compute-equivalence conversion must
measure the baseline slope and the gain under one protocol.

## 6. A related negative result: BERT geometry ≠ LM parameterization

Initializing the LM's embedding table from BERT contextual prototypes performed
*identically to a shuffled permutation* of those prototypes; freezing them cost 0.8
bits/char; the lemma-plus-displacement form that reconstructs BERT's inflectional geometry
at rank 1 lost to a free embedding. A good map of a masked LM's representation space is not
a good input parameterization for a causal LM. They are different objects.

## 7. Limitations

- Single machine; 2–3 seeds; treebank-scale corpora (17k–518k tokens) for the
  cross-linguistic result, 200k-paragraph wikitext subset for English.
- The cross-linguistic ordering is not perfectly monotonic (Finnish).
- Korean (no FEATS), Arabic (lemmatization makes it 100% "inflected"), and Czech (parse
  failure) were dropped; the language sample is Indo-European-heavy plus Turkish/Finnish.
- No comparison to a modern large-vocab BPE at scale, or to subword-regularization
  baselines; the trunk is tiny by contemporary standards.
- "Compute-equivalence" figures depend on the LR-schedule caveat (§5.6) and are reported
  only where slope and gain share a protocol.

## 8. Takeaway

For a causal language model: **lemmatize and tag inflection.** It is close to zero-cost
(1.02× tokens with a byte fallback, zero inference FLOPs), it helps most exactly where data
is scarce and morphology is rich — the low-resource morphologically-complex setting — and
it is a structured-pooling effect more than a linguistic-correctness effect. Derivation, and
the elaborate operator "language" we started with, are not worth the complexity for an LM,
though they remain valid descriptions of the geometry of English word embeddings.
