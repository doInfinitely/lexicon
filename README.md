# lexicon

Does telling a language model about English morphology help it?

Short answer: **inflection yes, derivation no** — and most of the received wisdom about
why turns out to be wrong, including several things this repo previously asserted.

Everything is scored in **bits per character of the original text**, the only measure
comparable across tokenizers. Baseline is GPT-2 BPE. Trunk is a 10.65M-parameter GPT
(L6 d384); embedding parameters are excluded from "model size" because they are a lookup
table and cost no FLOPs.

## What holds

**Word-level tokenization with a byte fallback beats BPE, and is shorter.**
`free` (one token per word, 16k vocab, rare words spelled in UTF-8 bytes) scores 2.025
against BPE's 2.143 on 10k paragraphs of wikitext-103 — at **0.926x the tokens**.

**Morphological factorization of the embedding table helps on top of that.**
Instead of giving `walked` its own vector, define it:

    e(word) = E_root + d_slot + U_slot (V_slot^T E_root)     # rank 32
    slot 0 (identity): d_0 = 0, U_0 = V_0 = 0, pinned        # gauge fix

`walked` has no parameter of its own. It cannot drift from `walk`. This precomputes into
an ordinary [V, d] table, so it costs **zero inference FLOPs** — it is a structured prior
on the embedding matrix, not a runtime component.

| 10k paragraphs | bpe | free | translate | affine | shufroot |
|---|---|---|---|---|---|
| bits/char | 2.1434 | 2.0248 | 2.0357 | **2.0140** | 2.0211 |

- `affine − free      = −0.0108 ± 0.0038`  enforced sharing helps
- `affine − shufroot  = −0.0071 ± 0.0045`  and it is **morphology**, not low-rank
  regularization: pointing `walked` at a *random* root, with identical capacity and
  identical rank constraint, is measurably worse.

**Seven inflectional operators do all the work.** Plural, gerund, 3sg, participle, past,
comparative, superlative.

**Compute-equivalence: 2.0x**, measured with the baseline slope and the gain under the same
protocol (byte fallback, warmup + constant LR, each arm's own best held-out checkpoint):

| trunk | params | bpe | lex-v6 | delta |
|---|---|---|---|---|
| L6 | 10.65M | 1.5564 | 1.5009 | −0.0555 |
| L8 | 25.22M | 1.4893 | 1.4329 | −0.0563 |

bpe's own curve gives 0.0540 bits/char per doubling of trunk params, so a 0.0555 gain is
worth 2.04x the parameters, bought for 1.021x the tokens. **The advantage is flat across
trunk size** (L6 and L8 agree to 0.0008) — it does not grow with scale, contrary to what
earlier, protocol-confounded runs suggested.

## What does not hold

**Derivation contributes nothing.** Not as operator tokens, not as affine maps, not with
rotations allowed, not with zero sequence-length cost, not composed four deep, not at any
corpus size from 1.4M to 22M words. `deriv-aff − infl = −0.0014 ± 0.0027`.
Three independent experimental conditions. The 27 derivational operators —
`noun.agent`, `suf.ly`, `adj.able` — buy zero bits/char.

**`Ax` is not enough; `Ax + d` is required.** A linear map fixes the origin and cannot
translate. But equally: **`x + d` alone is not enough either.** Pure translation *loses*
to free embeddings (+0.008…+0.011). Both terms are necessary.

**A good map of BERT's embedding space is not a good parameterization for a causal LM.**
Seen three times: freezing BERT prototypes as the input embedding costs 0.8 bits/char;
initializing from them scores *identically to a shuffled permutation*; and the
lemma-plus-displacement form that reconstructs BERT's inflectional geometry at rank 1
loses to letting the word be free. The relation atlas (`ATLAS.md`) describes the geometry
of a masked LM's representations. A next-token predictor wants something else.

## Caveats, stated because they are load-bearing

- **The advantage depends on the learning-rate schedule.** `lex-v6 − bpe` is −0.131 under
  OneCycleLR-scored-at-the-end and −0.075 under warmup+constant-LR-best-checkpoint, same
  tokenizer, same data, same steps. Any conversion of bits into "equivalent parameter
  count" therefore reports the optimizer as much as the tokenizer. The 2.0x figure above is
  measured with both sides under one protocol. **The `affine` numbers below are still under
  the older schedule and are not directly comparable to it** (rerun in flight).
- **The affine gain decomposes into two opposing effects.** Since
  `affine − free = (affine − shufroot) + (shufroot − free)`, and shufroot is the same
  rank-constrained table pointing at random roots:

  | | morphology | low-rank regularization | total |
  |---|---|---|---|
  | 10k | −0.0071 | −0.0037 | −0.0108 |
  | 40k | −0.0187 | +0.0077 | −0.0109 |

  Identical totals for opposite reasons. Regularization helps when data is scarce and hurts
  when it is not; morphology's contribution *grows* with data, which is the reverse of the
  sample-efficiency story. Sharing only pays if the shared parameter is well estimated: at
  10k, `E_walk` is noisy and forcing `walked` to inherit from it propagates the noise.
- Data-efficiency (advantage grows as the corpus shrinks: −0.075 → −0.082 → −0.132 over a
  16x range) is measured at only one rung where early stopping actually triggered. It may
  be **overfitting resistance** rather than sample efficiency. These are different
  mechanisms with the same signature.
- 2–3 seeds throughout. Seed sd on BPE bits/char is ~0.006.

## Errors this repo has made, preserved because they generalize

- **The `<unk>` cheat.** `BPETok` mapped out-of-vocab tokens to a single frequent `<unk>`
  while `bits_per_char` credited every character of the swallowed word. BPE `<unk>`ed 5.70%
  of tokens, our tokenizer 2.87%. Two consequences: the "20% sequence-length penalty" was
  mostly BPE's cheat (token ratio 1.199x → **1.020x** with a byte fallback), and
  "derivation actively hurts" was an artifact — the more complete dictionary escaped less
  often, so it cheated less, so it scored worse.
- **Gauge freedom, four times.** A free linear map in front of a structured operator makes
  the operator's parameters unidentifiable: `R_{QV}(QWx) = Q · R_V(Wx)`. A randomly
  initialized, untrained reflection plane once scored 0.300 against a trained 0.301. Pin
  the identity slot.
- **Confounded arms.** Changing the dictionary, the escape brackets, and the operator
  order in one run, three times, and then attributing the delta to whichever one had a
  story attached.
- **Nulls that were no-ops.** Shuffling the targets of a fitted translation is
  permutation-invariant (`d = mean(T) − mean(S)`), so the "null" scored exactly what the
  fit scored: 0.590 vs 0.590.
- **Strawman baselines.** Comparing prototype-initialized embeddings (std 0.02) against
  PyTorch's default `N(0,1)` — and then, after fixing that, leaving *positional* embeddings
  at `N(0,1)` so word identity was 50x quieter than position.

## Layout

    lexicon/          all experiments (wt_*.py are the wikitext language-model studies)
    dictionary/       forest_v*.json — the morphological dictionaries (v6 = inflection only)
    real/*.log        raw experiment logs; real/*.json, real/ts/*.json — results
    ATLAS.md          the English Relation Shape Atlas (29 relations x 6 operator shapes)
    RESULTS.md        earlier structural findings and what survived their controls
    LANGUAGE.md       the operator-token language spec (largely superseded)
    harbor/           the original Harbor task solution

Dictionaries are built from MorphyNet (Wiktionary-derived morphology) plus lemminflect for
inflection. Hand-written affix rules were previously used and produced `number = numb + er`,
`station = state + ion`, `business = busy + ness`. Do not use hand-written affix rules.
