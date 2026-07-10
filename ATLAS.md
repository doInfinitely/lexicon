# The English Relation Shape Atlas

29 of 32 WordNet relations (3 too small after word-level splitting).
Frozen distilbert, no adapter. Word-level holdout: test words appear in no
training pair. Symmetric relations canonicalised before splitting. R@1 ranks
against all 38,142 words, source excluded; `any` credits every sanctioned
target. `detect` = pair-probe AUC vs POS- and frequency-matched random pairs.

| relation | n | fan-in | direction | eff dim | detect AUC | best linear | best R@1_any |
|---|---|---|---|---|---|---|---|
| `deriv:adj_ness` | 305 | 1.0 | 0.52 | 14 | 1.00 | reflection | 0.755 |
| `deriv:adj_ly` | 1185 | 1.0 | 0.42 | 23 | 0.99 | orthogonal | 0.741 |
| `deriv:noun_less` | 153 | 1.0 | 0.48 | 18 | 0.99 | translation | 0.517 |
| `deriv:verb_tion` | 445 | 1.02 | 0.38 | 31 | 0.98 | translation | 0.420 |
| `deriv:verb_ment` | 195 | 1.0 | 0.40 | 21 | 0.96 | translation | 0.386 |
| `deriv:verb_er` | 1127 | 1.05 | 0.28 | 55 | 0.97 | orthogonal | 0.375 |
| `deriv:verb_able` | 206 | 1.07 | 0.38 | 23 | 0.96 | translation | 0.287 |
| `deriv:un_adj` | 598 | 1.0 | 0.41 | 20 | 0.98 | translation | 0.202 |
| `deriv:over_adj` | 113 | 1.0 | 0.39 | 23 | 0.96 | reflection | 0.196 |
| `deriv:re_verb` | 413 | 1.0 | 0.26 | 48 | 0.92 | orthogonal | 0.077 |
| `infl:verb_3pSg` | 634 | 1.0 | 0.34 | 50 | 1.00 | reflection | 0.672 |
| `infl:noun_plural` | 639 | 1.0 | 0.32 | 48 | 1.00 | orthogonal | 0.651 |
| `infl:verb_Ving` | 1907 | 1.02 | 0.38 | 35 | 0.99 | orthogonal | 0.632 |
| `infl:verb_Ved` | 1667 | 1.02 | 0.39 | 33 | 0.99 | orthogonal | 0.619 |
| `infl:adj_comparative` | 1253 | 1.02 | 0.28 | 53 | 0.97 | orthogonal | 0.387 |
| `lex:derivationally_related` | 7435 | 1.04 | 0.14 | 76 | 0.98 | orthogonal | 0.667 |
| `lex:member_meronym` | 1111 | 1.26 | 0.17 | 68 | 0.98 | reflection | 0.614 |
| `lex:part_meronym` | 6248 | 1.62 | 0.14 | 113 | 0.93 | orthogonal | 0.557 |
| `lex:part_holonym` | 6248 | 2.78 | 0.14 | 113 | 0.95 | orthogonal | 0.498 |
| `lex:member_holonym` | 1111 | 1.79 | 0.17 | 68 | 0.98 | orthogonal | 0.496 |
| `lex:attribute` | 725 | 1.3 | 0.08 | 41 | 0.99 | reflection | 0.417 |
| `lex:substance_holonym` | 340 | 1.31 | 0.17 | 44 | 0.97 | translation | 0.344 |
| `lex:entailment` | 741 | 1.93 | 0.11 | 70 | 0.87 | orthogonal | 0.324 |
| `lex:similar_to` | 10123 | 2.03 | 0.05 | 89 | 0.84 | orthogonal | 0.295 |
| `lex:substance_meronym` | 340 | 1.51 | 0.17 | 44 | 0.98 | orthogonal | 0.265 |
| `lex:cause` | 231 | 1.57 | 0.14 | 44 | 0.91 | affine | 0.253 |
| `lex:antonym` | 1736 | 1.06 | 0.10 | 57 | 0.97 | orthogonal | 0.250 |
| `lex:hyponym` | 64201 | 1.87 | 0.10 | 132 | 0.80 | orthogonal | 0.238 |
| `lex:hypernym` | 64201 | 3.98 | 0.10 | 132 | 0.81 | orthogonal | 0.199 |

## By family

| family | direction | eff dim | detect AUC | identity | translation | orthogonal | mlp |
|---|---|---|---|---|---|---|---|
| inflectional | 0.342 | 43.9 | 0.988 | 0.296 | 0.476 | 0.548 | 0.592 |
| derivational | 0.392 | 27.7 | 0.973 | 0.147 | 0.347 | 0.275 | 0.361 |
| lexicographic | 0.127 | 77.9 | 0.925 | 0.120 | 0.130 | 0.321 | 0.383 |

## Findings

1. **Translation is a morphology-only premise.** identity->translation gains +0.179
   (inflectional) and +0.200 (derivational) but **+0.009** (lexicographic).

2. **Semantic relations are isometries.** For lexicographic relations `orthogonal`
   gains +0.201 where translation gains +0.009, and wins 10 of 14. Meaning-relations
   rotate words; they do not displace them.

3. **Verification is easy, generation is hard, for every relation.** detect AUC
   0.93-0.99 vs best generation 0.39-0.59. Detectability is the ONLY strong predictor
   of generability (spearman +0.741, p=4e-6); 'is it a direction' is not (+0.32,
   p=0.09), nor effective dimension (-0.12), nor fan-in (-0.29).

4. **Cosine is the wrong metric**, and it was the Harbor task's grading metric.
   Over all 29x6 shape fits, spearman(cos, R@1_any) = +0.181. `affine` has the
   highest mean cosine (0.810) and mediocre retrieval (0.211); the MLP has the
   lowest cosine (0.495) and the best retrieval (0.411). Ridge predicts a generic
   central point: close to everything, useful for nothing.

## Caveats

- The MLP is trained with a **retrieval** objective; the closed-form shapes minimise
  squared error. The R@1 comparison structurally favours the MLP -- and it still gains
  only +0.043 / -0.019 / +0.004 by family. Capacity is not the constraint. Comparisons
  among the linear shapes are fair.
- `detect AUC` uses **matched random** negatives. Against the hard negatives retrieval
  actually produces (the target's own synonyms and morphological relatives), the same
  probe falls from 0.824 to 0.719 and reranking the generator's top-50 *lowers* R@1
  from 0.277 to 0.080. 'Verification is easy' does not license 'build a verifier'.
- Effective dimension is a participation ratio; it is comparable across relations here
  because all are measured in the same frozen space (unlike the earlier 13.5-vs-61.4
  cross-space comparison, which was invalid).
