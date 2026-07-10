# The Lexeme Language

A token language in which the lexicon is *generated* rather than listed. Words
are not atoms: a word is either a lexeme, or an operator applied to lexemes.

## 1. Token classes

| class | count | example |
|---|---|---|
| structural | 7 | `<bos> <eos> <pad> <unk> </op> <wp> </wp>` |
| operators | 27 | `<op:noun.plural>`, `<op:antonym>`, `<op:hypernym/3>` |
| lexemes | 34,593 | `<lex:walk>` |
| wordpiece escape | 30,522 | `<wp:##ing>` |
| **total** | **65,149** | |

The 34,593 lexemes generate 51,148 attested surface forms; 16,555 of those are
compositional, to derivation depth 4. Nothing is unreachable: anything outside
the vocabulary is spelled with the wordpiece escape.

## 2. Grammar

```
expr    := lexeme | apply | escape
lexeme  := "<lex:" root ">"
apply   := "<op:" name ["/" arity] ">" expr{arity} "</op>"
escape  := "<wp>" wordpiece+ "</wp>"
```

Operators bracket their inputs, so arity is read off the stream and nesting is
free. Unary operators omit the arity suffix.

```
walkers       <op:noun.plural> <op:noun.agent> <lex:walk> </op> </op>
cold          <op:antonym> <lex:hot> </op>
carnivore     <op:hypernym/2> <lex:cat> <lex:dog> </op>
document      <op:hypernym/3> <lex:coupon> <lex:diploma> <lex:giro> </op>
quux          <wp> <wp:qu> <wp:##ux> </wp>
```

## 3. Operator inventory

**Slot realisers (arity 1).** A lexeme is category-neutral; a slot spells it
out. POS-typed: `noun.agent` takes a verb to a noun, `adj.comp` an adjective to
an adjective. Without the type constraint `teacher` parses as `adj.comp(teach)`,
because `-er` is homophonous.

```
noun.plural  verb.3sg  verb.ger  verb.past  verb.ptcp  adj.comp  adj.sup
noun.agent   noun.action  noun.action2  noun.quality  noun.quality2
noun.ist  noun.ism  verb.ize  adv  adj.y  adj.al  adj.ic  adj.ous
adj.able  adj.ful  adj.less
```

**Antonym (arity 1, involutive).** `antonym(antonym(x)) = x`. Realised as a
reflection across a learned surface. In the *frozen* space this is not exactly
involutive (round-trip 0.63) because antonym pairs lie outside the surface's
reach. In a **fine-tuned** space it is (round-trip 0.999).

**Hypernym (arity 2-4, set operator).** The least common subsumer of its
inputs. `hypernym(dog)` is ill-posed -- `canine`, `carnivore`, `mammal` are all
true, fan-in 7.14. `hypernym(cat, dog)` is a function. Held-out R@1: unary
0.159, binary 0.339, ternary 0.421, and the data thins past arity 4.

## 4. Embeddings

`dictionary/embeddings.npy` holds one row per lexeme: the **centroid** of its
surface forms in the all-but-the-top space. Learned lexeme vectors barely beat
the centroid (0.771 vs 0.746 on paradigm completion), so the mean is what we
keep. Initialise `<lex:*>` embeddings from these rows and slot operators from
the fitted linear maps.

A held-out lexeme centroid plus its slot operators regenerates **82.8%** of its
surface forms at rank-1 out of 51,148 candidates (base-word anchor: 70.3%).
Applying an operator to the *wrong* lexeme retrieves the right word 0.000 of
the time -- the operators realise a root, they do not memorise a table.

## 5. Model variants

Four input/output regimes, all expressible in this language:

1. **tokens in / tokens out** -- ordinary autoregression over the inventory.
2. **vectors in / vectors out** -- the model consumes and emits embeddings; a
   decoder maps a vector back to a word by softmax over cosine similarity to
   the lexeme table with temperature `T` (`T -> 0` is nearest-neighbour).
3. **vectors in / tokens out**, 4. **tokens in / vectors out** -- hybrids.

For a model that reads *both*, run two aligned streams with a per-position
switch at the embedding layer: `TokenEmbed[t]` or `W_in · v`, trained with
random modality dropout, and two output heads (token softmax, vector head).
Every derived word supplies free paired supervision: it has a token form
(`<op:verb.past> <lex:walk> </op>`) and a vector form (the operator's output).

Homographs get one lexeme entry per sense where the corpus supports it (909
words showed >1 sense at silhouette >= 0.30: `rock`, `hull`, `formula`,
`receiver`). Context-conditioned decoding picks the sense.

## 6. Known weaknesses -- read before trusting this

- **The antonym operator is weak.** Held-out R@1 0.23-0.26 out of 51k. Its
  gain over doing nothing comes mostly from the *space*, not the operator: in a
  fine-tuned space an identity operator scores 0.215 and the reflection 0.230.
- **Fine-tuning causes catastrophic forgetting.** Without rehearsal, adapting
  the space for antonymy drops `verb_Ved` retrieval from 0.422 to 0.076. With
  rehearsal on morphological relations, morphology *improves* (plural 0.719 ->
  0.871) and antonymy still gains. Always rehearse.
- **A free adapter absorbs operator structure.** A random, never-trained
  reflection plane matched a trained one (0.300 vs 0.301). Do not interpret an
  operator's parameters when an adapter sits in front of it.
- **Cosine is not retrieval.** Across 174 operator fits, `spearman(cos, R@1) =
  +0.181`. Score operators by rank-1 retrieval against the full vocabulary.
- **Semantic operators are near-closed classes.** `male_female` derives ~10
  held-out words because English has ~40 gendered pairs. Morphology is
  productive; most semantic relations are not. The exception is arity-2+
  hypernymy, which is productive across the noun taxonomy.
- `happy` parses as `adj.y(hap)`. Etymologically defensible, practically odd.
  A frequency floor on roots would fix it.
