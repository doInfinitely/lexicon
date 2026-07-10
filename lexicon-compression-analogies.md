# Lexicon Compression via Learned Analogy Operators

**Category:** Natural Language Processing · **Estimated Difficulty:** Hard · **Calibration target:** Gemini 3.5 Flash · **Compute:** CPU-only · **Shape:** Optimize-a-metric

## Economic Value

Language models and embedding systems carry enormous vocabularies where many words are systematically related — plurals, hypernyms, antonyms, derivations. If a compact base lexicon plus a small set of learned operators (e.g. `hypernym(embed(cat)) ≈ embed(animal)`) can reconstruct the full vocabulary's embeddings, this enables: (1) vocabulary compression for edge deployment, (2) structured word representations that expose compositional semantics, (3) better generalization to rare/unseen words through operator composition. This mirrors real production work in morphological analyzers, knowledge graph embedding, and vocabulary-efficient language models. The task requires jointly solving a combinatorial problem (which words are base vs. derived?) and a continuous optimization problem (fine-tune the model so operator outputs land near target embeddings) — the kind of planning+optimization that separates strong engineering from naive baselines.

## Why This Problem Is Hard

- **Joint combinatorial + continuous optimization is the core challenge.** The agent must simultaneously decide which words belong in the base lexicon (a discrete selection problem), which operator maps each derived word to its base word(s) (an assignment problem), and how to fine-tune the embedding model so operator outputs match target embeddings (a continuous optimization problem). Decoupling these — e.g., greedily picking the lexicon first, then learning operators — leads to suboptimal local optima because the best lexicon depends on what the operators can learn, and vice versa.

- **Operator design requires understanding relation structure.** The BATS dataset contains relations of different arities and types: unary morphological transforms (plural, past-tense), binary semantic relations (hypernym, meronym), and encyclopedic associations (capital-of, member-of). The agent must design operators that capture these different structures — a single operator architecture won't work for all. Too few operators underfit; too many defeat the compression goal.

- **The laziest wrong method: keep most words as base, use operators only for trivial morphological inflections.** A strong model will recognize that plurals (cat→cats) and verb conjugations (run→ran) are easy to learn, so it will put all "hard" words in the base lexicon and only derive trivial inflections. This achieves high accuracy but terrible compression ratio — the verifier requires both metrics above threshold simultaneously.

- **The genuine crux the agent must derive:** how to iteratively refine the base lexicon by measuring which non-base words the current operators can reconstruct well, and which base words are most "productive" (enable the most derivations). This requires a training loop that alternates between (1) updating operator parameters and fine-tuning embeddings, and (2) re-evaluating which words should be base vs. derived based on current reconstruction quality. Purely greedy or one-shot lexicon selection fails because it can't anticipate what the operators will learn to do.

## Environment Construction

### Key Resources
- **BATS (Bigger Analogy Test Set)** — Gladkova, Drozd & Matsuoka (NAACL 2016). 40 relation categories × 50 word pairs each. 4 types: inflectional morphology (10), derivational morphology (10), lexicographic semantics (7), encyclopedic semantics (13). Tab-separated files with multiple valid answers separated by `/`. Download: https://my.pcloud.com/publink/show?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV
- **distilbert-base-uncased** — HuggingFace transformers. Pre-trained, CPU-feasible fine-tuning.
- **Hint-probe:** "The key insight is that the base lexicon should be selected iteratively: start with all words as base, train operators on the relation pairs, then remove from the base lexicon any word whose embedding can be reconstructed within tolerance by applying an operator to other base word(s). Repeat until the lexicon stabilizes. Words involved in many relations as the *source* (e.g. 'cat' appears in plural, hypernym, meronym relations) should be prioritized as base words. Use cosine similarity > 0.85 as the reconstruction criterion during pruning."

### Architecture
Single CPU container: `python:3.11-slim` + PyTorch 2.3.1 (CPU) + transformers + numpy + scipy. No GPU.

| File | Contents | Notes |
|---|---|---|
| `data/bats/` | Full BATS dataset: 40 subdirectories of tab-separated pair files | Read-only, SHA-pinned |
| `data/vocab.json` | Sorted list of all unique words appearing in BATS (~2000 words) | Read-only, derived from BATS at build time |
| `data/relations.json` | Metadata: relation name → category, arity, list of (source, targets) pairs | Read-only, derived from BATS at build time |
| `data/train_pairs.json` | 70% of pairs per relation (stratified split) for training | Read-only |
| `data/test_pairs.json` | 30% of pairs per relation (held-out) for evaluation | Hidden in tests/ layer |

The agent implements:
- `solution/model.py` — `LexiconCompressor` class with methods:
  - `get_base_lexicon() → list[str]` — returns the base word list
  - `get_operators() → dict[str, callable]` — returns named operators
  - `encode(word: str) → torch.Tensor` — returns the embedding for any word (base words use the fine-tuned model directly; derived words compose operators over base word embeddings)
  - `reconstruct(word: str) → tuple[str, torch.Tensor]` — returns (derivation_expression, embedding) where derivation_expression shows how the word was composed (e.g. "hypernym(cat)" or base word name)
- `solution/train.py` — `train(checkpoint_path: str)` → trains operators, selects lexicon, fine-tunes embeddings, saves everything

### Grading Approach

**1. Compression gate (hard requirement):**

| Gate | Metric | Threshold | What the lazy method scores |
|---|---|---|---|
| Compression ratio | N_total / N_base (total words / base lexicon size) | ≥ T_compress (~2.5) | ~1.2 (keeps 80%+ as base) |

**2. Accuracy gate (primary — continuous, natural failure diversity):**

| Gate | Metric | Threshold | What the lazy method scores |
|---|---|---|---|
| Train reconstruction | Mean cosine similarity between `encode(word)` and reference distilbert embedding, across training-split derived words | ≥ T_train (~0.80) | ~0.70 (trivial averaging operator) |
| Held-out reconstruction | Mean cosine similarity on held-out pairs: apply the correct operator to the source word's base embedding, compare to target word's reference embedding | ≥ T_test (~0.65) | ~0.50 (operators don't generalize) |

The continuous cosine similarity metric gives natural failure diversity — different agents fail on different relation categories (morphological vs. semantic vs. encyclopedic).

Exact thresholds calibrated by builder. Initial values above are estimates. The builder should run the lazy baseline (keep all words as base, train identity operators only on inflectional morphology) and set thresholds between lazy and oracle.

**3. Structural gates (hard requirements):**
- Base lexicon is a proper subset of the full vocabulary (compression ratio check)
- Every non-base word has a valid derivation expression (operator name + source word(s) that are all in the base lexicon or derivable from it)
- No circular derivations
- At least 3 distinct named operators (prevents degenerate single-operator compression)
- Model parameter count ≥ 1000 (prevents lookup tables)

**4. Anti-cheat (Appendix B):**
- Reference embeddings computed independently by the verifier using a frozen copy of distilbert-base-uncased (not the agent's fine-tuned model) — the agent's `encode()` must produce embeddings close to the *reference* model's embeddings, not its own. This prevents the agent from collapsing all embeddings to a single point.
- Actually: the evaluation compares the agent's composed embedding for derived words against the reference distilbert embedding. The agent fine-tunes its own model, and the verifier checks that the operator-composed outputs land near where distilbert puts those words. So the agent is scored on how well its operators + fine-tuned embeddings approximate the pretrained distilbert space.
- SHA-pin all read-only data files
- Held-out test pairs hidden in tests/ layer (agent trains on train_pairs.json only)
- Verifier loads the agent's model in an isolated process, calls its interface methods, compares against independently computed reference embeddings

### Agent Starting State

`/workspace/` contains:
- `data/bats/` — the full BATS dataset (40 relation directories)
- `data/vocab.json` — all unique words
- `data/relations.json` — relation metadata with category/arity
- `data/train_pairs.json` — training pairs (70% stratified split)
- `solution/model.py` — stub `LexiconCompressor` class with NotImplementedError
- `solution/train.py` — stub `train()` function

The instruction tells the agent:
- The goal: compress the vocabulary by finding a minimal base lexicon + learned operators such that every word can be expressed as an operator applied to base word embedding(s)
- The model interface (class name, method signatures, input/output types)
- That distilbert-base-uncased is available (pre-downloaded in the image)
- The evaluation criteria: compression ratio ≥ threshold AND reconstruction cosine similarity ≥ threshold on both training and held-out pairs
- That held-out pairs come from the same relation categories but different word pairs
- The training time budget
- That operators should be named and correspond to semantic/morphological relation types

The instruction does NOT reveal:
- Which relation categories are hardest to learn operators for
- The optimal base lexicon selection strategy
- How to balance compression vs. accuracy
- The specific held-out pairs

## Appendix: Making It Harder

- **Stricter compression ratio** — push T_compress from 2.5 to 3.5+, forcing the agent to derive more words through operator composition (fewer base words)
- **Tighter held-out accuracy** — push T_test toward 0.75, requiring operators that genuinely generalize to unseen word pairs
- **Operator count constraint** — limit the maximum number of operators (e.g. ≤ 10), forcing the agent to discover that multiple BATS categories can share operators
- **Derivation depth limit** — require all derived words to be reachable in ≤ 2 operator applications from base words (prevents trivial chaining)
- **Cross-category held-out** — hold out entire relation categories (not just pairs within categories), testing whether operators transfer to unseen relation types
- **Larger vocabulary** — augment BATS with additional word pairs from WordNet, increasing the vocabulary to 5000+ words
