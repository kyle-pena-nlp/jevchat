# Experiment log

Everything measured while turning Jev into a chat model, with the numbers that
decided each design choice. Several of these overturned an expectation, so the
failures are recorded as carefully as the wins.

Reproducible measurements are marked **`jevchat bench`**. The rest were one-off
scripts run against the live API; their numbers are quoted here because the scripts
were not kept.

**Read the [noise floor](#0-the-noise-floor) first.** It was measured last and it
re-scales every other result on this page.

---

## 0. The noise floor

*How much does Jev disagree with itself?*

The same 88-option `ascii` question, asked twice in one request, differing only in
the order of the options. Five contexts.

| | value |
|---|---|
| top-1 agreement | 4/5 |
| Spearman | 0.91 |
| total variation distance | 0.30 |
| mean \|log-odds difference\| | 0.90 nats |

Jev does not reproduce its own distribution exactly. One case even flipped its
top-1 symbol between the two askings.

**This bounds everything else.** No sampler can converge on a target more precisely
than the oracle defines it, and any comparison in this document closer than
~0.9 nats or ~0.9 Spearman is at or near the floor. It was measured too late —
several earlier conclusions would have been stated more cautiously with it in hand.

---

## 1. The API

Measured against `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`.

| limit | value | how it fails |
|---|---|---|
| options per `choice` question | **255** | `400 Too many choices. Must have at most 255 choices.` |
| levels per `score` question | **10** | `400 Too many score levels. Must have at most 10 levels.` |
| request size | **>65,715 input tokens** | `400 {"error_type": "max_tokens_exceeded"}` |

A single request carrying 65,715 input tokens succeeded; 16 questions of 255
described options (~88,000 tokens) failed. Cloudflare's model page lists a
32,000-token context — that figure does not describe this endpoint.

### Question types

`score` is **not** a probability. Its `score` field is the expected index under the
returned `probabilities` map. Verified at three widths:

| levels | probabilities | Σ i·pᵢ | returned `score` |
|---|---|---|---|
| 3 | {0: 0.70, 1: 0.04, 2: 0.26} | 0.56 | 0.57 |
| 5 | — | 2.000 | 1.98 |
| 8 | — | 2.890 | 2.87 |

The `probabilities` map *is* natively normalised (sums to 1 ± 0.01 rounding), for
both `choice` and `score`.

**`choice` is the only type that can carry an alphabet** — `score` caps at 10 levels
and is ordinal, which letters are not.

### Quantisation

Probabilities are rounded to two decimals, so anything below 0.005 returns exactly
`0.0`. The surviving support is roughly constant regardless of alphabet size:

| alphabet | options | non-zero in one call |
|---|---|---|
| `lower26` | 28 | 17 |
| `ascii` | 89 | 25 |
| `tokens` | 255 | 25 |

230 of the token alphabet's 255 options carry no information on any given call.

### Parallel questions are nearly free

Latency against question count, one request, 89-option questions:

| questions | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| seconds | 0.63 | 0.51 | 0.46 | 0.54 | 0.73 | 0.95 |

Input tokens scale linearly; wall time barely moves. **This single fact shapes every
strategy in the project** — a whole decision tree, or a whole bucketed vocabulary,
costs about one round trip.

---

## 2. Prompt framing

*Does telling Jev to judge the concatenation help?*

Two wordings, same eight prefix-continuation cases, `tokens` alphabet.

| framing | top-1 | typical confidence on correct answers |
|---|---|---|
| "Which symbol comes next?" | 6/8 | 0.76, 0.24, 0.86 |
| "…makes `answer_so_far` + option a valid prefix of a grammatical, factually correct answer" | 6/8 | 0.94, 0.58, 0.95, 0.97, 0.93 |

Top-1 unchanged, distributions far sharper. The second wording shipped as the
default instructions.

---

## 3. Position bias

*Does option order change the answer?*

Same state, same 89 options, different orderings of the criteria map. Target: the
correct continuation `'i'` after `"The capital of France is Par"`.

| criteria order | rank of `'i'` |
|---|---|
| natural `a, b, c, …` | 6 |
| shuffled (4 different seeds) | 1, 1, 1, 1 |

Asked with an empty question and empty prefix, Jev's prior over `ascii` puts
**80% of its mass on `STOP` and `A`** — the first two entries.

**`jevchat bench`**, 12 character continuations on `ascii`:

| criteria order | top-1 | top-3 | ms/step | in-tok/step |
|---|---|---|---|---|
| natural | 3/12 | 4/12 | 231 | 1,893 |
| shuffled, ensemble 1 | 5/12 | 6/12 | 313 | 1,893 |
| shuffled, ensemble 4 | 5/12 | 8/12 | 227 | 6,492 |
| shuffled, ensemble 12 | 5/12 | 8/12 | 325 | 18,756 |

**Shipped:** `shuffle_criteria = true`. Free, and the largest single lever before
hypothesis options.

### Ensemble recovers the quantised tail

Averaging N shuffled orderings gives effective granularity 0.01/N:

| ensemble | non-zero of 255 | smallest non-zero |
|---|---|---|
| 1 | 25 | 0.0100 |
| 2 | 36 | 0.0050 |
| 4 | 52 | 0.0025 |
| 8 | 66 | 0.0013 |
| 16 | — | `max_tokens_exceeded` |

It buys support, not accuracy. **Shipped:** `ensemble = 1` by default.

---

## 4. Strategies

### bisect — earlier/later questions (shipped, not recommended)

Sort the alphabet, split in half repeatedly with `noul` questions, then one choice
question in the group reached.

**Result: worse.** Splits come back near coin flips (0.45–0.65), and multiplying
three or four of them dilutes the signal faster than the finer leaf resolution
recovers it.

| mode | top-1 | top-3 | P(correct) | non-zero | in-tok |
|---|---|---|---|---|---|
| choice, symbol | 2/12 | 6/12 | 0.249 | 18 | 1,893 |
| bisect, cutoff 20 | 3/12 | 7/12 | 0.128 | 70 | 5,915 |

Better coverage (70 symbols carry probability against 18), half the accuracy. On
`words1k` it scored 5/10 at 58,133 tokens per step. It is the interesting mode, not
the good one.

### buckets — slices with an OTHER escape (shipped, default)

Split the alphabet into questions of ≤254 options, each with an extra `OTHER`
option meaning "the next symbol is not in this bucket". OTHER is what makes the
buckets comparable, since Jev normalises within a question.

**Go/no-go**, 6 buckets from `tokens`, 6 contexts: the bucket holding the answer had
the lowest `P(OTHER)` **6/6**; concatenation recovered the right answer 5/6; removing
the correct answer raised `P(OTHER)` by +0.30 on average.

**Scaling**, 1,009 words, bucket count K:

| K | bucket size | holder found | top-1 | ΣP(OTHER) (ideal K−1) | in-tok | latency | requests |
|---|---|---|---|---|---|---|---|
| 4 | 253 | 10/10 | 9/10 | 2.04 (3) | 6,970 | 0.3s | 1 |
| 8 | 127 | 10/10 | 9/10 | 5.39 (7) | 7,526 | 0.2s | 1 |
| 16 | 64 | 10/10 | 9/10 | 12.99 (15) | 8,638 | 0.2s | 1 |
| 32 | 32 | 10/10 | 9/10 | 27.64 (31) | 10,862 | 0.3s | 1 |

Three findings:

* **Holder identification is perfect up to K=32.** The false-positive pressure I
  expected at larger K never appeared.
* **`P(OTHER)` is miscalibrated but its ranking is sound.** It reaches only 68% of
  its ideal sum at K=4, improving to 89% at K=32 — Jev under-reports "not here."
  Calibration *improves* as buckets shrink.
* **Weighting each bucket by `1 − P(OTHER)` scored identically to plain
  concatenation**, so the simpler form shipped.

Per-option descriptions cost 2.3× the tokens (17,618 vs 7,526 at K=8) for no
accuracy change. **Shipped:** `bucket_describe = false`.

---

## 5. Vocabulary

*Coverage or ranking — which matters more?*

Ten next-word continuations, scoring only cases the alphabet can express.

| alphabet | options | expressible | top-1 | in-tok/step | ms/step | requests |
|---|---|---|---|---|---|---|
| `tokens` | 255 | 4/10 | 3/4 | 2,262 | 230 | 1 |
| `words1k` | 1,122 | 10/10 | **9/10** | 8,593 | 263 | 1 |
| `bpe2k` | 2,433 | 6/10 | 5/6 | 18,388 | 370 | 1 |
| `bpe5k` | 5,228 | 9/10 | 7/9 | 38,949 | 564 | 1 |

**5,228 options resolve in a single request.** The 255 cap is a per-question limit,
not a real constraint.

**Coverage is the differentiator, not ranking** — top-1 on expressible cases sits
between 75% and 90% everywhere. What separates the alphabets is how often the word
you need is on the list at all.

**BPE frequency order is not chat coverage.** `bpe2k` has twice the options of
`words1k` and expresses fewer targets: BPE ranks by compression value, so fragments
like `'ning'` and `'ced'` outrank whole words. `' blue'` is GPT-2 token 4171,
`' cold'` 4692, `' honey'` 12498. A curated thousand-word list beats a 2,433-piece
BPE prefix.

`bpe5k` writes the most fluent sentences (`"The capital is Parys"`, `"They make a
lot of food"` where `words1k` produces word salad) but costs 5× the tokens and
spreads `P(correct)` down to 0.07 from 0.32.

**Shipped:** `words1k` as the default alphabet.

---

## 6. Hypothesis options

*Offer the resulting texts instead of the symbols.*

Instead of options `'a' 'i' 's' … STOP`, offer `'…France is Para'`,
`'…France is Pari'`, and — for STOP — the reply left unchanged, `'…France is Par'`.
Jev then ranks finished strings rather than appending in its head. Options show
only the reply's tail so cost stays flat as the reply grows.

**`jevchat bench`**, 12 character continuations on `ascii`:

| mode | top-1 | top-3 | P(correct) | non-zero | ms/step | in-tok | questions |
|---|---|---|---|---|---|---|---|
| choice, symbol, no shuffle | 3/12 | 3/12 | 0.181 | 14 | 208 | 1,893 | 1 |
| choice, symbol | 2/12 | 6/12 | 0.249 | 18 | 175 | 1,893 | 1 |
| choice, symbol, ensemble 4 | 2/12 | 9/12 | 0.223 | 24 | 213 | 6,492 | 4 |
| **choice, hypothesis w24** | **10/12** | **11/12** | **0.526** | 11 | 194 | 1,420 | 1 |
| choice, hypothesis w40 | 8/12 | 10/12 | 0.554 | 10 | 185 | 1,406 | 1 |
| bisect, cutoff 20 | 3/12 | 7/12 | 0.128 | 70 | 192 | 5,915 | 23 |
| buckets, symbol | 3/12 | 8/12 | 0.227 | 22 | 172 | 1,106 | 2 |
| **buckets, hypothesis w40** | **10/12** | **11/12** | **0.555** | 11 | 185 | 1,520 | 2 |

Three to five times the top-1, double the probability mass on the right symbol,
**for fewer input tokens** than symbol options with their per-option descriptions.
The largest single improvement in the project.

Three qualifications:

* **Neutral at word level.** On `words1k` word continuations, symbol and hypothesis
  both score 9/10 (P(correct) 0.318 vs 0.320). The win is concentrated where a
  symbol alone carries no meaning — a bare `'i'` is unjudgeable, `' blue'` is not.
* **No help at the first step.** With an empty reply there is no prefix, so a
  hypothesis option is just the symbol again.
* **The unchanged option attracts spurious mass**, being a valid prefix of every
  extension of itself:

| | mean P(stop), reply unfinished | mean P(stop), reply complete | separation |
|---|---|---|---|
| symbol options | 0.00 | 0.80 | +0.80 |
| hypothesis options | 0.08 | 0.63 | +0.55 |

Left uncompensated, generation stopped after two or three characters. **Shipped:**
`stop_bias = 0.5`, `min_steps = 3`, which produced `"Paris."` from a character
alphabet — the first correct multi-character generation in the project.

---

## 7. Pairwise comparison and IIA

*Can Bradley–Terry replace bulk scoring?*

Eight candidate symbols per context, all 28 pairs asked in both orders as parallel
questions, three contexts.

| context | BT fit error (mean \|pred−obs\|) | intransitive triples | Spearman(BT, full) | top-1 agree |
|---|---|---|---|---|
| `"…is Par"` | 0.073 | 0/56 | 0.90 | yes |
| `"…sky is bl"` | 0.084 | 0/56 | 0.74 | yes |
| `"Grass is gre"` | 0.051 | 0/56 | 0.71 | yes |

**Transitivity is clean — zero intransitive triples anywhere.** A single latent
strength per token explains the pairwise comparisons well, so a Metropolis–Hastings
chain driven by pairwise questions has a well-defined stationary distribution. MH
needs only ratios, never a normalising constant, and a ratio is a 2-option question.

**But IIA fails.** Pairwise and bulk disagree by **2.02–2.27 nats**, and
directionally: the bulk question is far peakier.

| | pairwise (BT) | bulk |
|---|---|---|
| `'i'` after `"…is Par"` | 0.70 | 0.95 |
| `'u'` after `"…sky is bl"` | 0.39 | 0.94 |

Fitting a single temperature gives α = 1.46–1.78 and only reduces the residual from
2.11 → 1.30, 2.02 → 1.67, 2.27 → 2.11 nats. **A scale factor explains about a third
of the gap and α is not stable across contexts.**

Against the 0.90-nat noise floor, the violation is real but roughly 1.1–1.4 nats
above noise — smaller than the raw figure suggests.

The pairwise distribution is not simply a degraded copy: it is flatter at the top
but **sharper at the bottom**, since 2–5 of 8 candidates read exactly `0.00` in the
bulk question and every candidate gets a finite strength pairwise.

---

## 8. Anchor linking (failed)

*Replace OTHER with shared reference tokens.*

Put the same few tokens in every bucket. Under Luce's axiom each bucket's reported
log-probabilities are the true log-strengths shifted by one constant, so a shared
anchor pins that constant: `log Z_k = log s_a − log p_k(a)`. This is common-item
equating from psychometrics. Anchors are *measured* where OTHER is *asserted*, so it
should have been better.

Reconstructing an 88-option `ascii` question from 6 buckets, 5 contexts:

| method | top-1 | Spearman | TV dist | \|log-odds err\| |
|---|---|---|---|---|
| *noise floor* | *4/5* | *0.91* | *0.30* | *0.90* |
| OTHER (measured 3×) | 4/5 | **0.74 – 0.76** | 0.33 – 0.40 | 0.91 – 1.09 |
| anchors, 2 | 2/5 | 0.31 | 0.49 | 1.12 |
| anchors, 4 | 4/5 | 0.30 | 0.43 | 1.04 |
| anchors, 8 | 4/5 | 0.23 | 0.38 | 1.11 |

OTHER was re-measured alongside each anchor-set size (the member pool differs,
since anchors are held out), and came back at 0.74/0.75/0.76 Spearman each time.

**Anchors lose decisively, and more anchors made it worse.**

The obvious explanation — that the 0.01 floor zeroes the anchors — is **wrong**.
Only 9% of anchor readings are zeroed, and dropping those from the offset changed
nothing (0.31→0.31, 0.30→0.30, 0.23→0.23).

The real cause: **anchors in the same bucket disagree about that bucket's scale by
1.62 nats** (a factor of 5). Under Luce every anchor would imply the same offset.
This is a cleaner measurement of the IIA violation than §7, because it holds set
*size* fixed and varies only set *composition*. More anchors means more mutually
inconsistent estimates, hence the degradation with 8.

OTHER survives because it needs only one number per bucket, read directly. A biased
direct measurement beat an unbiased inference through a model that does not hold.

Note also that OTHER's reconstruction (0.75 / 0.36 / 0.99) sits close to the noise
floor (0.91 / 0.30 / 0.90) — bucketing costs less fidelity than the raw gap suggests.

---

## 9. Trie filtering for a 50k vocabulary

*Use a character model to shortlist BPE tokens.*

Build a depth-D trie of next-character distributions from `ascii`, then score all
49,861 `bpe50k` tokens locally by the product of their leading characters'
probabilities. Gold = any BPE token that is a prefix of the correct continuation.

| trie | recall@1 | recall@8 | recall@16 | mean survivors |
|---|---|---|---|---|
| depth 1, branch 2 | 0/8 | 0/8 | 0/8 | 29,557 |
| depth 2, branch 2 | 6/8 | 7/8 | 7/8 | 1,183 |
| **depth 3, branch 2** | 6/8 | **8/8** | **8/8** | **148** |
| depth 3, branch 5 | 6/8 | 8/8 | 8/8 | 1,261 |

**Depth 3 cuts 49,861 tokens to 148 and keeps a prefix-correct token in the top 8
in every case** — a 337× prune with no loss. Depth 1 is worthless: one character
(usually a space) leaves ~30k survivors.

Recall is excellent, **ranking is mediocre** (recall@1 only 6/8) — the trie is a
candidate generator, not a distribution.

Two caveats. The trie is **sequential across levels** (level 2 needs level 1's
branches), and the implementation used one request per *node* because Jev takes one
`state` per request while each trie path has a different `answer_so_far`. Batching
per level should be possible under hypothesis presentation, since option labels
carry the text — but the instructions say options are the reply "with one more
symbol added," which is false at depth ≥2, so that is **untested**.

Also: filtering is truncation, not simulation. The 49,713 discarded tokens get zero,
not a small number. That is exactly what top-k and nucleus sampling already do, but
it is not the full distribution.

---

## What shipped

| decision | because |
|---|---|
| `choice`, not `score` | `score` caps at 10 levels and is ordinal |
| `shuffle_criteria = true` | position bias; free; 3/12 → 5/12 |
| `presentation = hypothesis`, `window = 40` | 3–5× top-1, fewer tokens |
| `stop_bias = 0.5`, `min_steps = 3` | the unchanged-reply option over-attracts |
| `strategy = buckets`, `bucket_size = 127` | most accurate *and* cheapest |
| `alphabet = words1k` | coverage beats size; `tokens` cannot say "white" |
| `ensemble = 1` | buys support, not accuracy |
| `bucket_describe = false` | 2.3× tokens, no gain |
| OTHER, not anchors | anchors disagree by 1.62 nats |
| concatenate, not `1 − P(OTHER)` weighting | scored identically |

## Open questions

* **Simulating the full distribution** rather than truncating to a shortlist. The
  pairwise/Bradley–Terry route is the only one measured that assigns finite mass to
  everything it touches, but it converges to a flatter distribution than bulk
  scoring and the temperature correction does not close the gap.
* **Batching trie levels** into one request per level (§9) — untested, worth ~10×.
* **Semantic bucketing.** `bisect` failed partly because alphabetical halves are
  semantically meaningless. GPT-2's own token embedding matrix (50257 × 768) would
  give a free similarity metric over exactly `bpe50k`, enabling clustered buckets
  and neighbourhood proposals for MCMC.
* **Parallelising bucket requests.** They are independent; `buckets.ask` sends them
  sequentially. Worth ~3× on the large alphabets.
* **The noise floor itself.** Everything above is bounded by Jev disagreeing with
  itself at 0.90 nats. Repeated measurement under re-randomised conditions is the
  one provable way to reduce it; `shuffle_criteria` and `ensemble` are the existing
  levers.
