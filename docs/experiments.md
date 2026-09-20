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

The floor was re-measured alongside §10 and §11, and it moves:

| measurement | top-1 | Spearman | TV | \|log-odds\| |
|---|---|---|---|---|
| §0 | 4/5 | 0.91 | 0.30 | 0.90 |
| §10 | 4/5 | 0.95 | 0.23 | 0.69 |
| §11 | 4/5 | — | 0.22 | 0.73 |

So the floor is itself a range — Spearman 0.91–0.95, TV 0.22–0.30, 0.69–0.90 nats —
and comparisons in that band should be read as ties.

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

## 10. Representative weighting, flat and hierarchical

*Replace OTHER with a question over bucket winners.*

Pass 1 asks K plain buckets (no OTHER, no anchors) for `p(x | bucket)`. Pass 2 asks
**one** question over each bucket's argmax, so those K numbers are on a single
normalised scale by construction. The representative is a *probe*, not a stand-in:

```
w_k  ∝  r_k / p(rep_k | bucket k)          p(x) = w_k · p(x | k)
```

which is exact under Luce — a bucket whose winner is strong but which is otherwise
empty gets scaled down correctly.

Reconstructing an 88-option `ascii` question, 5 contexts:

| method | top-1 | Spearman | TV | \|log-odds\| | non-zero |
|---|---|---|---|---|---|
| *noise floor* | *4/5* | *0.95* | *0.23* | *0.69* | *11* |
| OTHER, K=6 | 4/5 | 0.89 | 0.40 | 0.85 | 21 |
| reps, K=6 | 5/5 | 0.51 | 0.29 | 0.89 | 57 |
| OTHER, K=22 | 2/5 | 0.70 | 0.49 | 0.97 | 31 |
| reps, K=22 | 5/5 | 0.45 | 0.38 | 1.02 | 48 |
| hierarchical 22→5→1 | 4/5 | 0.24 | 0.53 | 1.06 | 82 |

**The predicted dynamic-range failure did not happen.** `w_k = r_k / p(rep_k|k)`
multiplies two quantised values, giving 10⁻⁴ resolution, so representatives resolve
*more* tail than OTHER (57 non-zero against 21) rather than less. The multiplicative
range expected from a hierarchy already appears at two levels.

**Three levels is worse** — TV 0.53. Errors compound across levels, so two is the
design.

**Spearman should not be used for these comparisons.** Ground truth has ~11 non-zero
values out of 88, so 77 are tied at zero and their ranking is arbitrary. A method is
penalised for assigning finite mass to symbols the reference calls zero, which is
the behaviour we want. This also weakens the Spearman column in §8.

**reps vs OTHER is inside the noise.** Re-run with fresh samples (§11) reversed the
ordering — reps 0.37 / OTHER 0.35 against this run's 0.29 / 0.40. Five contexts is
not enough to separate them against a 0.9-nat floor.

---

## 11. Nucleus reselection and reweighting (the robust win)

*Project the first estimate back down, reselect a nucleus, re-aggregate.*

After pass 2 gives a first global estimate `p0`, take the top-m of each bucket by
`p0`, pool them, and score the pool in **one** question. Two ways to use that:

* **splice** — the pooled distribution becomes the head; the tail keeps `p0`'s
  original weights.
* **reweight** — the pooled question re-estimates each bucket's weight, applied to
  the *whole* bucket, tail included. Exact under Luce, since `q(x)/p(x|k) = Z_k/Q`
  is constant across a bucket's members:

```
w_k  ∝  Σ_{x ∈ nucleus_k} q(x)  /  Σ_{x ∈ nucleus_k} p(x | k)
```

Averaging over m members is better conditioned than dividing by a single quantised
argmax probability.

| method | top-1 | TV | \|log-odds\| | non-zero |
|---|---|---|---|---|
| *noise floor* | *4/5* | *0.23* | *0.76* | *10* |
| OTHER | 4/5 | 0.35 | 0.80 | 22 |
| reps (argmax probe only) | 3/5 | 0.40 | 0.87 | 61 |
| splice, m=3 | 4/5 | 0.27 | 0.76 | 56 |
| **reweight, m=3** | 4/5 | 0.28 | **0.67** | 58 |
| splice, m=6 | 3/5 | 0.29 | 0.79 | 39 |
| reweight, m=6 | 3/5 | 0.32 | 0.80 | 55 |

**TV: splice and reweight tie** (0.27 / 0.28). TV is mass-weighted, so it is
dominated by the head, which both fix.

**\|log-odds\|: reweight wins, 0.67 against 0.76.** That metric weights all pairs
equally, so it is the tail-sensitive one — exactly what reweighting the whole bucket
was meant to improve. It is the best figure measured anywhere in this log, and it
sits *below* the single-sample noise floor, which is coherent: the reconstruction
averages six bucket questions and so can be less noisy than one reference sample.

Reweighting also **retains more support** (58 vs 56 at m=3, 55 vs 39 at m=6),
because correcting `w_k` lifts a bucket's tail along with its head rather than
overwriting the head and leaving the tail on stale weights.

`m=3` beats `m=6` on both metrics — a smaller nucleus overwrites less of the first
estimate.

Cost: 3 sequential round trips (each pass needs the previous pass's winners). Only
one and a half turns of the up-down-up loop were run; iterating further is untested
and costs one request per turn.

**Scaling note.** The pooled question holds K×m options — 18 at K=6, m=3, but 1,179
for `bpe50k`'s 393 buckets, over the 255 cap. That is where a hierarchy earns its
place, rather than the three-level version in §10.

### The measurement problem

**There is no ground truth for the tail.** A single 88-option question resolves ~11
symbols, so it cannot validate a method that resolves 58. Every tail comparison on
this page is therefore unverified. The only available check is *reproducibility* —
run the same method twice with different bucket assignments and see whether the same
tail symbols light up. Signal should reproduce; noise should not. Untested.

---

## 12. `refine` shipped as a strategy

§10 and §11 built as `jevchat/refine.py`, reachable as `--strategy refine` and
included in `jevchat bench`.

**On `words1k` (1,122 options — needs bucketing), 10 word continuations:**

| mode | top-1 | P(correct) | non-zero | ms/step | in-tok | questions | reqs |
|---|---|---|---|---|---|---|---|
| buckets, hypothesis w40 | 9/10 | 0.311 | 46 | 310 | 12,074 | 10 | 1 |
| refine, 0 rounds | 10/10 | 0.574 | 867 | 961 | 19,415 | 73 | 3 |
| refine, 1 round m=3 | 9/10 | 0.661 | 870 | 1,183 | 21,775 | 74 | 4 |
| refine, 2 rounds m=3 | 9/10 | 0.659 | 870 | 1,243 | 24,135 | 75 | 5 |
| **refine, 1 round m=6** | 9/10 | **0.674** | **871** | 992 | 21,774 | 74 | 4 |

**Twice the probability on the right symbol, and 19× the vocabulary resolved** —
871 of 1,122 symbols carry finite mass against 46 for `buckets`. Cost is ~4 requests
and ~1s per symbol against 1 request and 0.3s.

A second round adds nothing (0.659 vs 0.661); the up-down-up loop converges after
one turn.

**On `ascii` (89 options — fits one question), 12 character continuations:**

| mode | top-1 | P(correct) | non-zero | in-tok | reqs |
|---|---|---|---|---|---|
| choice, hypothesis w40 | **12/12** | **0.594** | 12 | 1,406 | 1 |
| buckets, hypothesis w40 | 10/12 | 0.543 | 10 | 1,520 | 1 |
| refine, 1 round m=6 | 11/12 | 0.563 | **62** | 3,348 | 3 |

When the alphabet fits in one question, bucketing can only add error, so plain
`choice` wins on accuracy. `refine` still resolves 5× the support. **It earns its
keep on alphabets that do not fit.**

### The censoring fix, which the whole thing depended on

First implementation took a reported `0.00` literally. A bucket whose winner rounded
to zero got `w_k = 0`, which **silently deleted every symbol in it**:

| | non-zero of 1,122 | P(correct) |
|---|---|---|
| literal zero | 14 – 25 | 0.86 |
| censored at 0.0025 | 867 – 871 | 0.67 |

Sweeping `bucket_size` 16 → 254 (71 → 5 buckets) barely moved the literal-zero
figure (16, 21, 24, 25), so it was not a bucket-count problem: *any* winners question
rounds weak buckets away. Jev's 0.01 rounding means a reported `0.00` is a **censored
observation** — "somewhere below 0.005" — and using the interval's midpoint keeps
those buckets alive.

This is the §10 objection I raised, retracted, and then hit in practice. It only
shows up once the head is concentrated: on `ascii` the mass is spread widely enough
that most buckets survive regardless.

Note the trade it exposes — concentration against coverage. Literal zeros gave
`P(correct) = 0.86` by deleting everything uncertain; censoring spreads the mass back
out and drops it to 0.67. The second is the honest distribution.

Generation is comparable or slightly better:

```
where do fish live?   buckets → " water."          refine → " They live in water"
what do bees make?    buckets → " honey. none"     refine → " honey."
```

**Not the default.** `buckets` stays default on latency: 0.3s against 1s per symbol
matters in a chat. `refine` is the mode to reach for when the distribution itself is
the output.

### At the full 50k

`refine` asks one question over *every* bucket's winner, so the bucket count must
fit inside a single question: `n_buckets <= 254`, which forces
`bucket_size >= 197` on `bpe50k`. The default 127 gives 393 buckets and fails
outright — the reason the resolution layer in §14 exists.

One symbol of `bpe50k`, `bucket_size 254` (197 buckets):

| presentation | s/symbol | requests | input tokens | non-zero of 49,862 |
|---|---|---|---|---|
| `symbol` | **6.2** | 11 | 338,039 | 5,259 |
| `hypothesis` | 11.1 | 19 | 599,452 | 5,478 |

~$0.014 per emitted token. A four-token reply took 21.5s and produced **"Paris."**,
with a sane distribution behind it — `' Paris'` 0.164, `'Paris'` 0.094, then `'P'`
and `' P'` trailing, which is the tokenizer correctly hedging between the
space-prefixed and bare forms.

It resolves 5,259 of 49,862 symbols (10.5%) — better than any single question could,
but far short of the 77% `refine` reaches on `words1k`, because 197 buckets of 254
options each still have a 0.01 floor *within* each bucket.

**The refinement round degenerates here.** The pool is `n_buckets × m` and must also
fit one question, so `m` clamps to 1 and the pool becomes exactly the winners set —
the round re-asks the same question. It still averages noise, but it is not the
up-down-up refinement of §11. Breaking that needs the hierarchy from §10, which is
the one place a third level would earn its keep rather than compounding error.

---

## 13. A bigram alphabet

*Every two-character ASCII string, so each step commits two characters.*

`tools/make_ngram_alphabet.py` builds 87² = 7,569 pairs over the single characters
of `ascii`, plus STOP.

**Mixing lengths does not work.** The first build included the 87 unigrams as well,
for parity. Under hypothesis options the unigram always won:

```
after "P":   'a' 0.115   vs   'ar' 0.083
after "Pa":  's' 0.145   vs   'r' 0.090,  'ri' 0.062     -> "Pas"
```

A shorter option is a valid prefix of strictly more continuations, so it is never
less plausible. The alphabet degenerated to unigrams while still paying 7,657
options of dilution. `--min-n 2 --n 2` makes every option the same length and
removes the bias.

**Measured** against the next *two* characters of a known continuation, 6 cases:

| alphabet | strategy | options | top-1 | P(correct) | ranks |
|---|---|---|---|---|---|
| `ascii` (one char) | choice | 89 | 3/6 | 0.432 | 2,1,1,2,1,2 |
| `bigrams` | refine | 7,570 | 2/6 | 0.257 | 7,1,2,3,1,15 |
| `bigrams` | buckets | 7,570 | 2/6 | 0.022 | 4,4,1,3,1,79 |

Two readings, and they point opposite ways:

* **Per step, `ascii` is better** — 0.432 against 0.257, and every rank within 2.
* **Per character, bigrams is not obviously worse.** `ascii` needs two steps to
  cover what one bigram covers, so its two-character accuracy is roughly 0.43² ≈
  0.19 against the bigram's 0.257 in one step.

Cost settles it for now: bigrams needs `refine` (3 requests, ~1.5 s/step) against
`ascii`'s single request at ~0.3 s, so **2.5× more per character**, and generation
is visibly worse (`"P\nC\nA\n"` against `ascii`'s `"Paris."`).

**`refine` is what makes it usable at all.** On the same alphabet, `buckets` scores
P(correct) = 0.022 against `refine`'s 0.257 — **12×**. The larger the vocabulary,
the more the OTHER-stitched weights cost you.

The obvious next step is pruning: most of the 7,569 pairs are nonsense (`'qz'`,
`'P\n'`, `'7%'`) and collectively soak up mass. A vocabulary of plausible pairs
only would cut the dilution that this measurement is dominated by.

---

## 14. Settings resolved from the alphabet and strategy

Defaults were a single flat set, so several combinations needed manual flags and
one failed outright — `-a bpe50k -s refine` errored, since 393 buckets cannot be
probed by one 255-option question.

Now an alphabet can declare preferences in its JSON, and `Config.resolve()` applies
them to anything the user has not named. Precedence: **command line > `jevchat.toml`
> the alphabet's preferences > what the strategy requires > plain defaults.**

| alphabet | resolves to |
|---|---|
| `ascii`, `lower26`, `bigrams` | `repetition_penalty = 1.0` (double letters) |
| `bpe2k`, `bpe5k`, `bpe50k` | `presentation = symbol` (same score, 43% cheaper) |
| any, with `strategy = refine` | `bucket_size` raised until the probe fits one question |

```
bpe50k   49,862 options -> bucket_size 197 (254 buckets)  symbol      rep_pen 1.1
bigrams   7,570 options -> bucket_size 127 ( 61 buckets)  hypothesis  rep_pen 1.0
words1k   1,122 options -> bucket_size 127 (  9 buckets)  hypothesis  rep_pen 1.1
```

---

## 15. Clustering the buckets

*Does it matter which symbols share a bucket?*

`buckets.build` cut the alphabet into contiguous slices of its own order, which for
`bpe50k` is BPE frequency order — arbitrary with respect to spelling. The idea:
group look-alikes, so a bucket question makes a *fine* discrimination among
competitors while the winners question makes an *easy* one among dissimilar
representatives.

**`bigrams`** (7,570 options, prefix-clustered by construction), next two characters,
8 cases:

| bucket order | top-1 | P(correct) | ranks |
|---|---|---|---|
| clustered | 5/8 | 0.226 | 4,1,1,1,1,22,1,3 |
| clustered, repeat | 5/8 | 0.245 | 3,1,1,1,1,15,2,1 |
| shuffled, seed 0 | 2/8 | 0.157 | 11,1,2,6,1,208,13,2 |
| shuffled, seed 1 | 3/8 | 0.167 | 5,1,1,9,1,133,6,5 |

Within-condition variance is small and between-condition is large, so the effect is
real: **~45% more probability on the right symbol**, and the shuffled runs throw out
ranks of 208 and 133 where clustering keeps everything under 22.

**`bpe50k`** (49,862 options), gold = any token prefixing the continuation, 6 cases:

| bucket order | top-1 | P(gold) | ranks |
|---|---|---|---|
| frequency (the alphabet's own) | 2/6 | 0.098 | 2,1,4,2,1,2 |
| **lexicographic** | **5/6** | 0.120 | **1,1,1,1,1,2** |

**2/6 → 5/6 from sorting the vocabulary before bucketing.** One config change.

**`words1k`** showed **no effect** — sorted and shuffled both scored P(correct) 0.705.

That contrast is the finding. Clustering helps only when the sort order groups
options that genuinely compete for the same slot. Lexicographic order does that for
character n-grams and subword pieces, where a shared prefix means a shared
constraint. It does nothing for whole words, where alphabetical adjacency is
semantically arbitrary — `bear` and `beautiful` are neighbours in sort order and
unrelated in use.

**Shipped:** the subword and n-gram alphabets declare `bucket_order = "sorted"`;
`words1k` keeps its own order. Getting the same benefit on a word alphabet would
need semantic clustering — which is what GPT-2's embedding matrix was suggested for
in the open questions.

---

## 16. Beam search (implemented, does not help)

*Keep several candidate replies alive instead of committing symbol by symbol.*

Every other strategy commits to one symbol per step, and since each Jev call is
stateless a bad symbol is never repaired. Beam search keeps the `beam_width` best
partial replies, ranked by mean log probability, so a prefix that looked fine at
step 2 can be abandoned at step 5.

**First measurement of actual replies.** Every other number in this log is
single-step accuracy given a human-written prefix. Twelve questions with checkable
one-word answers, scored on whether the finished reply contains the answer:

| beam width | correct | s/reply | tokens/reply | requests/reply |
|---|---|---|---|---|
| 1 | 11/12 | 1.2 | 43,371 | 4 |
| 3 | 11/12 | 6.1 | 300,545 | 19 |
| 5 | 10/12 | 16.9 | 893,829 | 51 |

**No improvement, at 5× the time and 7× the tokens for width 3.** Width 5 is worse.

The reason is structural: beam search attacks error *compounding*, and these replies
are two to four symbols long — there is nothing to compound. It would matter for
long-form prose, which §"What to expect" establishes is out of reach anyway. The
remaining failures are not search failures either: width 1's miss was
`"They need water."` for *do people need water?*, which is a correct answer the
substring metric rejects.

That flaw is worth stating plainly — **the metric undercounts**, so real accuracy is
above 11/12 and the differences between widths are inside its noise.

### The bug this nearly hid

The first implementation ran each step's distribution through the whole sampling
pipeline, including `temperature = 0.0`, which collapses a distribution to a single
symbol. With one candidate per beam there is nothing to branch on, so widths 3 and 5
used **exactly the same 4 requests as width 1** and produced near-identical replies.

It surfaced from the request counts, not the output — the replies looked plausible
either way, and the unit tests missed it because they set `temperature = 1.0`
explicitly. Beam expansion now applies only the semantic guards (`stop_bias`,
`min_steps`, `no_repeat_space`, repetition penalty) and skips
`temperature`/`top_p`/`top_k`, which are sampling controls. Two regression tests
cover it.

**Shipped but off.** `beam_width` defaults to 1.

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
| `refine` available, not default | 2× P(correct) and 19× support, at 4× the requests |
| rounded `0.00` treated as censored | taking it literally deleted whole buckets |
| alphabets declare their own defaults | one flat set needed manual flags, and `refine` + `bpe50k` failed outright |
| fixed-length n-grams, never mixed | a shorter option is always a valid prefix of more continuations |
| `bucket_order = "sorted"` for subword/n-gram alphabets | 2/6 → 5/6 top-1 on `bpe50k`; no effect on words |
| `beam_width = 1` (beam available, off) | no accuracy gain, 5× the time; replies are too short to compound error |

## Open questions

* **Validating the tail.** No reference resolves deeper than a single question, so
  every claim about low-probability symbols on this page is unverified. A
  reproducibility check (same method, different bucket assignment, do the same tail
  symbols recur?) is the cheapest way in.
* **More replicates.** Five contexts against a 0.9-nat floor could not separate
  representatives from OTHER. Anything claiming a TV difference below ~0.08 needs
  more.
* **A better end-to-end metric.** §16's substring check rejects correct paraphrases
  (`"They need water."` for *do people need water?*), so it undercounts and cannot
  resolve small differences. Everything else in this log is single-step.
* **Simulating the full distribution** rather than truncating to a shortlist. The
  pairwise/Bradley–Terry route is the only one measured that assigns finite mass to
  everything it touches, but it converges to a flatter distribution than bulk
  scoring and the temperature correction does not close the gap. Nucleus
  reselection (§11) is the most promising measured route.
* **Batching trie levels** into one request per level (§9) — untested, worth ~10×.
* **Semantic bucketing.** §15 showed clustering buckets by *spelling* is worth
  2/6 → 5/6 on `bpe50k` but nothing on a word alphabet, where alphabetical order is
  semantically arbitrary. GPT-2's own token embedding matrix (50257 × 768) would
  give a free similarity metric for clustering `words1k`-style vocabularies the same
  way.
* **Parallelising bucket requests.** They are independent; `buckets.ask` sends them
  sequentially. Worth ~3× on the large alphabets.
* **The noise floor itself.** Everything above is bounded by Jev disagreeing with
  itself at 0.90 nats. Repeated measurement under re-randomised conditions is the
  one provable way to reduce it; `shuffle_criteria` and `ensemble` are the existing
  levers.
