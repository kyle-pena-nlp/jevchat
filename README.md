# jevchat

[Jev](https://docs.typesafe.ai/api) is a *System One* model: it does not write text.
You hand it a state and a typed question, and it hands back a calibrated decision —
a choice, a score, or a yes/no probability.

`jevchat` turns that into a chat model. At every step it asks Jev one question:

> Given the user's question and the reply written so far, which symbol comes next?

The options are an alphabet plus a way to stop. Jev returns a probability for each
one, and the sampler draws the next symbol from that normalised distribution.
Append, repeat, and stop when STOP is drawn. The decision model becomes a language
model, one symbol at a time.

By default the options are not the symbols but the **texts they would produce**, so
Jev ranks finished strings rather than appending in its head — that one change
roughly triples character-level accuracy. The reply left unchanged is how it says
"finished".

```
state = { task, question, conversation, answer_so_far }   ← the whole reply
          │
          ▼
    one request, many parallel questions — the alphabet sliced into buckets,
    each option showing the reply's tail with one more symbol on the end
          │
          ▼
    '…France is Para' 0.02   '…France is Pari' 0.70   '…France is Par' 0.21 (STOP)
          │
          ▼
    fold back to symbols → bias → ban → temperature → top-k → top-p → draw
          │
          ▼
    append, and repeat
```

## Setup

```bash
poetry install
```

Put your Jev key in `.env` next to `pyproject.toml` (git-ignored):

```
api_key="..."
```

`JEV_API_KEY` and `TYPESAFE_API_KEY` are also accepted. Values in `.env` win over
exported ones, so editing the file is enough to switch keys.

## Use

```bash
poetry run jevchat                          # interactive chat
poetry run jevchat ask "do people need water?"
poetry run jevchat alphabets                # what you can sample from
poetry run jevchat bench                    # compare every mode (table below)
```

The reply appears as it is sampled, in a panel with a live readout of the
generation rate — symbols/s, characters/s, milliseconds per API call, elapsed
time — and the top few symbols Jev scored at the last step, so you can watch the
distribution the sampler is drawing from.

**Ctrl-C cancels.** The first press stops generation once the in-flight request
returns and keeps the partial reply; a second press aborts immediately. In chat,
the partial reply stays in the conversation history. `ask` exits 130 when cancelled.

Chat commands: `/help`, `/alphabet [name]`, `/temp <v>`, `/stop-bias <v>`,
`/reset`, `/stats`, `/exit`.

## Modes

Two things are swappable: **how** the distribution over the next symbol is
obtained (`-s/--strategy`), and **what** it is over (`-a/--alphabet`). Every
combination below is a runnable command.

### Strategies

`choice` asks one question over the whole alphabet. `bisect` sorts the alphabet
and asks earlier/later yes-no questions until the group is small, then asks one
choice question inside it.

```bash
# choice — one question over the whole alphabet (the default)
poetry run jevchat -s choice ask "how many eyes do people have?"

# ...without the re-ordering that cancels Jev's position bias (worst mode)
poetry run jevchat -s choice --no-shuffle-criteria ask "how many eyes do people have?"

# ...averaging 4 re-orderings, sent as 4 parallel questions in one request
poetry run jevchat -s choice --ensemble 4 ask "how many eyes do people have?"

# bisect — earlier/later down to groups of 20, each split asked both ways
poetry run jevchat -s bisect ask "how many eyes do people have?"

# ...cheaper: bigger groups, each split asked once
poetry run jevchat -s bisect --bisect-cutoff 32 --no-bisect-swap ask "how many eyes do people have?"

# buckets — the alphabet split across many questions, each with an OTHER escape.
# The only strategy that can hold more than 255 symbols.
poetry run jevchat -a words1k -s buckets ask "what colour is snow?"
poetry run jevchat -a bpe5k -s buckets --bucket-size 127 ask "what is the capital of france?"
```

### Presentations

```bash
# hypothesis — options are the resulting texts (the default)
poetry run jevchat -p hypothesis --window 40 ask "what colour is snow?"

# symbol — options are the bare symbols, as the first version of this did
poetry run jevchat -p symbol ask "what colour is snow?"
```

### Alphabets

```bash
poetry run jevchat -a lower26 -t 0 ask "what is 2+2?"   # a-z and space only
poetry run jevchat -a ascii   -t 0 ask "what is 2+2?"   # spells anything
poetry run jevchat -a tokens  -t 0 ask "do people need water?"   # whole words

# these three exceed 255 options, so they need --strategy buckets
poetry run jevchat -a words1k -s buckets -t 0 ask "what colour is grass?"
poetry run jevchat -a bpe2k   -s buckets -t 0 ask "where do fish live?"
poetry run jevchat -a bpe5k   -s buckets -t 0 ask "what do bees make?"
```

### Combining them

```bash
poetry run jevchat -a tokens -s bisect --bisect-cutoff 20 ask "do people need water?"
poetry run jevchat -a ascii -s choice --ensemble 12 -t 0.2 --repetition-penalty 1.0 \
    ask "what colour is grass?"
```

### How they compare

`jevchat bench` scores the same twelve hand-checked continuations in every mode.
Each case is one step of real generation through the same code path the loop
uses, so the table describes the modes as they actually run.

```bash
poetry run jevchat -a ascii bench                      # the table below
poetry run jevchat -a words1k bench --cases word       # word-level cases
poetry run jevchat bench --mode bisect                 # just the bisect rows
```

`--cases word` swaps in next-*word* continuations, which is the fair test for the
word and subword alphabets. A case whose answer an alphabet cannot express is
skipped and reported rather than scored, so the numbers measure ranking rather
than vocabulary coverage.

```
                        12 char continuations · alphabet ascii (89 options)
 mode                         top-1  top-3  P(correct)  non-zero  ms/step  in-tok  questions
 choice, symbol, no shuffle    3/12   3/12       0.181        14      208    1893          1
 choice, symbol                2/12   6/12       0.249        18      175    1893          1
 choice, symbol, ensemble 4    2/12   9/12       0.223        24      213    6492          4
 choice, hypothesis w24       10/12  11/12       0.526        11      194    1420          1
 choice, hypothesis w40        8/12  10/12       0.554        10      185    1406          1
 bisect, cutoff 20             3/12   7/12       0.128        70      192    5915         23
 buckets, symbol               3/12   8/12       0.227        22      172    1106          2
 buckets, hypothesis w40      10/12  11/12       0.555        11      185    1520          2
```

Twelve cases is a small sample, so read the gaps between neighbouring rows as
noise. The large ones reproduce:

* **Hypothesis options are the biggest lever in the project** — three to five times
  the top-1 of symbol options, double the mass on the right symbol, for *fewer*
  input tokens. See [Hypothesis options](#hypothesis-options).
* **Shuffling the option order is free and worth it.** `no shuffle` → `symbol`
  moves top-3 from 3/12 to 6/12 at identical cost.
* **`bisect` is worse on accuracy and better on coverage.** It puts probability on
  ~70 symbols where a single choice call reaches 18, but P(correct) roughly halves.
  Each split comes back near a coin flip (0.45–0.65), and multiplying three or four
  of those dilutes the signal faster than the finer leaf resolution recovers it.
  It is the interesting mode, not the good one.

`ensemble` buys support, not accuracy: 18 → 24 non-zero symbols going 1 → 4, at
3.4× the input tokens. It defaults to 1 for that reason.



## Hypothesis options

There are two ways to ask Jev the same question. Under `--presentation symbol` the
options are the symbols themselves — `'a'`, `'i'`, `' the'` — and Jev has to append
the option to the reply in its head before judging it. The instructions used to say
exactly that: *"judge grammar and spelling on the concatenation, not on the option
on its own."*

Under `--presentation hypothesis` the options are the **resulting texts**:

```
answer_so_far = "The capital of France is Par"

symbol      options:  'a'  'i'  's'  …  STOP
hypothesis  options:  '…he capital of France is Para'
                      '…he capital of France is Pari'
                      '…he capital of France is Pars'
                      '…he capital of France is Par'     <- unchanged: this is STOP
```

The append is already done, so Jev only ranks finished strings — which is what a
decision model is built for. It is the single largest improvement in the project:
on character alphabets it roughly triples top-1 and doubles the probability mass
landing on the right symbol, for fewer input tokens than symbol options with their
per-option descriptions.

**STOP is the reply with nothing added.** Choosing it means "this is already the
best answer", so termination needs no special token and no extra question.

Three things to know before relying on it:

* **It is neutral at word level.** On `words1k` word continuations, symbol and
  hypothesis both score 9/10 (P(correct) 0.318 vs 0.320). The win is concentrated
  where a symbol alone carries no meaning: a bare `'i'` is unjudgeable, `' blue'`
  is not.
* **It does not help at the first step.** With an empty reply there is no prefix,
  so a hypothesis option is just the symbol again. The loop's worst moments are its
  first ones, and this does not fix them.
* **The unchanged option attracts spurious mass**, because it is a valid prefix of
  every extension of itself. Mean P(stop) on *unfinished* replies is 0.08 under
  hypothesis against 0.00 under symbol. `stop_bias 0.5` and `min_steps 3` are the
  defaults that compensate; without them generation stops after two or three
  characters.

### The window

Options show the reply's **tail**, not all of it, so their cost stays flat as the
reply grows — the whole reply still reaches Jev once, in `state.answer_so_far`.
Without a window, options carry the full prefix and cost grows with length:

| prefix | 255 options | 1,122 options (`words1k`) |
|---|---|---|
| 50 chars | ~3,600 tok | ~16,000 tok |
| 200 chars | ~13,000 tok | ~57,000 tok — near the measured request ceiling |

`window = 40` is the default. 24 and 40 scored the same within noise.

## Past the 255 cap

255 is a limit on a **question**, not on a vocabulary. `buckets` slices the
alphabet into questions of at most 254 options and gives each one an extra OTHER
option meaning "the symbol that comes next is not in this bucket".

OTHER is the part that matters. Jev normalises within a question, so N separate
slices would otherwise give N unrelated conditional distributions with no way to
weigh them against each other. With OTHER, every question reports how much of its
mass belongs to its own members versus everything else, and what is left after
dropping OTHER concatenates directly.

Because parallel questions are near-free, a whole vocabulary still costs about one
round trip. Measured on ten next-word continuations, scoring only the cases each
alphabet can express:

| alphabet | options | expressible | top-1 | tokens/step | ms/step | questions | requests |
|---|---|---|---|---|---|---|---|
| `tokens` | 255 | 4/10 | 3/4 | 2,262 | 230 | 3 | 1 |
| `words1k` | 1,122 | 10/10 | **9/10** | 8,593 | 263 | 10 | 1 |
| `bpe2k` | 2,433 | 6/10 | 5/6 | 18,388 | 370 | 21 | 1 |
| `bpe5k` | 5,228 | 9/10 | 7/9 | 38,949 | 558 | 43 | 1 |

**5,228 options resolve in a single request in 558 ms** — twenty times what one
question holds, at about twice the latency. With hypothesis options the same
alphabet costs 55,669 input tokens, still inside one request.

Three things this measurement says:

* **Coverage is the differentiator, not ranking.** Top-1 on expressible cases sits
  between 75% and 90% for every alphabet. What separates them is how often the word
  you need is on the list at all. `tokens` could express four of the ten targets.
* **BPE frequency order is not chat coverage.** `bpe2k` has twice the options of
  `words1k` and expresses fewer of the targets: BPE ranks by compression value, so
  fragments like `'ning'` and `'ced'` outrank whole words. `' blue'` is token 4171,
  `' cold'` 4692, `' honey'` 12498. A curated thousand-word list beats a 2,433-piece
  BPE prefix, which is why `words1k` is the one to reach for.
* **Bigger buckets won.** `bucket_size 127` was both the most accurate and the
  cheapest, because every bucket pays for its own copy of the OTHER description.

### words1k or bpe5k

`bpe5k` writes the more fluent sentences — real subword pieces let it assemble
words `words1k` has no entry for:

```
what is the capital of france?   words1k → " The capital isP city is P is a city…"
                                 bpe5k   → " The capital is Parys"
what do bees make?               bpe5k   → " They make a lot of food"
```

But it is slower (~700 ms/step against ~260), costs about five times the tokens,
and spreads its probability so thin that `P(correct)` drops to 0.07 against 0.32 —
5,228 options is a lot of places for mass to go. On the benchmark it scores 7/9
where `words1k` scores 9/10, and it still cannot express `' honey'`, because BPE
orders by compression value rather than by usefulness in a chat.

So `words1k` is the default and `bpe5k` is the one to reach for when you want
sentences rather than single-word answers. Shrinking the window to save tokens does
not work — at `--window 16` the same question gives `" the C Paree"`.

### Why not rejection sampling

The obvious way to use OTHER is accept–reject: propose one bucket, accept with
probability `1 − P(OTHER)`, redraw otherwise. It would be exact if `P(OTHER)` were
calibrated. It is not. Summed across buckets it should equal `K − 1`; measured, it
reaches only 68% of that at `K=4`, improving to 89% at `K=32` as the buckets shrink.
Jev under-reports OTHER — it would rather commit to a listed option than admit the
answer is absent — so accept–reject would over-accept from cold buckets and
reintroduce the very bias it was meant to remove.

Batching sidesteps the problem entirely. Asking every bucket in one request costs
about one round trip, so there is nothing to reject: concatenate and normalise.
Weighting each bucket by `1 − P(OTHER)` instead scored identically, so the simpler
form is what ships.

What does survive is the *ranking*: across every K from 4 to 32, the bucket holding
the answer had the lowest `P(OTHER)` in 10 out of 10 cases.

### Building a vocabulary

```bash
python tools/make_word_alphabet.py                        # words1k.json
python tools/make_bpe_alphabet.py --limit 5400 --out bpe5k.json
```

`tools/make_bpe_alphabet.py` pulls GPT-2's real BPE vocabulary and keeps the
lowest-numbered tokens; pass `--limit 50257` for all of it. GPT-2 gives ids 0–255 to single bytes and then numbers
merged pieces in the order BPE added them, which is by descending frequency — so
the low ids are a byte-level fallback plus the commonest English word pieces, and
no frequency corpus is needed. Output is committed, so jevchat itself never needs
the script or a network.

## What to expect

This is a decision model doing a sequence model's job, and it shows. Each call is
independent and stateless, so Jev has no plan it is carrying forward — it only sees
the prefix. Short factual answers come out right. On defaults:

```
what is 2+2?            →   four.        what do bees make?    →   honey.
what colour is snow?    →   white.       do people need water? →   Yes.
where do fish live?     →   They live in water
```

Out-of-vocabulary answers have to be spelled, and that is where the alphabet
matters more than anything else. `words1k` has no proper nouns, so it flails at
`what is the capital of france?`; the character alphabet gets it:

```
poetry run jevchat -a ascii -s choice -t 0 --repetition-penalty 1.0 \
    ask "what is the capital of france?"      →   Paris.
```

Longer prose degrades. Once the prefix is garbled, every later step is conditioned
on garbage and it does not recover. Character-level spelling is the weak point:
Jev continues `"Yes, the sky is bl"` → `"u"` at 0.48, but cannot reliably spell a
word it has not been handed most of.

Per-step accuracy is the ceiling on all of this: about 10/12 in the best mode
above, and roughly half that without hypothesis options. Roughly 50% per character is exactly why long replies fall apart: at that rate a
twenty-character answer has essentially no chance of surviving intact, and errors
never get repaired because each call only sees the prefix. Short answers need two
or three correct decisions, so they land.

Throughput is ~3–5 symbols/s at 200–350 ms per step, and **latency is flat in
alphabet size** — 28 options and 255 options cost the same. That is the whole
argument for the token alphabet.

### Position bias

Jev systematically favours whichever options appear **first** in the criteria map.
Asked with an empty question and an empty prefix, 80% of its mass over `ascii`
lands on `STOP` and `A` — the first entries. With the alphabet in natural `a, b,
c…` order the correct continuation of `"The capital of France is Par"` ranks 6th;
re-order the same map and it ranks 1st.

So `shuffle_criteria` re-orders the alphabet on every call. It costs nothing and
is the single largest quality lever here.

`ensemble` goes further: it scores N different orderings and averages them. Jev
answers every question in a request in parallel and in isolation, and request
latency is close to flat in the number of questions — 1 question takes 0.63 s,
32 take 0.95 s — so N orderings cost about one call's wall time. Input tokens do
scale with N, which is the real price (at `ensemble = 4` over `ascii`, ~6.5k input
tokens per step against 1.9k).

What averaging buys is the tail that rounding throws away (below), not accuracy —
see the table in [Modes](#how-they-compare). It defaults to 1.

### Quantisation: the tail is rounded off

Jev rounds `probabilities` to two decimals, so anything it scores below 0.005
comes back as exactly `0.0` and is indistinguishable from impossible. In a single
call the surviving support is about the same size no matter how big the alphabet
is:

| alphabet | options | non-zero in one call |
|---|---|---|
| `lower26` | 28 | 17 |
| `ascii` | 89 | 25 |
| `tokens` | 255 | 25 |

So 230 of the token alphabet's 255 options carry no information on any given
call. Different orderings surface different tails, and averaging N of them gives
an effective granularity of 0.01/N:

| `ensemble` | non-zero of 255 | smallest non-zero |
|---|---|---|
| 1 | 25 | 0.0100 |
| 2 | 36 | 0.0050 |
| 4 | 52 | 0.0025 |
| 8 | 66 | 0.0013 |

That is the whole argument for the ensemble: it is the only way to see the
low-probability symbols at all. It is also why `top_p` is set below 1 — the
surviving tail is coarse enough to be worth trimming. `bisect` attacks the same
problem from the other side, by never asking a question big enough to get rounded
flat: it reaches ~69 non-zero symbols in one request.

`no_repeat_space` and `repetition_penalty` exist for the same bias; without them
generation stalls on a run of spaces or one repeated symbol.

## Alphabets

The alphabet is swappable — it is just JSON, and it is the main lever on both
quality and speed. Jev allows at most **255 choices** per question, which is the ceiling on an
alphabet's size.

| name | options | emits | strategy |
|---|---|---|---|
| `lower26` | 28 | `a`–`z`, space | any |
| `ascii` | 89 | letters, digits, punctuation, line breaks | any |
| `tokens` | 255 | 168 whole words, suffixes, bare letters, punctuation | any |
| `words1k` | 1,122 | 1,009 common words, plus letters/digits/suffixes to spell the rest | `buckets` |
| `bpe2k` | 2,433 | GPT-2's most frequent BPE pieces, plus single bytes | `buckets` |
| `bpe5k` | 5,228 | the same, twice as far down the frequency order | `buckets` |
| `bpe50k` | 49,862 | **all** of GPT-2's BPE vocabulary | `buckets` |

The first three fit in one question. The last three do not, so they need
`--strategy buckets`; `--strategy choice` refuses them with a message saying so.

`lower26` is the literal version of the idea: a score over the letters of the
alphabet, so everything arrives lowercase and unpunctuated. `ascii` is the
general-purpose one — it can spell anything. `tokens` is ~5× fewer calls and far
more grammatical, because choosing between whole words is the kind of decision Jev
is actually good at; the catch is that its word list is fixed and small, so it
cannot say "Paris" except by spelling it out of the bare letters it keeps as a
fallback. Pick `tokens` for speed and fluency, `ascii` for coverage.

The character alphabets want `--repetition-penalty 1.0`: penalising a recently
drawn letter also blocks legitimate double letters like the `ee` in "green".

### Writing your own

Any `.json` file works; pass a path to `--alphabet`, or drop it in `./alphabets/`
and pass the bare name.

```json
{
  "name": "vowels",
  "description": "vowels and a space",
  "stop": { "key": "STOP", "description": "The reply is complete." },
  "symbols": [
    "a",
    "e",
    { "key": "SPACE", "emit": " ", "description": "a single space" }
  ]
}
```

`key` is what Jev sees and returns, `emit` is what gets appended to the reply
(defaults to `key`), and `description` is the criteria text Jev reads. A plain
string is shorthand for all three. Descriptions are worth writing carefully —
they are the only thing telling Jev what an option means.

## Configuration

`jevchat.toml` holds the defaults, and is searched for in the working directory
and its parents. Every CLI flag overrides it.

| option | default | meaning |
|---|---|---|
| `alphabet` | `words1k` | builtin name, bare name in `./alphabets/`, or path |
| `model` | `jev-latest` | Jev model id |
| `strategy` | `buckets` | `choice`, `bisect`, or `buckets` (the one that holds >255 options) |
| `presentation` | `hypothesis` | `hypothesis` (options are resulting texts) or `symbol` |
| `window` | `40` | characters of the reply's tail shown in each hypothesis option |
| `bucket_size` | `127` | symbols per bucket question; at most 254, leaving room for OTHER |
| `bucket_batch` | `48` | most bucket questions per request; backs off on overflow |
| `bucket_describe` | `false` | per-option descriptions: ~2.3× the tokens, no accuracy gain |
| `bisect_cutoff` | `20` | stop bisecting once a group is this small |
| `bisect_swap` | `true` | ask each split both ways to cancel Jev's lean towards yes |
| `shuffle_criteria` | `true` | re-order the alphabet each call to cancel Jev's position bias |
| `ensemble` | `1` | orderings scored per step, as parallel questions in one request |
| `temperature` | `0.4` | `0` is greedy; applied as `p^(1/T)` |
| `top_p` / `top_k` | `0.9` / `0` | nucleus and top-k filters |
| `stop_bias` | `0.5` | multiplier on `STOP`: `<1` longer, `>1` shorter |
| `repetition_penalty` | `1.1` | divides the weight of symbols drawn in the last `repetition_window` steps |
| `repetition_window` | `8` | how far back the penalty looks |
| `no_repeat_space` | `true` | forbid a symbol opening with whitespace right after whitespace |
| `min_steps` | `3` | symbols drawn before `STOP` is allowed |
| `max_steps` / `max_chars` | `400` / `1500` | hard caps |
| `seed` | unset | makes a reply reproducible |
| `history_turns` | `6` | prior turns sent as context |
| `live` / `show_dist` | `true` / `3` | display mode, and how many scores to show |
| `timeout` / `max_retries` | `60.0` / `4` | transport; retries 429/5xx with backoff |

## Why `choice` and not `score`

Jev's `score` type looks like the obvious fit for "a score over the letters", but
it cannot carry an alphabet: it is capped at **10 levels**, and 26 letters is
rejected outright with `400 Too many score levels. Must have at most 10 levels.`
It is also ordinal — it reads its criteria as a ranked scale, which letters are
not.

Its `score` field is not a probability either; it is the expected index under the
distribution. Asking it a 3-level question returns `probabilities` `{0: 0.70,
1: 0.04, 2: 0.26}` and `score` `0.57`, which is `0·0.70 + 1·0.04 + 2·0.26`.

`choice` is the type that fits: up to 255 unordered options, and its
`probabilities` map is natively a normalised distribution over exactly the keys
you sent — that map is what jevchat samples from. (`score` returns a `probabilities`
map over level indices too, and it is also natively normalised.)

The other ceiling is request size, which caps `ensemble` × alphabet size. Cloudflare's
model page lists a 32,000-token context, but that does not describe the native
endpoint: measured against `api.typesafe.ai`, a single request carrying **65,715
input tokens** succeeded, and the next step up (16 questions of 255 described
options, ~88,000 tokens) failed. So the practical ceiling is somewhere above 65k.
jevchat reports the failure as `Jev's context window overflowed with 16 question(s)
of 255 options. Lower `ensemble` or use a smaller alphabet.`

## How sampling works

`jevchat/sampler.py` shapes Jev's raw probabilities before drawing, in this order:

0. **fold** — under hypothesis options Jev answers in whole texts, so those are
   translated back into alphabet keys first (`jevchat/present.py`).
1. **normalise** — Jev's values are clamped and rescaled to sum to 1. They already
   sum to 1 (±0.01 from rounding), so this mostly repairs rounding and drops the
   zeroed-out tail.
2. **bias** — `stop_bias` scales `STOP`; `repetition_penalty` divides recent symbols.
3. **ban** — `STOP` while `step < min_steps`, and leading-whitespace symbols when
   the reply already ends in whitespace.
4. **temperature** — `p' ∝ p^(1/T)`; `T=0` collapses to the top symbol.
5. **top-k**, then **top-p** — truncate the tail, renormalise.
6. **draw** — inverse-CDF sample from a seedable `random.Random`.

`Draw.raw` keeps the unshaped distribution, which is what the live readout shows,
so you can see what Jev actually scored versus what the sampler did with it.

## Layout

```
jevchat/
  alphabet.py    loading and validating alphabets (255-choice ceiling)
  alphabets/     lower26.json, ascii.json, tokens.json
  client.py      POST /v1/systemone, parallel questions, retries, usage accounting
  bisect.py      the earlier/later tree, and folding its answers into one distribution
  buckets.py     slicing an alphabet across questions, each with an OTHER escape
  present.py     symbol vs hypothesis options, and folding answers back to keys
  benchmark.py   the twelve continuations behind `jevchat bench`
  config.py      defaults, jevchat.toml, .env key loading
  sampler.py     normalise, bias, ban, temperature, top-k, top-p, draw
  generate.py    Scorer (one state -> one distribution) and the loop itself
  display.py     live panel and raw-stream renderers
  interrupt.py   first Ctrl-C cancels, second aborts
  cli.py         chat / ask / bench / alphabets
tools/
  wordlist.py            the hand-built common-word list
  make_word_alphabet.py  -> alphabets/words1k.json
  make_bpe_alphabet.py   -> alphabets/bpe2k.json, bpe5k.json (needs a network)
```

## The experiment log

[`docs/experiments.md`](docs/experiments.md) records everything measured to get
here — the API's real limits, the framings and strategies that were tried, the
numbers that decided each default, and the ideas that failed (bisection, anchor
linking) with the measurements that killed them. Start with the noise floor: Jev
disagrees with *itself* at 0.90 nats, which bounds how precisely any of this can
be read.

## Tests

```bash
poetry run pytest
```

122 tests, all offline — a scripted fake client for the generation loop and an
`httpx.MockTransport` for the HTTP layer. No API key and no network needed.
`jevchat bench` is the part that does hit the API.
