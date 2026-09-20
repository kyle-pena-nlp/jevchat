# jevchat

We know [Jev](https://docs.typesafe.ai/api).

`jevchat` turns that into a chat model. At every step it asks Jev one question:

> Given the user's question and the reply written so far, which symbol comes next?

The options are an alphabet plus an option to stop emitting. Jev returns a probability for each
one, and the sampler draws the next symbol from that normalised distribution.
Append, repeat, and stop when STOP is drawn.

There are several alphabets and sampling strategies available.  

The idea is for fun, the cost is somewhat impractical, and the results are hilarious.

<img width="450" height="403" alt="image" src="https://github.com/user-attachments/assets/8be3a47a-da3c-4df6-a617-d485c61a7db7" />


This was a Claude accelerated experiment.  I described the sampling algorithms, strategies, and so on, and it implemented them.

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

# refine — buckets, then a question over the winners, then a rescored nucleus.
# Twice the probability on the right symbol and ~19x the vocabulary resolved.
poetry run jevchat -a words1k -s refine ask "where do fish live?"
poetry run jevchat -a words1k -s refine --refine-nucleus 6 --refine-rounds 2 ask "…"
```

### Presentations

```bash
# hypothesis — options are the resulting texts (the default)
poetry run jevchat -p hypothesis --window 40 ask "what colour is snow?"

# symbol — options are the bare symbols, as the first version of this did
poetry run jevchat -p symbol ask "what colour is snow?"
```

### Beam search

```bash
# keep 3 candidate replies alive instead of committing symbol by symbol
poetry run jevchat -b 3 ask "what is the opposite of hot?"
```

Costs one score per live beam per step. Above width 1, `temperature`, `top_p` and
`top_k` stop applying — beams are ranked by probability, not drawn from.

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
  refine.py      bucket, probe the winners, project back down, rescore the nucleus
  present.py     symbol vs hypothesis options, and folding answers back to keys
  beam.py        keeping several candidate replies alive instead of committing
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
  make_bpe_alphabet.py   -> alphabets/bpe2k.json, bpe5k.json, bpe50k.json (network)
  make_ngram_alphabet.py -> alphabets/bigrams.json
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

158 tests, all offline — a scripted fake client for the generation loop and an
`httpx.MockTransport` for the HTTP layer. No API key and no network needed.
`jevchat bench` is the part that does hit the API.
