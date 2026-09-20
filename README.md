# jevchat

We know [Jev](https://docs.typesafe.ai/api).

`jevchat` turns that into a chat model. At every step it asks Jev one question:

> Given the user's question and the reply written so far, which symbol comes next?

The options are an alphabet plus an option to stop emitting. Jev returns a probability for each
one, and the sampler draws the next symbol from that normalised distribution.
Append, repeat, and stop when STOP is drawn.

There are several alphabets (including truncated token lists) and sampling strategies available.  

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

## Tests

```bash
poetry run pytest
```

158 tests, all offline — a scripted fake client for the generation loop and an
`httpx.MockTransport` for the HTTP layer. No API key and no network needed.
`jevchat bench` is the part that does hit the API.
