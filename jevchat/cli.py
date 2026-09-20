"""Command line entry point."""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import alphabet as alphabet_mod
from .alphabet import Alphabet, AlphabetError
from .client import JevClient, JevError
from .buckets import BUCKET_ORDERS
from .config import STRATEGIES, Config, ConfigError, load_api_key
from .present import PRESENTATIONS
from .display import make_renderer
from .generate import Done, Step, Turn, generate
from .interrupt import sigint_cancel
from .sampler import SamplerError

console = Console()
err_console = Console(stderr=True)


def sampling_options(func):
    """Options shared by every command that generates text."""
    decorators = [
        click.option("-a", "--alphabet", "alphabet_ref", metavar="NAME|PATH",
                     help="Alphabet to sample from (builtin name or .json path)."),
        click.option("-t", "--temperature", type=float,
                     help="0 is greedy, higher is more random."),
        click.option("--top-p", type=float, help="Nucleus filter, in (0, 1]."),
        click.option("--top-k", type=int, help="Keep only the k best symbols (0 disables)."),
        click.option("--stop-bias", type=float,
                     help="Multiplier on STOP: <1 gives longer replies, >1 shorter."),
        click.option("--repetition-penalty", type=float,
                     help="Divide the weight of recently drawn symbols (1.0 disables). "
                          "Use 1.0 for the character alphabets."),
        click.option("--repetition-window", type=int,
                     help="How many recent symbols the repetition penalty looks at."),
        click.option("--no-repeat-space/--allow-repeat-space", "no_repeat_space",
                     default=None,
                     help="Forbid whitespace directly after whitespace."),
        click.option("--shuffle-criteria/--no-shuffle-criteria", "shuffle_criteria",
                     default=None,
                     help="Re-order the alphabet each call to cancel Jev's position bias."),
        click.option("-s", "--strategy", type=click.Choice(STRATEGIES),
                     help="choice: one question over the whole alphabet. "
                          "bisect: earlier/later questions down to a small group."),
        click.option("--bisect-cutoff", type=int, metavar="N",
                     help="Stop bisecting once a group has N or fewer symbols."),
        click.option("--bisect-swap/--no-bisect-swap", "bisect_swap", default=None,
                     help="Ask each split both ways to cancel Jev's lean towards yes."),
        click.option("-p", "--presentation", type=click.Choice(PRESENTATIONS),
                     help="symbol: options are the symbols. hypothesis: options are "
                          "the resulting texts, so Jev ranks finished strings."),
        click.option("--window", type=int, metavar="N",
                     help="Characters of the reply's tail shown in each hypothesis "
                          "option."),
        click.option("--bucket-size", type=int, metavar="N",
                     help="Symbols per bucket question (buckets strategy)."),
        click.option("--bucket-batch", type=int, metavar="N",
                     help="Most bucket questions to put in one request."),
        click.option("--bucket-order", type=click.Choice(BUCKET_ORDERS),
                     help="How to cut the alphabet up: given (its own order), "
                          "sorted (look-alikes together), or shuffled."),
        click.option("--bucket-describe/--no-bucket-describe", "bucket_describe",
                     default=None,
                     help="Send a description with every option. Costs ~2.3x the "
                          "input tokens and did not improve accuracy."),
        click.option("--refine-nucleus", type=int, metavar="N",
                     help="Symbols per bucket pooled for rescoring (refine strategy)."),
        click.option("--refine-rounds", type=int, metavar="N",
                     help="Reweighting rounds after the winners question."),
        click.option("--ensemble", type=int, metavar="N",
                     help="Score N re-orderings per step as parallel questions in one "
                          "request, and average them. Latency is near flat in N; input "
                          "tokens are not."),
        click.option("--history-turns", type=int,
                     help="Prior chat turns sent to Jev as context."),
        click.option("--max-steps", type=int, help="Hard cap on sampled symbols."),
        click.option("--max-chars", type=int, help="Hard cap on reply length."),
        click.option("--min-steps", type=int, help="Symbols before STOP is allowed."),
        click.option("-b", "--beam-width", type=int, metavar="N",
                     help="Candidate replies kept alive at once. 1 is plain "
                          "sampling; more costs one score per beam per step."),
        click.option("--beam-length-penalty", type=float,
                     help="Beams rank by mean log probability ^ this."),
        click.option("--seed", type=int, help="Seed the sampler for a reproducible reply."),
        click.option("--model", help="Jev model id (default jev-latest)."),
        click.option("--show-dist", type=int, metavar="N",
                     help="Show the top N scored symbols each step (0 hides them)."),
        click.option("--live/--no-live", "live", default=None,
                     help="Live panel, or raw streaming suitable for pipes."),
        click.option("--config", "config_path", type=click.Path(path_type=Path),
                     help="Path to jevchat.toml."),
        click.option("--env-file", type=click.Path(path_type=Path),
                     help="Path to the .env holding api_key."),
    ]
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def _build(params: dict) -> tuple[Config, Alphabet, JevClient]:
    config_path = params.pop("config_path", None)
    env_file = params.pop("env_file", None)
    alphabet_ref = params.pop("alphabet_ref", None)
    if alphabet_ref:
        params["alphabet"] = alphabet_ref

    config = Config.load(config_path=config_path, overrides=params)
    alpha = alphabet_mod.load(config.alphabet)
    config = config.resolve(alpha)
    client = JevClient(
        load_api_key(env_file=env_file),
        model=config.model,
        api_base=config.api_base,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
    return config, alpha, client


def _run_turn(
    client: JevClient,
    alpha: Alphabet,
    config: Config,
    question: str,
    history: list[Turn],
    rng: random.Random,
) -> Done:
    renderer = make_renderer(console, alpha, live=config.live, show_dist=config.show_dist)
    cancelled_note = Text("  cancelling after this call… (Ctrl-C again to abort)", style="yellow")

    with sigint_cancel(on_first=lambda: err_console.print(cancelled_note)) as cancel:
        renderer.start(question)
        result: Done | None = None
        try:
            for event in generate(
                client, alpha, config, question, history=history, cancel=cancel, rng=rng
            ):
                if isinstance(event, Step):
                    renderer.update(event)
                else:
                    result = event
                    renderer.finish(event)
        finally:
            renderer.close()
    assert result is not None
    return result


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@sampling_options
@click.pass_context
def main(ctx: click.Context, **params: object) -> None:
    """Chat with Jev by sampling its score over an alphabet, one symbol at a time.

    Run with no subcommand to open an interactive chat.
    """
    ctx.ensure_object(dict)
    ctx.obj.update(params)
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@main.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Interactive chat (the default)."""
    try:
        config, alpha, client = _build(dict(ctx.obj))
    except (ConfigError, AlphabetError) as exc:
        raise click.ClickException(str(exc)) from exc

    rng = random.Random(config.seed)
    history: list[Turn] = []

    console.print(
        Text(f"jevchat · {config.model} · alphabet {alpha.name} ({alpha.size} options) · "
             f"{config.strategy}/{config.presentation} · temp {config.temperature}",
             style="cyan")
    )
    console.print(Text("/help for commands, Ctrl-C to stop a reply, Ctrl-D to quit.\n",
                       style="dim"))

    with client:
        while True:
            try:
                question = console.input("[bold green]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not question:
                continue
            if question.startswith("/"):
                action, config, alpha = _command(question, config, alpha, history, client)
                if action == "quit":
                    break
                continue

            try:
                done = _run_turn(client, alpha, config, question, history, rng)
            except (JevError, SamplerError, AlphabetError) as exc:
                err_console.print(Text(f"error: {exc}", style="red"))
                continue

            history.append(Turn("user", question))
            if done.text:
                history.append(Turn("assistant", done.text))
            console.print()


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.pass_context
def ask(ctx: click.Context, question: tuple[str, ...]) -> None:
    """Answer one QUESTION and exit."""
    try:
        config, alpha, client = _build(dict(ctx.obj))
    except (ConfigError, AlphabetError) as exc:
        raise click.ClickException(str(exc)) from exc

    rng = random.Random(config.seed)
    with client:
        try:
            done = _run_turn(client, alpha, config, " ".join(question), [], rng)
        except (JevError, SamplerError, AlphabetError) as exc:
            raise click.ClickException(str(exc)) from exc
    if done.reason == "cancelled":
        sys.exit(130)


@main.command()
@click.option("--mode", "only", multiple=True, metavar="NAME",
              help="Run only the named mode(s); repeatable. Default: all of them.")
@click.option("--cases", "case_set", type=click.Choice(["char", "word"]), default="char",
              show_default=True,
              help="char: next-character continuations. word: next-word continuations, "
                   "for the word and subword alphabets.")
@click.pass_context
def bench(ctx: click.Context, only: tuple[str, ...], case_set: str) -> None:
    """Score the same continuations in every mode and print the comparison.

    This is the table in the README. One step of real generation per case, so it
    measures the modes exactly as they run.
    """
    from . import benchmark

    params = dict(ctx.obj)
    try:
        base, alpha, client = _build(params)
    except (ConfigError, AlphabetError) as exc:
        raise click.ClickException(str(exc)) from exc

    modes = [(name, kw) for name, kw in benchmark.MODES
             if not only or any(o.lower() in name.lower() for o in only)]
    if not modes:
        raise click.ClickException(
            f"no mode matched {', '.join(only)}; known modes are "
            + "; ".join(name for name, _ in benchmark.MODES)
        )

    cases = benchmark.CASE_SETS[case_set]
    table = Table(title=f"{len(cases)} {case_set} continuations · alphabet {alpha.name} "
                        f"({alpha.size} choices)", title_style="cyan", header_style="bold")
    table.add_column("mode")
    for column in ("top-1", "top-3", "P(correct)", "non-zero", "ms/step",
                   "in-tok/step", "questions", "reqs"):
        table.add_column(column, justify="right")

    with client:
        for name, overrides in modes:
            config = base.with_overrides(**overrides).resolve(alpha)
            try:
                result = benchmark.run_mode(client, alpha, config, name, cases)
            except (JevError, AlphabetError) as exc:
                err_console.print(Text(f"{name}: {exc}", style="red"))
                continue
            table.add_row(
                name,
                f"{result.top1}/{result.cases}",
                f"{result.top3}/{result.cases}",
                f"{result.mass:.3f}",
                f"{result.nonzero:.0f}",
                f"{result.ms:.0f}",
                f"{result.input_tokens}",
                f"{result.questions}",
                f"{result.requests}",
            )
    console.print(table)
    missing = [c.expected for c in cases
               if benchmark._key_for(alpha, c.expected) is None]
    if missing:
        console.print(Text(
            f"{len(missing)} of {len(cases)} cases skipped: {alpha.name} cannot express "
            + ", ".join(repr(m) for m in missing)
            + ". Scores are over the rest.", style="yellow"))


@main.command("alphabets")
def list_alphabets() -> None:
    """List the alphabets available to sample from."""
    table = Table(title="alphabets", title_style="cyan", header_style="bold")
    table.add_column("name")
    table.add_column("choices", justify="right")
    table.add_column("description")
    for name in alphabet_mod.builtin_names():
        alpha = alphabet_mod.load(name)
        table.add_row(alpha.name, str(alpha.size), alpha.description)
    console.print(table)
    console.print(
        Text(f"Jev allows at most {alphabet_mod.MAX_CHOICES} choices per question. "
             "Point --alphabet at your own .json to add one.", style="dim")
    )


# ----------------------------------------------------------------- /commands
HELP = """\
/help                 this message
/alphabet [name]      show or switch the alphabet
/temp <value>         set the sampling temperature
/stop-bias <value>    <1 for longer replies, >1 for shorter
/reset                forget the conversation history
/stats                token and call totals for this session
/exit                 quit
"""


def _command(
    line: str,
    config: Config,
    alpha: Alphabet,
    history: list[Turn],
    client: JevClient,
) -> tuple[str, Config, Alphabet]:
    parts = line.split()
    name, args = parts[0][1:].lower(), parts[1:]

    if name in {"exit", "quit", "q"}:
        return "quit", config, alpha
    if name == "help":
        console.print(Text(HELP, style="dim"))
    elif name == "reset":
        history.clear()
        console.print(Text("history cleared", style="dim"))
    elif name == "alphabet":
        if not args:
            console.print(Text(f"{alpha.name} ({alpha.size} choices): {alpha.description}",
                               style="dim"))
        else:
            try:
                alpha = alphabet_mod.load(args[0])
                config = replace(config, alphabet=args[0])
                console.print(Text(f"alphabet → {alpha.name} ({alpha.size} choices)",
                                   style="dim"))
            except AlphabetError as exc:
                err_console.print(Text(str(exc), style="red"))
    elif name in {"temp", "temperature"}:
        config = _set_float(config, "temperature", args)
    elif name in {"stop-bias", "stopbias"}:
        config = _set_float(config, "stop_bias", args)
    elif name == "stats":
        usage = client.usage
        console.print(Text(f"{usage.calls} calls · {usage.questions} questions · "
                           f"{usage.input_tokens} in · {usage.output_tokens} out",
                           style="dim"))
    else:
        err_console.print(Text(f"unknown command {parts[0]}; try /help", style="red"))
    return "continue", config, alpha


def _set_float(config: Config, field: str, args: list[str]) -> Config:
    if not args:
        console.print(Text(f"{field} = {getattr(config, field)}", style="dim"))
        return config
    try:
        updated = replace(config, **{field: float(args[0])})
        updated.validate()
    except (ValueError, ConfigError) as exc:
        err_console.print(Text(f"bad value: {exc}", style="red"))
        return config
    console.print(Text(f"{field} → {getattr(updated, field)}", style="dim"))
    return updated


if __name__ == "__main__":
    main()
