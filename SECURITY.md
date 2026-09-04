# Security Policy

## Supported versions

`errorbars` is at v0.1. Only the latest release receives fixes.

## Reporting a vulnerability

Use GitHub's **[Report a vulnerability](https://github.com/jimmyjames177414/errorbars/security/advisories/new)**
form, which opens a private advisory. Please do not open a public issue for anything
exploitable.

Expect an acknowledgement within a week. This is a small project maintained in spare time;
if something is urgent, say so in the title.

## Threat model, stated plainly

`errorbars` is a local command-line tool. It has a narrow attack surface, and being
specific about it is more useful than a generic policy:

**It reads files you point it at.** `analyze` parses JSON and JSONL from a results
directory; `run` parses a YAML spec and a JSONL dataset. YAML is parsed with
`yaml.safe_load`, never `yaml.load`, so a spec cannot construct arbitrary Python objects.
A results directory from an untrusted source is still untrusted input. It can make the
tool raise or produce nonsense, and the statistics it reports are only as trustworthy as
the trials file they came from.

**It compiles regexes from your spec.** An intervention with
`selector: {type: regex, ...}` and the `regex` scorer both compile user-supplied patterns.
A pathological pattern can cause catastrophic backtracking. Treat an `experiment.yaml`
from someone else the way you would treat a shell script from someone else.

**It makes outbound HTTPS requests, but only where you point it.** The
OpenAI-compatible provider POSTs to the `base_url` in your spec and nowhere else.
`errorbars power` makes no network calls at all, and there is a test that removes sockets
from the process to prove it.

**It writes to two places.** The results directory you name, and the response cache
(`$XDG_CACHE_HOME/errorbars` or `~/.cache/errorbars`, overridable with `--cache-dir`,
disabled with `--no-cache`).

## Credentials

API keys are read from an environment variable named in your spec (`api_key_env`,
defaulting to `OPENAI_API_KEY`). Keys are:

- never written to the results directory,
- never written to the response cache,
- never included in an error message,
- never logged.

A missing key is not an error, because local servers such as Ollama need none. In that
case the `Authorization` header is omitted rather than sent empty.

## Prompts and responses in results directories

`trials.jsonl` stores `prompt_sha256`, **not** the prompt text, so a results directory can
be shared for re-analysis without handing over the dataset. It does store
`response_text` in full, because re-scoring without re-calling the model is the point of
separating trials from outcomes. **If your model's responses contain sensitive data, the
results directory contains it too.** Treat it accordingly.

The response cache stores full responses on disk by design. `--no-cache` turns it off.

## Dependencies

Runtime: `numpy` and `PyYAML`. That is the entire list, and keeping it that short is
partly a supply-chain decision. Development additionally uses `pytest`, `mypy`, `ruff` and
`jsonschema`.
