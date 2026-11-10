# Contributing to Rung

Bug reports, documentation fixes, tests and embodiment adapters are welcome. For a
larger feature, open an issue describing the use case before starting an implementation.
Use [GitHub issues](https://github.com/Manda-Robotics/rung/issues) for bugs and proposals;
include the Rung revision, Python version, command and a minimal reproduction.

## Development setup

Use Python 3.11 or newer. From a clone of the repository:

```sh
uv sync --locked --all-extras
make check
make examples
```

The tests run offline with no GPU, simulator or API key. Optional SDK and accelerated
perception tests skip if their dependencies are absent; `--all-extras` installs them.
Isaac Sim integration requires a separate GPU environment and calibration; see the
[RoboLab guide](deploy/robolab/README.md).

## Before submitting a pull request

```sh
make fmt
make check
make coverage
uv build
```

Ruff checks imports, common Python errors and formatting. Add a regression test when
fixing a behavior bug, and update the relevant guide when changing commands or contracts.
Keep generated runs, recordings, coverage output and reports out of commits.

CI runs the suite with branch measurement on Python 3.11 through 3.14, enforces an
80 percent combined line-and-branch coverage floor, exercises the offline examples,
and builds and installs the wheel in a clean environment. Coverage XML and HTML
reports are downloadable from each check run's artifacts. Maintainers can optionally
connect the repository to Codecov and set `CODECOV_TOKEN` in GitHub Actions secrets.
Public repositories can also use Codecov's
[tokenless upload option](https://docs.codecov.com/docs/codecov-tokens).

## Adapters and evaluation changes

Follow the [contract and adapter checklist](docs/contracts.md), and run the embodiment
probes before a policy. Keep success grading separate from policy inputs. Document
changes to task splits, timing, budgets or declared inputs using the
[evaluation protocol](docs/protocol.md).

## License and security

Contributions are licensed under [Apache-2.0](LICENSE). Preserve third-party notices.
Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
