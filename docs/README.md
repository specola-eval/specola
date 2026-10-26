# Documentation

Start with the [README tutorial](../README.md#from-simulator-to-verdict).
Use these guides when you need more detail.

- [Install Specola](install.md): the extras, the Python version each
  simulator adapter needs, and each simulator's setup, with the install
  check behind each.
- [Writing specifications](writing-specs.md): discover signals, write
  selectors and formulas, check them, and understand robustness.
- [Cheat sheet](cheat-sheet.md): the language on one page, known traps
  called out.
- [Running the monitor](running.md): terminal controls, result files,
  the queries file, startup refusals, and run states.
- [Python API](python-api.md): `Monitor` and `Session` inside your own
  rollout loop, and the engine underneath.
- [Web dashboard](web-dashboard.md): live viewing, recordings, and replay.
- [Surface parity](surface-parity.md): which commands, drive modes, and
  flags work on each bundled adapter, as measured, with the command and
  evidence per cell.

## For contributors

- [Adding an adapter](adding-an-adapter.md): the adapter contract, your
  own module versus a bundled one, and the two review gates bundling adds.
- [Adapter capabilities](adapter-capabilities.md): which signals and hooks
  each adapter carries, with the simulator source for every one; a new
  adapter fills its row before writing code.
- [Parser testing guide](testing/parser-testing-guide.md): extend the
  specification language and its tests.
