# sim-agentic-grpo

This repository trains a multi-turn coding agent with synchronous GRPO on SWE-bench, and simulates how long one of those training steps takes when the system around the agent changes.

The training run is the workload. It produces the traces, timelines, and cost-model measurements. The simulator replays that work on a different machine, parallelism, or concurrency and reports the step time and the timeline, without launching another run.

## Training

`src/agentic_grpo` is a [verl](https://github.com/verl-project/verl) agent loop. Each sampled prompt is one episode: the model generates, the loop parses a tool call, a tool runs inside a SWE-bench Docker container, the observation is appended, and this repeats until the agent submits a patch or hits a turn or context limit. [SGLang](https://github.com/sgl-project/sglang) serves the rollout synchronously (`policy_lag = 0`). The reward is binary: `1.0` if the official SWE-bench harness says the patch resolved the instance, otherwise `0.0`. verl then computes the GRPO advantage and updates the actor.

The run this repo is set up for:

| | |
| --- | --- |
| Model | Qwen3-30B-A3B-Instruct-2507 |
| Data | SWE-bench Verified, 500 instances, 3 epochs |
| Hardware | 8× NVIDIA B200 |
| Agent | [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) tools (`bash`, editor, `submit`) |
| Batch | 256 prompts × 8 samples |

Every episode records tokens, turn timings, and a timestamped timeline of rollout and training. Those records are what the simulator is checked against. Setup and the launch script are in [TRAINING.md](TRAINING.md). What each logged number means is in [METRICS.md](METRICS.md).

## Simulator

`src/simulator` is a trace-driven discrete-event simulator of one GRPO step. A trace says what work there was. An adapter turns it into an IR of sessions, requests, tool tasks, and resource leases. A config bundle says what system runs that work: SGLang replicas, tensor-parallel size, KV cache, concurrency caps, container and eval slots, and a cost model fitted on a measured grid. The engine then runs the work on that system.

The scheduler is SGLang's own scheduler, with the model worker stubbed out. Each forward is priced by the cost model. Container starts, tool calls, and grading take their durations from the trace, or from priors when the trace did not record them. The clock follows SGLang's overlap schedule: a request's arrival is when the previous turn finished generating plus the tool time from the trace, not a timestamp copied out of a previous run.

It answers changes that leave the token stream alone: tensor-parallel size, replica count, concurrency cap, KV dtype, hardware. A change that would make the model write something else — a different turn cap, tool set, prompt, or model — is outside what a replay of an old trace can say, and the simulator is built to refuse those questions. The design contract, including which numbers may be inputs and which the engine has to compute, is [SIMULATOR.md](SIMULATOR.md). An example bundle is [configs/simulator/example_bundle.yaml](configs/simulator/example_bundle.yaml).

```
trace ── adapter ──▶ IR (the work)  ──┐
                                      ├──▶ engine ──▶ predicted timeline, step time
config bundle ────────────────────────┘
```

## Layout

| Path | Role |
| --- | --- |
| `src/agentic_grpo/` | The verl agent loop, tools, SWE-bench reward, metrics, and timeline |
| `src/simulator/` | Trace adapters, IR, cost model, and the timeline engine |
| `configs/grpo_swebench.yaml` | verl trainer config for the 8× B200 run |
| `configs/simulator/` | Simulator config bundles |
| `scripts/run_grpo.sh` | Launch training |
| `scripts/measure_cost_grid.py`, `scripts/fit_cost_model.py` | Measure and fit the forward-pass cost model |
| `scripts/priors_template.py`, `scripts/show_priors.py` | List prior knobs and what a bundle resolves them to |
| `scripts/simulator_dev/` | Checks of the simulator against our own runs |
| `analysis/` | Per-run metrics, timelines, and simulator reports |

## Tests

```bash
pytest -q
```

The agent-loop tests stub verl when it is not installed, so they run without CUDA. The scheduler tests that drive a real SGLang `Scheduler` skip themselves unless the model config is on disk.
