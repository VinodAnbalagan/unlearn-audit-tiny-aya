# Unlearning recovery audit on Tiny Aya (working title)

Lexsi AI Safety and Alignment Hackathon 2026, Track 2b. Work in progress.

## Layout
- `src/unlearn_audit/core.py`: model loading, generation and scoring, answer log-likelihood, residual-stream capture, steering hooks, parameter selection and snapshots.
- `scripts/pilot.py`: day-1 feasibility pilot (forget known facts with NPO, then test held-out steering recovery and Tamil queries).
- `scripts/compat.py`: CircuitKIT and SafeTune compatibility gate on Tiny Aya.
- `modal_app.py`: runs both on a Modal A100.
- `tests/test_core_cpu.py`: CPU logic test on a tiny random Cohere2 model (no download needed).

## Run
```bash
pip install -r requirements.txt          # local CPU test only needs torch + transformers
python tests/test_core_cpu.py            # should print ALL CHECKS PASSED
modal run modal_app.py::compat           # library gate on A100
modal run modal_app.py::pilot            # pilot on A100
modal volume get unlearn-audit-runs / ./runs_from_modal
```
Requires access to the gated `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0) and a Modal secret `huggingface` with `HF_TOKEN`.
