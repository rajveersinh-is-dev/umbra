
### 2024-04-18: Improve Test Coverage for Multivariate Amputation Engine
Increased the test coverage of `umbra/benchmark/amputation.py` from 72% to 99%. Extensive unit tests were added in `tests/test_amputation_and_benchmarks.py` to cover input validation edge cases, fallback execution branches in `_calibrate_logit_shift` (e.g. standard deviation equal to 0, extreme probability brackets), and different RNG generators. This addresses the priority item regarding finding the most critical untested module and comprehensively improving overall test coverage.
