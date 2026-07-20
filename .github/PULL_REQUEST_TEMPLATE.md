## Validation

- [ ] `bash tools/ci-lint.sh`
- [ ] `bash tools/test-default.sh -c release --parallel --num-workers 3 --quiet`

The full default suite is a local pre-merge requirement. If a change cannot
run it, explain why and record the focused validation used instead. Do not add
hosted macOS build or test jobs without explicit repository-owner approval.
