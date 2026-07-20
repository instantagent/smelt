# Security Policy

## Supported versions

Instant Agent is pre-1.0 and moves quickly. Security fixes are applied to the
latest release and to `main`. There is no backport guarantee for older tags
before 1.0.

## Reporting a vulnerability

Please report security issues privately — do **not** open a public issue.

Email **security@instantagent.dev** with:

- a description of the issue and its impact,
- steps to reproduce (a proof of concept if you have one), and
- any suggested remediation.

We aim to acknowledge reports within 3 business days and to keep you informed
as we work on a fix. Please give us a reasonable window to release a fix before
any public disclosure; we will credit reporters who wish to be named.

## Scope

Instant Agent runs models locally on your machine — there is no hosted service.
The most relevant areas for reports are:

- the package install/publish path (fetching and verifying `.agent` blobs from
  a registry), and
- any way a malicious package could cause code execution or escape its expected
  file-system footprint during build, install, or run.

Models produce text or audio and can be wrong or unsafe in the ordinary
generative-model sense; that is a model-quality matter, not a security report.
