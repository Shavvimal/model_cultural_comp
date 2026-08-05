# Security Policy

## Supported versions

This repository is the code artefact behind a paper and is developed on a rolling
basis. Fixes are applied to the latest `main` and released from there; please ensure
you are running the most recent version before reporting an issue.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through GitHub's built-in vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** to open a private advisory.

This keeps the details confidential until a fix is available. We aim to
acknowledge reports within a few days and will coordinate a fix and disclosure
timeline with you.

## Sensitive surface

The two things worth protecting here are licensed survey data and the API keys used
to collect model responses. Neither is in the repository, and neither should ever
enter it — including via an issue, a PR diff, or a notebook output cell.

- **WVS / EVS / IVS microdata.** The Integrated Values Surveys inputs are obtained
  under data-use agreements with the WVS Association and GESIS that permit personal
  research use but **forbid redistribution**. The data must never be committed,
  attached to an issue, vendored into a fixture, or fetched by CI. `data/` and
  `*.pkl` are gitignored precisely so an accidental `git add -A` cannot leak them; a
  change that removes or narrows those ignore rules is a security-relevant change.
  If you find survey microdata reachable anywhere in the repository or its git
  history, report it privately rather than opening an issue — history rewriting has
  to happen before the leak is public knowledge.
- **Ollama and Ollama-Cloud API keys.** The collection harness talks to a local or
  hosted Ollama endpoint. Keys and hosts are supplied as explicit arguments or
  environment variables at run time and are never read from, or written to, tracked
  files. Do not paste a key into a bug report, a log excerpt, or a modelfile; rotate
  any key that has been shared, then report.
- **Collected model responses.** Stored responses under `data/collection/` are
  derived outputs, not survey microdata, but excerpts pasted into reports should
  still be trimmed to the minimum needed to reproduce.

When in doubt, describe the issue abstractly in the private advisory and we will
follow up for the minimum detail needed to reproduce. Do not include working exploit
payloads, credentials, or data extracts in public channels.
