# Releasing

A release is a set of git tags. Pushing a tag runs `.github/workflows/release.yml`, which
publishes to crates.io or PyPI and creates the GitHub release. **A published version can never be
re-uploaded**, so everything before the first tag is about being sure.

| Tag | Publishes |
| --- | --- |
| `annex-vX.Y.Z` | `annex` to crates.io |
| `annex-multivector-vX.Y.Z` | `annex-multivector` to crates.io |
| `annex-server-vX.Y.Z` | `annex-server` to crates.io |
| `annex-py-vX.Y.Z` | the `ANNexDB` wheels (Linux x86_64 and aarch64, macOS universal2, Windows x86_64) to PyPI |

The workflow refuses a tag whose version does not match its manifest (`annex-py` also checks
`pyproject.toml`), and it uses the `## [X.Y.Z]` section of `CHANGELOG.md` as the GitHub release
notes. With no such section the release says "No changelog entry found".

## Order of operations

1. **Merge everything that belongs in the release.** Versions are bumped in the four manifests
   and `python/annex-py/pyproject.toml`, the changelog has an `## [X.Y.Z]` section, and CI is
   green on `master`.
2. **Run the pre-flight on the commit you will tag:** `scripts/release-preflight.sh`. It checks
   the commit is on `master`, the versions agree, the changelog section exists, nothing at that
   version is already tagged or published, CI is green on the commit, and that `cargo publish
   --dry-run` passes from a clean checkout. It ends by printing the exact tag commands.
3. **Tag `annex` first and push only that tag.** The other three depend on it.
4. **Wait** for its Release run to go green and for `annex X.Y.Z` to appear on crates.io (about
   five minutes; `scripts/release-verify.sh X.Y.Z` shows it).
5. **Tag and push the other three together** (`annex-multivector`, `annex-server`, `annex-py`).
   The wheel build takes longer than the crates.
6. **Verify:** `scripts/release-verify.sh X.Y.Z`. It checks the three crates, the wheels on PyPI,
   that each GitHub release exists with real notes, and that a clean `pip install
   ANNexDB==X.Y.Z` builds, searches, saves and reloads an index. Re-run it until it is green.
7. **Update the website** (see below) and anything that quotes the version.

Always tag the full commit hash that the pre-flight verified, not `master`, so the tags match
what was checked even if `master` moves.

## If something goes wrong

- **A tag run fails before anything is uploaded** (for example `cargo publish` rejects the
  package): fix it on `master`, delete the tag (`git push origin :refs/tags/<tag>` and
  `git tag -d <tag>`), and tag the fixed commit. Nothing was published, so the version is free.
- **A crate was published but a later step failed** (for example creating the GitHub release):
  the version is spent. Create the GitHub release by hand from the tag, with the changelog
  section as its notes. Do not try to publish the same version again.
- **A published version turns out to be bad:** `cargo yank` it and release a new patch version.
  Yanking does not delete it.

## The website

The website (annex-site) shows the latest release and a changelog built from these GitHub
releases. It reads them when it builds, so it needs a rebuild after a release.

`.github/workflows/notify-site.yml` does this: when the Release workflow finishes successfully it
calls a Vercel Deploy Hook. To enable it, create a Deploy Hook in the Vercel project (Settings,
Git, Deploy Hooks) and store its URL as the repository secret `ANNEX_SITE_DEPLOY_HOOK`. Without
the secret the workflow does nothing.

It listens for the Release workflow finishing (`workflow_run`) rather than for a `release`
event on purpose: releases created by the Release workflow itself use `GITHUB_TOKEN`, and GitHub
does not start new workflow runs for events caused by that token. Each tag run triggers a
rebuild, so expect up to four.
