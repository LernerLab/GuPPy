# Making a release

A release takes two pull requests, one on each side of publishing it. The version in
`pyproject.toml` is already the one being released, since the previous release's post-release
pull request set it.

## 1. Build the changelog

On a `pre_release` branch, assemble the [changelog entries](development_environment.md#changelog-entries)
into `CHANGELOG.md`, with the release version and the day you will publish it:

```bash
towncrier build --version 2.0.0-beta4 --date "October 6th, 2026"
```

This writes a dated `# v2.0.0-beta4 (October 6th, 2026)` section at the top of `CHANGELOG.md` and
deletes the entry files it read. Commit both, open a pull request titled `Pre-Release 2.0.0-beta4`,
and read the assembled section to check the entries hold up as a list. Run
`towncrier build --draft --version 2.0.0-beta4` first to preview it without touching any files.

The pull request adds no changelog entry of its own, so the `detect-changelog-entry` check fails
on it; that check is not required to merge.

## 2. Publish

Merge the pull request, then publish a GitHub release tagged `v2.0.0-beta4` at that merge commit,
so the release holds exactly the entries the build read. Use the new `CHANGELOG.md` section as the
release notes. Publishing the release runs `auto-publish.yml`, which builds the package from the
tagged commit and uploads it to PyPI.

## 3. Bump the version

On a `post_release` branch, bump `version` in `pyproject.toml` to the next version and open a pull
request titled `Post-Release 2.0.0-beta4`. Every run records the GuPPy version that produced it in
`GuPPyParamtersUsed.json`, so the bump makes runs made from `main` report the version in
development rather than the one already released.
