# Publishing to PyPI with trusted publishing

No tokens to store or leak. PyPI trusts this GitHub repo through OIDC, the same setup as Evalwarden.

## One-time setup on PyPI (about 2 minutes)

1. Sign in at pypi.org, go to your account, Publishing, and add a pending publisher.
2. PyPI project name: open-redactor
3. Owner: jurayh
4. Repository: open-redactor
5. Workflow filename: publish.yml
6. Environment name: pypi

## One-time setup on GitHub

Add the workflow file from this repo at publish-workflow.yml.txt as .github/workflows/publish.yml through the GitHub web UI. The stored automation token cannot create workflow files, so this one file is a manual add. Then in the repo create an environment named pypi under Settings, Environments.

## Every release after that

1. Bump the version in pyproject.toml and src/open_redactor/__init__.py together.
2. Build and check locally with python -m build and twine check dist/*.
3. Push and tag: git tag v0.2.0 style, matching the version.
4. The workflow builds and publishes when a GitHub Release is created from the tag.
