---
description: push to main, monitor GitHub Actions CI, and verify docs deployment on GitHub Pages
---

# Push to Main and Monitor CI

After pushing commits or tags to `main`, always verify that the GitHub Actions CI pipeline passes before considering the work done.

// turbo-all

1. Push the current branch / tag to GitHub:
```
git push origin main
```
   If pushing a tag, also run:
```
git push origin --tags
```

2. Wait a few seconds for GitHub to register the new run, then find the latest workflow run ID:
```
gh run list --branch main --workflow ci.yml --limit 5
```

3. Watch the run live in the terminal (replace `<RUN_ID>` with the ID from the previous step — pick the most recent one):
```
gh run watch <RUN_ID> --exit-status
```
   This streams real-time job/step status and returns a non-zero exit code if the run fails.

4. If the run fails, inspect the failing job logs:
```
gh run view <RUN_ID> --log-failed
```
   Read the output carefully, identify the root cause, fix the code, commit, and repeat from step 1.

5. Once `gh run watch` exits successfully (exit code 0), the CI is green.

6. Verify the `docs` job succeeded specifically:
```
gh run view <RUN_ID> --json jobs --jq '.jobs[] | select(.name | startswith("Deploy docs")) | {name, status, conclusion}'
```
   Conclusion must be `success`. If it is `failure`, dump its logs:
```
gh run view <RUN_ID> --log-failed
```

7. Confirm the GitHub Pages site is live and reflects the latest content by fetching the docs URL:
```
curl -sI https://bmsuisse.github.io/rustfuzz/ | head -5
```
   Expect `HTTP/2 200`. If you get a non-200 or a stale cache, wait ~60 s and retry — Pages deployments can lag slightly behind the CI job completion.

## Key CI jobs to watch

| Job | Trigger | What it does |
|---|---|---|
| `test` | every push to `main` | cargo check, maturin build, pytest, pyright |
| `docs` | push to `main` or tag | builds & deploys MkDocs → https://bmsuisse.github.io/rustfuzz/ |
| `linux / musllinux / windows / macos` | tags only | builds release wheels |
| `sdist` | tags only | builds source distribution |
| `release` | tags only | publishes to PyPI |

> **Note:** For a plain `main` push (no tag) only `test` and `docs` run.  
> For a tagged release all jobs including `release` (PyPI publish) run.
