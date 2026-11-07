---
name: ship
description: Commit current work on a feature branch and push it through the no-mistakes gate. Run only when the user types /ship.
disable-model-invocation: true
---

1. If on `main`, create a feature branch first.
2. Commit any uncommitted changes with a short descriptive message.
3. Run `git push no-mistakes`.
4. Tell the user to run `no-mistakes` in the terminal to review the results.