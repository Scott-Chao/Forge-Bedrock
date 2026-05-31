---
name: check-commit
description: Pre-commit readiness checklist for forge-bedrock. Use this whenever the user asks to commit, says "准备提交", or has completed implementation work and wants to verify everything is clean before committing. Guides the user through diff review, README progress updates, CLAUDE.md maintenance, linting, testing, and final commit.
---

# Check Commit

Pre-commit readiness workflow for the forge-bedrock project.

**Announce:** "Using check-commit skill to prepare this commit."

## Workflow

Run these steps in order. If any step fails, stop and report before proceeding.

### Step 1: Review Diff

Run these to understand what changed:

```bash
git status
git diff
git diff --cached
```

Summarize the changes for the user in 2-3 sentences. Ask: "Does this look correct?" Wait for confirmation before proceeding.

### Step 2: Update README.md Progress

Check if the changes implement any roadmap items listed in README.md (under the "Roadmap & TODO List" section). Look for `- [ ]` items that are now completed.

If new items should be checked off:
- Change `- [ ]` to `- [x]` for completed items
- Show the user which items were updated
- If no roadmap items are affected, skip this step silently

### Step 3: Check CLAUDE.md

Check if CLAUDE.md needs updating based on the changes:

```bash
# Check if CLAUDE.md was modified in this diff
git diff HEAD -- CLAUDE.md
```

Also think about whether the changes introduce new patterns, dependencies, or conventions that should be documented in CLAUDE.md. If yes, ask the user if they want to update it. If no, skip silently.

### Step 4: Run Ruff

```bash
ruff check .
```

- If ruff reports issues, show the errors and stop. Do not proceed until lint is clean.
- Do NOT auto-fix without asking the user first.

### Step 5: Run Tests

```bash
pytest tests/ -v
```

- If any tests fail, show the failures and stop. Do not proceed until tests pass.
- If the user wants to proceed with failing tests, respect their decision but note the risk.

### Step 6: Commit

All checks passed. Commit with a message in unordered list format:

```bash
git add -A
git commit -m "<description>

- <change 1>
- <change 2>
- <change 3>"
```

Rules:
- First line: short description of the overall change (no more than 70 chars)
- Following lines: bullet points starting with `- ` listing specific changes
- Use English for the commit message
- Do NOT add `Co-Authored-By` trailer
- Show the user the full commit message before committing
- Wait for confirmation before executing the commit

## Quick Reference

| Step | Command | On Failure |
|------|---------|------------|
| 1. Diff | `git status && git diff` | N/A (informational) |
| 2. README | Manual check of `- [ ]` items | N/A |
| 3. CLAUDE.md | `git diff HEAD -- CLAUDE.md` | Ask user |
| 4. Ruff | `ruff check .` | **Stop** |
| 5. Tests | `pytest tests/ -v` | **Stop** |
| 6. Commit | `git commit` | N/A |

## Red Flags

- Do NOT auto-fix ruff issues without asking
- Do NOT skip tests even if "it's just a small change"
- Do NOT add Co-Authored-By to the commit message
- Do NOT commit without showing the user the diff summary first
- Do NOT proceed past Step 4 or 5 if checks fail (unless the user explicitly overrides)
