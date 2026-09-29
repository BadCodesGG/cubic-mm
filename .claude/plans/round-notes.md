# Round notes (orchestrator)

## Round 0 (done 2026-09-29)
- Pipeline: 200 neurons nearest the centroid, hi 281k nodes / lite 205k, 4.2 MB, `pack.py --check` green.
- Scene: hero-17.png passes the gate on depth and scale; restraint left to the real sim. 654k ribbon instances at 6.06 ms/frame on WebGPU (RTX 40); WebGL2 fallback renders at 16.7 ms.
- Fixed: strict-mode double-mount rendering on a detached canvas (zero-size swapchain), pointer-lock refusals spamming the console. Verified clean in a fresh in-app tab.
- Look board for the user: https://claude.ai/artifact/J3DDGoJ1DqSUnMqW8HePJ1
- Dev server for the user: launch config `cubic-mm-dev` (port 3050) in portfolio/.claude/launch.json.

## Round 1 (spawned 2026-09-29)
Worktrees under .claude/worktrees/: r1-pipeline (Sonnet), r1-sim (Opus), r1-nav (Sonnet), r1-audio (Sonnet).
Merge order when they land: sim, then nav, then audio, then pipeline (data last, biggest and least conflicting).
app.ts is the merge hotspot: each builder keeps its edits to a commented block.
User's bar: unique, shareable, real data on screen.

## Round 2 and definition of done (2026-09-29)
- Merged r2/quality and r2/about. Review findings (6 engine, 9 UI) fixed in 12aa809 and 9774112.
- All checks green on 9774112: tsc, lint, 119 tests, build, smoke (WebGPU, WebGL2, parity 1.6%), pack --check.
- Repo: github.com/BadCodesGG/cubic-mm (private). main = scaffold base; PR from feat/one-cubic-millimetre carries the build.
- Ideas for a follow-up round (from the user's "what would improve it" question): scripted intro, return-to-cluster + minimap, shareable camera links, cascade counter and lines, cell search, time control, screenshot mode, regenerate hero-still.jpg.
