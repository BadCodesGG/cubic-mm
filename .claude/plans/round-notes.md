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
