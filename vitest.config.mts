import path from "node:path";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  test: {
    // Agent worktrees are full checkouts of other branches; their copies of the tests are not ours.
    exclude: [...configDefaults.exclude, ".claude/worktrees/**"],
  },
});
