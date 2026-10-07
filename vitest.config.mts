import path from "node:path";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  test: {
    // Local tool state, including other checkouts of this repo; their tests are not ours.
    exclude: [...configDefaults.exclude, ".claude/**"],
  },
});
