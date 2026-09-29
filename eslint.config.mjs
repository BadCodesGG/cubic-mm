import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Worktrees under .claude/ are other checkouts of this repo, each linted from its own root.
    ".claude/**",
    ".venv/**",
  ]),
]);

export default eslintConfig;
