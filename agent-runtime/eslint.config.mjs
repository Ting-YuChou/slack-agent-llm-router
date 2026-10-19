import tseslint from "typescript-eslint";

// Typed lint only: catches promises nobody awaits or handles (an unhandled
// rejection terminates Node 22), not style.
export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**"] },
  {
    files: ["src/**/*.ts", "test/**/*.ts", "extensions/**/*.ts", "plugins/**/*.ts"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        project: ["./tsconfig.json", "./tsconfig.extensions.json"],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: { "@typescript-eslint": tseslint.plugin },
    linterOptions: { reportUnusedDisableDirectives: "error" },
    rules: {
      // node:test registers top-level tests by returning a promise the runner owns.
      "@typescript-eslint/no-floating-promises": ["error", {
        ignoreVoid: false,
        allowForKnownSafeCalls: [{ from: "package", package: "node:test", name: ["test", "describe", "it", "suite"] }],
      }],
      "@typescript-eslint/no-misused-promises": "error",
      "@typescript-eslint/await-thenable": "error",
    },
  },
);
