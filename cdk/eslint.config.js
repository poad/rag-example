// @ts-check

import eslint from '@eslint/js';
import stylistic from '@stylistic/eslint-plugin';
import stylisticTs from '@stylistic/eslint-plugin-ts';
import tseslint from 'typescript-eslint';
// @ts-expect-error ignore type errors
import eslintImport from "eslint-plugin-import";

import vitest from "@vitest/eslint-plugin";

import { includeIgnoreFile } from '@eslint/compat';
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const gitignorePath = path.resolve(__dirname, "./.gitignore");

export default tseslint.config(
  includeIgnoreFile(gitignorePath),
  eslint.configs.recommended,
  ...tseslint.configs.strict,
  ...tseslint.configs.stylistic,
  {
    ignores: [
      '**/*.d.ts',
      '*.{js,jsx}',
      'src/tsconfig.json',
      'src/stories',
      '**/*.css',
      'node_modules/**/*',
      './.next/*',
      'out',
      '.storybook',
      'dist',
      '.vinxi',
      '.output',
    ],
  },
  {
    files: ['{bin,lib,lambda}/**/*.{ts,tsx}'],
    ...eslintImport.flatConfigs.recommended,
    ...eslintImport.flatConfigs.typescript,
    plugins: {
      '@stylistic': stylistic,
      '@stylistic/ts': stylisticTs,
    },
    settings: {
      'import/internal-regex': '^~/',
      'import/resolver': {
        node: {
          extensions: ['.ts', '.tsx'],
        },
        typescript: {
          alwaysTryTypes: true,
        },
      },
    },
    rules: {
      '@stylistic/semi': ["error", "always"],
      '@stylistic/ts/indent': ['error', 2],
      'comma-dangle': ["error", "always-multiline"],
      semi: ["error", "always"],
      quotes: ["error", "single"],
      indent: ['error', 2]
    },
  },
  {
    files: ["test/**"], // or any other pattern
    plugins: {
      vitest
    },
    rules: {
      ...vitest.configs?.recommended.rules, // you can also use vitest.configs.all.rules to enable all rules
      "vitest/max-nested-describe": ["error", { "max": 3 }] // you can also modify rules' behavior using option like this
    },
  },
);
