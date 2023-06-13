'use strict';

module.exports = {
  extends: 'cheminfo/base',
  overrides: [
    {
      files: ['*.{ts,tsx,cts,mts}'],
      extends: ['plugin:import/typescript'],
      plugins: ['@typescript-eslint'],
      parser: '@typescript-eslint/parser',
      parserOptions: {
        sourceType: 'module',
        project: 'tsconfig.json',
      },
      rules: {
        // https://github.com/typescript-eslint/typescript-eslint
        // Last rules review: v5.27.0

        // Supported rules (https://github.com/typescript-eslint/typescript-eslint/tree/master/packages/eslint-plugin#supported-rules)
        '@typescript-eslint/adjacent-overload-signatures': 'error',
        '@typescript-eslint/await-thenable': 'error',
        '@typescript-eslint/ban-ts-comment': 'error',
        '@typescript-eslint/ban-types': 'error',
        '@typescript-eslint/consistent-generic-constructors': 'error',
        '@typescript-eslint/consistent-type-assertions': 'error',
        '@typescript-eslint/no-base-to-string': 'error',
        '@typescript-eslint/no-duplicate-enum-values': 'error',
        '@typescript-eslint/no-dynamic-delete': 'error',
        '@typescript-eslint/no-explicit-any': 'warn',
        '@typescript-eslint/no-extraneous-class': 'error',
        '@typescript-eslint/no-floating-promises': [
          'error',
          {
            ignoreVoid: true,
          },
        ],
        '@typescript-eslint/no-for-in-array': 'error',
        '@typescript-eslint/no-inferrable-types': 'error',
        // Only affects special edge cases.
        '@typescript-eslint/no-invalid-void-type': 'off',
        '@typescript-eslint/no-meaningless-void-operator': 'error',
        '@typescript-eslint/no-misused-new': 'error',
        '@typescript-eslint/no-misused-promises': 'error',
        '@typescript-eslint/no-namespace': 'error',
        '@typescript-eslint/no-non-null-assertion': 'error',
        '@typescript-eslint/no-this-alias': 'error',
        '@typescript-eslint/no-unnecessary-boolean-literal-compare': 'error',
        // Problematic when doing checks with array elements or values from records.
        '@typescript-eslint/no-unnecessary-condition': 'off',
        '@typescript-eslint/no-unnecessary-type-assertion': 'error',
        '@typescript-eslint/no-unnecessary-type-constraint': 'error',
        // TODO: enable no-unsafe rules when we are ready to ban `any` everywhere.
        '@typescript-eslint/no-unsafe-argument': 'off',
        '@typescript-eslint/no-unsafe-assignment': 'off',
        '@typescript-eslint/no-unsafe-call': 'off',
        '@typescript-eslint/no-unsafe-member-access': 'off',
        '@typescript-eslint/no-unsafe-return': 'off',
        '@typescript-eslint/no-useless-empty-export': 'error',
        '@typescript-eslint/no-var-requires': 'error',
        '@typescript-eslint/prefer-as-const': 'error',
        '@typescript-eslint/prefer-enum-initializers': 'warn',
        '@typescript-eslint/prefer-for-of': 'error',
        '@typescript-eslint/prefer-function-type': 'error',
        '@typescript-eslint/prefer-includes': 'error',
        '@typescript-eslint/prefer-literal-enum-member': 'error',
        '@typescript-eslint/prefer-namespace-keyword': 'error',
        // TODO: enable when we are ready for it.
        '@typescript-eslint/prefer-nullish-coalescing': 'off',
        '@typescript-eslint/prefer-optional-chain': 'warn',
        '@typescript-eslint/prefer-reduce-type-parameter': 'error',
        '@typescript-eslint/prefer-string-starts-ends-with': 'error',
        '@typescript-eslint/prefer-ts-expect-error': 'error',
        '@typescript-eslint/require-array-sort-compare': 'error',
        '@typescript-eslint/restrict-plus-operands': 'error',
        '@typescript-eslint/restrict-template-expressions': [
          'error',
          // TODO: remove allowAny when we are ready.
          { allowAny: true },
        ],
        '@typescript-eslint/switch-exhaustiveness-check': 'error',
        '@typescript-eslint/triple-slash-reference': 'error',
        '@typescript-eslint/unbound-method': 'error',

        // Extension rules (https://github.com/typescript-eslint/typescript-eslint/tree/master/packages/eslint-plugin#extension-rules)
        // There is no "@typescript-eslint/no-undef" rule, but we don't need it because TypeScript already flags missing global variables.
        'no-undef': 'off',
        'default-param-last': 'off',
        '@typescript-eslint/default-param-last': 'error',
        'dot-notation': 'off',
        '@typescript-eslint/dot-notation': 'error',
        'no-array-constructor': 'off',
        '@typescript-eslint/no-array-constructor': 'error',
        'no-dupe-class-members': 'off',
        '@typescript-eslint/no-dupe-class-members': 'error',
        'no-duplicate-imports': 'off',
        '@typescript-eslint/no-duplicate-imports': [
          'error',
          { includeExports: false },
        ],
        'no-empty-function': 'off',
        '@typescript-eslint/no-empty-function': 'warn',
        'no-extra-semi': 'off',
        '@typescript-eslint/no-extra-semi': 'error',
        'no-implied-eval': 'off',
        '@typescript-eslint/no-implied-eval': 'error',
        'no-loop-func': 'off',
        '@typescript-eslint/no-loop-func': 'error',
        'no-loss-of-precision': 'off',
        '@typescript-eslint/no-loss-of-precision': 'error',
        'no-redeclare': 'off',
        '@typescript-eslint/no-redeclare': 'error',
        'no-throw-literal': 'off',
        '@typescript-eslint/no-throw-literal': 'error',
        'no-unused-expressions': 'off',
        '@typescript-eslint/no-unused-expressions': 'error',
        'no-unused-vars': 'off',
        '@typescript-eslint/no-unused-vars': [
          'error',
          { vars: 'all', args: 'after-used', ignoreRestSiblings: true },
        ],
        'padding-line-between-statements': 'off',
        '@typescript-eslint/padding-line-between-statements': [
          'error',
          { blankLine: 'always', prev: 'directive', next: '*' },
        ],
        'no-return-await': 'off',
        '@typescript-eslint/return-await': 'error',
        '@typescript-eslint/naming-convention': [
          'error',
          {
            selector: ['typeLike'],
            format: ['PascalCase'],
          },
          {
            selector: ['variable', 'typeProperty'],
            format: ['camelCase', 'UPPER_CASE', 'PascalCase'],
            leadingUnderscore: 'allowSingleOrDouble',
          },
          {
            selector: ['function', 'typeMethod'],
            format: ['camelCase', 'PascalCase'],
          },
          {
            selector: [
              'method',
              'classProperty',
              'classMethod',
              'parameterProperty',
              'accessor',
            ],
            format: ['camelCase'],
            leadingUnderscore: 'allowSingleOrDouble',
          },
          {
            selector: [
              'objectLiteralProperty',
              'objectLiteralMethod',
              'enumMember',
            ],
            format: null,
          },
        ],
      },
    },
  ],
};
