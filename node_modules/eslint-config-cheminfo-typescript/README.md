# eslint-config-cheminfo-typescript

Shared ESLint config for TypeScript projects.

## Installation

```console
npx i -D eslint-config-cheminfo-typescript eslint
```

## Usage

Create a `.eslintrc.yml` with the following contents:

```yml
extends: [cheminfo-typescript, cheminfo-typescript/jsdoc, cheminfo-typescript/unicorn]
```

You can then customize the config for your project by changing rules in this file.

## References

This config extends our base [`eslint-config-cheminfo`](https://github.com/cheminfo/eslint-config) JavaScript config.
