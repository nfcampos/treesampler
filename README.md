# treesampler

[This is very much a work in progress, and is not ready for production use.]

A re-implementation of [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://arxiv.org/abs/2109.05093) that can be applied to code generation for any language with LSP support.

[LSP](https://microsoft.github.io/language-server-protocol/) is a protocol that defines a common interface between a language server and a language client. It is used by editors like VSCode to provide code completion, hover, and other features. The protocol is language agnostic, so it can be used for any language, as long as a language server is available.

## Installation

Support for each language is provided by a separate package. For example, to use treesampler for generating Python code, install `treesampler[py]`:

```bash
pip install git+https://github.com/nfcampos/treesampler.git#egg=treesampler[py]
```

For now only Python is supported, but more languages will be added in the future.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from treesampler import LspDiagnosticsProcessor, with_lsp

def generate(prompt, **kwargs):
    with with_lsp("python", server_python_module="ruff_lsp") as lsp_client:
      checkpoint = "Salesforce/codegen-350M-mono"
      model = AutoModelForCausalLM.from_pretrained(checkpoint)
      tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      processor = LspDiagnosticsProcessor(tokenizer, lsp_client)

      completion = model.generate(
          **tokenizer(prompt, return_tensors="pt"),
          logits_processor=[processor],
          **kwargs,
      )

      return tokenizer.decode(completion[0], skip_special_tokens=True)
```

## How it works

The idea is to use an LSP server to parse the code incrementally as it is being generated, generate diagnostics (linter errors and warnings) and we can use these diagnostics to constrain the generation process.

For example, if the LSP server reports a syntax error, we can use this information to prevent sampling tokens that would cause the syntax error.
If instead the LSP server reports a less severe warning, we can use this information to reduce the score of tokens that would cause the warning.

## How to add support for a new language

1. Find an existing LSP server for the language, this is a good place to look: https://microsoft.github.io/language-server-protocol/implementors/servers/

2. Write a test that uses it, see `tests/test_py.py` for an example.

3. Test the score adjustments produced by the base scorer, and optionally write a custom scorer for the language.

4. Contributing a new language is very welcome, please open a PR!
