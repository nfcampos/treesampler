from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from treesampler import LspDiagnosticsProcessor, with_lsp

# TODO use pipeline
def test_sampling_py():
    checkpoint = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    with with_lsp("python", server_python_module="ruff_lsp") as lsp_client:
        prompt = "def fibonacci(n):"
        tokenized = tokenizer(prompt, return_tensors="pt")
        completion = model.generate(
            **tokenized,
            num_beams=2,
            logits_processor=[
                LspDiagnosticsProcessor(tokenizer=tokenizer, lsp_client=lsp_client)
            ],
            max_new_tokens=1024,
            top_k=20,
            eos_token_id=[tokenizer.eos_token_id, 4299],  # def
        )

        print("decoded completion:\n")
        print(tokenizer.decode(completion[0], skip_special_tokens=True))
        for token in completion[0]:
            print(token.__repr__(), tokenizer.decode(token, skip_special_tokens=True))
