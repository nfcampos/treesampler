from human_eval.data import read_problems, write_jsonl
from transformers import AutoModelForCausalLM, AutoTokenizer

from treesampler import LspDiagnosticsProcessor, with_lsp

problems = read_problems("./data/HumanEval.jsonl.gz")
problems_to_use = list(problems.keys())


def generate_without_treesampler(prompt, **kwargs):
    checkpoint = "Salesforce/codegen-350M-multi"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    completion = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=512,
        eos_token_id=[tokenizer.eos_token_id, 4299],  # def
        **kwargs,
    )

    return tokenizer.decode(completion[0], skip_special_tokens=True)


def generate_with_treesampler(prompt, lsp_client, **kwargs):
    checkpoint = "Salesforce/codegen-350M-multi"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    processor = LspDiagnosticsProcessor(tokenizer=tokenizer, lsp_client=lsp_client)

    completion = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=512,
        eos_token_id=[tokenizer.eos_token_id, 4299],  # def
        logits_processor=[processor],
        **kwargs,
    )

    return tokenizer.decode(completion[0], skip_special_tokens=True)


if __name__ == "__main__":
    strategies = [
        (
            "codegen_350m_multi_with_treesampler_beams=2",
            lambda prompt: generate_with_treesampler(prompt, lsp_client, num_beams=2),
        ),
        (
            "codegen_350m_multi_without_treesampler_beams=2",
            lambda prompt: generate_without_treesampler(prompt, num_beams=2),
        ),
        # (
        #     "codegen_350m_multi_with_treesampler_beams=2_do_sample=True",
        #     lambda prompt: generate_with_treesampler(
        #         prompt, lsp_client, num_beams=2, do_sample=True
        #     ),
        # ),
        # (
        #     "codegen_350m_multi_without_treesampler_beams=2_do_sample=True",
        #     lambda prompt: generate_without_treesampler(
        #         prompt, num_beams=2, do_sample=True
        #     ),
        # ),
        (
            "codegen_350m_multi_with_treesampler_beams=4",
            lambda prompt: generate_with_treesampler(prompt, lsp_client, num_beams=4),
        ),
        (
            "codegen_350m_multi_without_treesampler_beams=4",
            lambda prompt: generate_without_treesampler(prompt, num_beams=4),
        ),
    ]

    with with_lsp("python", server_python_module="ruff_lsp") as lsp_client:
        for name, strategy in strategies:
            samples = []

            for task_id in problems_to_use:
                prompt = problems[task_id]["prompt"]
                print(f"generating completion for {task_id}")
                completion = strategy(prompt)
                samples.append(dict(task_id=task_id, completion=completion))

            write_jsonl(f"{name}.jsonl", samples)
