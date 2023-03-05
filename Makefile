.PHONY: = test test-watch

test:
	poetry run pytest

test-watch:
	poetry run ptw --ext .py

eval-sample:
	poetry run python -m scripts.generate

eval-score:
	poetry run python -m human_eval.evaluate_functional_correctness codegen_350m_nl_without_treesampler_beams=2.jsonl --problem_file data/HumanEval.jsonl.gz