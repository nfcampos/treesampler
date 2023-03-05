import torch
from transformers import LogitsProcessor, PreTrainedTokenizer

from treesampler.client import LspClient
from treesampler.scorer import BaseScorer


# https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/generation/logits_process.py#L274


class LspDiagnosticsProcessor(LogitsProcessor):
    tokenizer: PreTrainedTokenizer
    lsp_client: LspClient
    scorer: BaseScorer

    diagnostics_per_beam: int
    filter_value: float

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lsp_client: LspClient,
        scorer: BaseScorer = BaseScorer(),
        diagnostics_per_beam: int = 8,
        filter_value: float = -float("Inf"),
    ):
        self.tokenizer = tokenizer
        self.lsp_client = lsp_client
        self.scorer = scorer

        self.diagnostics_per_beam = diagnostics_per_beam
        self.filter_value = filter_value

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        n_beams = scores.shape[0]
        n_diagnostics = n_beams * self.diagnostics_per_beam

        # Get the top n_checks tokens
        topk = scores.topk(n_diagnostics, dim=1)

        # Get decoded inputs
        decoded_inputs = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        files = dict()

        for i in range(n_beams):
            # Get decoded topk tokens
            decoded_topk = self.tokenizer.batch_decode(
                topk.indices[i].reshape((n_diagnostics, -1)),
                skip_special_tokens=True,
            )

            for j in range(n_diagnostics):
                # Send each code_string to the LSP server for diagnostics
                code_string = decoded_inputs[i] + decoded_topk[j]
                files[(i, j)] = self.lsp_client.ask_for_diagnostics(code_string)

        # Collect diagnostics
        diagnostics = self.lsp_client.collect_diagnostics(files)

        for i in range(n_beams):
            for j in range(n_diagnostics):
                # Convert the diagnostics to a score
                code_string = decoded_inputs[i] + decoded_topk[j]
                scores[i, topk.indices[i, j]] = self.scorer(
                    scores[i, topk.indices[i, j]],
                    code_string,
                    diagnostics[(i, j)],
                    decoded_topk[j],
                )

            # # Set the scores of all tokens that are not in the topk to the filter value
            # others = torch.ones_like(scores[i], dtype=torch.bool)
            # others[topk.indices[i]] = False
            # scores[i, others] = self.filter_value

        return scores
