import os
import time
from typing import Any
import uuid

import pylspclient
from pylspclient.lsp_structs import TextDocumentItem


class LspClient(pylspclient.LspClient):
    lsp_endpoint: pylspclient.LspEndpoint

    def __init__(self, lsp_endpoint, languageId: str, rootUri: str):
        super().__init__(lsp_endpoint)
        self.languageId = languageId
        self.rootUri = rootUri

        # Collect diagnostics
        self._diagnostics = dict()

        def on_diagnostics(params):
            self._diagnostics[params["uri"]] = params["diagnostics"]

        self.lsp_endpoint.notify_callbacks[
            "textDocument/publishDiagnostics"
        ] = on_diagnostics

    def didClose(self, textDocument):
        self.lsp_endpoint.send_notification(
            "textDocument/didClose", textDocument=textDocument
        )

    def ask_for_diagnostics(
        self,
        text: str,
        languageId: str = None,
        uri: str = None,
        version: int = 1,
    ):
        if languageId is None:
            languageId = self.languageId
        if uri is None:
            uri = "file://" + os.path.join(self.rootUri, str(uuid.uuid4()))

        item = TextDocumentItem(
            uri=uri,
            languageId=languageId,
            version=version,
            text=text,
        )

        self.didOpen(item)

        return item

    def collect_diagnostics(self, files: dict[Any, TextDocumentItem]):
        # TODO is there a better way to wait for diagnostics?
        # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_pullDiagnostics would be ideal, but too new
        for _ in range(100):
            if all(file.uri in self._diagnostics for file in files.values()):
                break
            time.sleep(0.01)

        for file in files.values():
            self.didClose(file)

        return {
            key: self._diagnostics.pop(file.uri, None) for key, file in files.items()
        }
