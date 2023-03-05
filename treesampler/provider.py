import subprocess
import sys
import threading
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import pylspclient

from .client import LspClient

# https://github.com/yeger00/pylspclient/blob/master/examples/python-language-server.py


class ReadPipe(threading.Thread):
    def __init__(self, pipe):
        threading.Thread.__init__(self)
        self.pipe = pipe

    def run(self):
        line = self.pipe.readline().decode("utf-8")
        while line:
            print(line)
            line = self.pipe.readline().decode("utf-8")


@contextmanager
def with_lsp(
    language: str,
    server_python_module: str | None = None,
    server_executable: str | None = None,
) -> subprocess.Popen[bytes]:
    # Launch the LSP server
    if server_python_module is not None:
        proc = subprocess.Popen(
            [sys.executable, "-m", server_python_module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    elif server_executable is not None:
        proc = subprocess.Popen(
            [server_executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        raise ValueError(
            "Either server_python_module or server_executable must be provided."
        )

    # Read the LSP server's stderr to debug errors
    read_pipe = ReadPipe(proc.stderr)
    read_pipe.start()

    # Create the LSP client
    json_rpc_endpoint = pylspclient.JsonRpcEndpoint(proc.stdin, proc.stdout)
    lsp_endpoint = pylspclient.LspEndpoint(
        json_rpc_endpoint,
        dict(),
        {
            "window/logMessage": lambda params: None,
            "window/showMessage": lambda params: print(params),
        },
    )

    with TemporaryDirectory() as root_dir:
        lsp_client = LspClient(lsp_endpoint, language, root_dir)

        # Define the capabilities, we don't currently need any
        capabilities = dict()

        # Initialize the LSP server
        lsp_client.initialize(
            proc.pid, None, "file://" + root_dir, None, capabilities, "off", dict()
        )
        lsp_client.initialized()

        try:
            yield lsp_client
        finally:
            # Shutdown the LSP server
            lsp_client.shutdown()
            lsp_client.exit()
            proc.terminate()
