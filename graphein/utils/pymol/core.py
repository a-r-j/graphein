from __future__ import print_function

import os
import subprocess
import tempfile
import time

from ..dependencies import requires_external_dependencies
from .compat import Server

HOST = os.environ.get("PYMOL_RPCHOST", "localhost")
PORT = 9123


class MolViewer(object):
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = int(port)
        self._process = None

    def __del__(self):
        self.stop()

    def __getattr__(self, key):
        if not self._process_is_running():
            self.start(["-cKQ"])

        return getattr(self._server, key)

    def _process_is_running(self):
        return self._process is not None and self._process.poll() is None

    @requires_external_dependencies("pymol")
    def start(self, args=("-Q",), exe="pymol"):
        """Start the PyMOL RPC server and connect to it
        Start simple GUI (-xi), suppress all output (-Q):
            >>> viewer.start(["-xiQ"])
        Start headless (-cK), with some output (-q):
            >>> viewer.start(["-cKq"])
        """
        if self._process_is_running():
            print("A PyMOL RPC server is already running.")
            return

        assert isinstance(args, (list, tuple))

        self._process = subprocess.Popen([exe, "-R"] + list(args))

        self._server = Server(uri="http://%s:%d/RPC2" % (self.host, self.port))

        # wait for the server
        while True:
            try:
                self._server.bg_color("white")
                break
            except IOError:
                time.sleep(0.1)

    def stop(self):
        if self._process_is_running():
            self._process.terminate()

    def display(self, width=0, height=0, ray=False, timeout=120):
        """Display PyMol session
        :param width: width in pixels (0 uses current viewport)
        :param height: height in pixels (0 uses current viewport)
        :param ray: use ray tracing (if running PyMOL headless, this parameter
        has no effect and ray tracing is always used)
        :param timeout: timeout in seconds
        Returns
        -------
        fig : IPython.display.Image
        """
        from IPython.display import Image, display
        from ipywidgets import IntProgress

        progress_max = int((timeout * 20) ** 0.5)
        progress = None
        filename = tempfile.mktemp(".png")

        try:
            self._server.png(filename, width, height, -1, int(ray))

            for i in range(1, progress_max):
                if os.path.exists(filename):
                    break

                if progress is None:
                    progress = IntProgress(min=0, max=progress_max)
                    display(progress)

                progress.value += 1
                time.sleep(i / 10.0)

            if not os.path.exists(filename):
                raise RuntimeError("timeout exceeded")

            return Image(filename)
        finally:
            if progress is not None:
                progress.close()

            try:
                os.unlink(filename)
            except:
                pass


# Create a default instance for convenience
viewer = MolViewer()
