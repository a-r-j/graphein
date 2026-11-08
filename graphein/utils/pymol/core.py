import contextlib
import os
import subprocess
import tempfile
import time
from xmlrpc.client import ServerProxy as Server
from xmlrpc.client import Transport

from ..dependencies import requires_external_dependencies

HOST = os.environ.get("PYMOL_RPCHOST", "localhost")
PORT = 9123


class _TimeoutTransport(Transport):
    """XML-RPC transport whose connections time out after ``timeout`` s."""

    def __init__(self, timeout):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


class _StopOnTimeout:
    """Wraps an XML-RPC method so a timed-out call stops PyMOL. Dotted
    names (e.g. ``system.listMethods``) resolve as with ``ServerProxy``.
    """

    def __init__(self, viewer, method):
        self._viewer = viewer
        self._method = method

    def __getattr__(self, name):
        return _StopOnTimeout(self._viewer, getattr(self._method, name))

    def __call__(self, *args):
        try:
            return self._method(*args)
        except TimeoutError:
            # Don't leave a hung PyMOL to be reused by the next call
            self._viewer.stop()
            raise


class MolViewer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = int(port)
        self._process = None
        self._start_args = ["-cKQ"]
        self._timeout = None

    def __del__(self):
        # Don't wait for PyMOL to exit: this can run at interpreter shutdown
        with contextlib.suppress(Exception):
            if self._process_is_running():
                self._process.terminate()

    def __getattr__(self, key):
        # Protocol probes (copy, pickle, hasattr) aren't PyMOL commands
        if key.startswith("_"):
            raise AttributeError(key)
        if not self._process_is_running():
            self.start(self._start_args, timeout=self._timeout)

        return _StopOnTimeout(self, getattr(self._server, key))

    @contextlib.contextmanager
    def call_timeout(self, timeout):
        """Time out calls after ``timeout`` seconds (``None`` waits
        indefinitely) inside this block, then restore the previous timeout.
        """
        previous, self._timeout = self._timeout, timeout
        if self._process_is_running():
            self._connect(timeout)
        try:
            yield
        finally:
            self._timeout = previous
            if self._process_is_running():
                self._connect(previous)

    def _connect(self, timeout=None):
        """Create the XML-RPC client. Calls raise ``TimeoutError`` after
        ``timeout`` seconds without a reply (``None`` waits indefinitely).
        """
        self._server = Server(
            uri=f"http://{self.host}:{self.port}/RPC2",
            transport=_TimeoutTransport(timeout),
        )

    def _process_is_running(self):
        return self._process is not None and self._process.poll() is None

    @requires_external_dependencies("pymol")
    def start(self, args=("-Q",), exe="pymol", timeout=None):
        """Start the PyMOL RPC server and connect to it
        Start simple GUI (-xi), suppress all output (-Q):
            >>> viewer.start(["-xiQ"])
        Start headless (-cK), with some output (-q):
            >>> viewer.start(["-cKq"])

        :param args: PyMOL command line options.
        :type args: list[str] | tuple[str, ...]
        :param exe: PyMOL executable.
        :type exe: str
        :param timeout: Seconds to wait for the server to start, and for
            each later call to it. ``None`` waits indefinitely. If PyMOL is
            already running, nothing changes: use :meth:`call_timeout` to
            bound calls to it.
        :type timeout: float | None
        :raises RuntimeError: if PyMOL exits before the server is up.
        :raises TimeoutError: if the server is not up after ``timeout``
            seconds.
        """
        assert isinstance(args, (list, tuple))

        if self._process_is_running():
            return

        # Used to restart PyMOL the same way if it dies
        self._start_args, self._timeout = list(args), timeout
        self._process = subprocess.Popen([exe, "-R"] + list(args))

        # wait for the server, giving each probe only the time remaining
        start_time = time.monotonic()
        while True:
            remaining = (
                None
                if timeout is None
                else max(timeout - (time.monotonic() - start_time), 0.01)
            )
            self._connect(remaining)
            try:
                self._server.bg_color("white")
                self._connect(timeout)
                break
            except OSError:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"PyMOL exited with code {self._process.returncode} "
                        "before its RPC server started"
                    )
                if (
                    timeout is not None
                    and time.monotonic() - start_time >= timeout
                ):
                    self.stop()
                    raise TimeoutError(
                        f"PyMOL RPC server not up after {timeout} seconds"
                    )
                time.sleep(0.1)

    def stop(self):
        if self._process_is_running():
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

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
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        try:
            self._server.png(filename, width, height, -1, int(ray))

            for i in range(1, progress_max):
                if os.path.getsize(filename) > 0:
                    break

                if progress is None:
                    progress = IntProgress(min=0, max=progress_max)
                    display(progress)

                progress.value += 1
                time.sleep(i / 10.0)

            if os.path.getsize(filename) == 0:
                raise RuntimeError("timeout exceeded")

            return Image(filename)
        finally:
            if progress is not None:
                progress.close()

            with contextlib.suppress(OSError):
                os.unlink(filename)


# Create a default instance for convenience
viewer = MolViewer()
