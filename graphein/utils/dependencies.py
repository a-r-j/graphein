import os
import sys
from functools import wraps
from shutil import which
from typing import Optional


class MissingDependencyError(Exception):
    """Raised when a required dependency is missing."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def import_message(
    submodule: str,
    package: str,
    conda_channel: Optional[str] = None,
    pip_install: bool = False,
    extras: bool = False,
) -> str:
    """
    Return warning if package is not found.
    Generic message for indicating to the user when a function relies on an
    optional module / package that is not currently installed. Includes
    installation instructions. Typically used in conjunction without optional featurisation libraries

    :param submodule: graphein submodule that needs an external dependency.
    :type submodule: str
    :param package: External package this submodule relies on.
    :type package: str
    :param conda_channel: Conda channel package can be installed from, if at all. Defaults to ``None``.
    :type conda_channel: str, optional
    :param pip_install: Whether package can be installed via pip. Defaults to ``False``.
    :type pip_install: bool
    :param extras: Whether package would be installed with
        ``pip install graphein[extras]``. Defaults to ``False``.
    :type extras: bool
    """
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    installable = True
    if is_conda:
        if conda_channel is None:
            installable = False
            installation = f"{package} cannot be installed via conda"
        else:
            installation = f"conda install -c {conda_channel} {package}"
    elif pip_install:
        installation = f"pip install {package}"
    else:
        installable = False
        installation = f"{package} cannot be installed via pip"

    message = f"To use the Graphein submodule {submodule}, you need to install: {package} "
    if installable:
        message += f"\nTo do so, use the following command: {installation}"
    else:
        message += f"\n{installation}"

    if extras:
        message += (
            "\nAlternatively, you can install graphein with the extras: "
        )
        message += "\n\npip install graphein[extras]"
    return message


def is_tool(name: str, error: bool = False) -> bool:
    """Checks whether ``name`` is on ``PATH`` and is marked as an executable.

    Source:
    https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    :param name: Name of program to check for execution ability.
    :type name: str
    :param error: Whether to raise an error.
    :type error: bool. Defaults to ``False``.
    :return: Whether ``name`` is on PATH and is marked as an executable.
    :rtype: bool
    :raises MissingDependencyError: If ``error`` is ``True`` and ``name`` is
        not on ``PATH`` or is not marked as an executable.
    """
    found = which(name) is not None
    if not found and error:
        raise MissingDependencyError(name)
    return found


# Decorator for checking if a function has the required dependencies
def requires_external_dependencies(*deps):
    """
    A decorator to check if all required dependencies are installed before
    calling the decorated function. If a dependency is missing, it raises
    a MissingDependencyError.

    :param deps: A list of dependencies (as strings) to check for.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing_deps = [dep for dep in deps if not is_tool(dep)]
            if missing_deps:
                missing = ", ".join(missing_deps)
                raise MissingDependencyError(
                    f"Missing dependencies: {missing}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


_lib_check_cache = {}


def requires_python_libs(*libs):
    """
    A decorator to check if all required Python library dependencies are installed
    before calling the decorated function. If a library is missing, it raises
    an ImportError with details about the missing libraries. Caches check results
    to avoid repeated imports.

    :param libs: A list of library names (as strings) to check for.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing_libs = []
            for lib in libs:
                # Check if the library check result is cached
                if lib in _lib_check_cache:
                    if not _lib_check_cache[lib]:
                        missing_libs.append(lib)
                else:
                    try:
                        __import__(lib)
                        _lib_check_cache[lib] = True
                    except ImportError:
                        _lib_check_cache[lib] = False
                        missing_libs.append(lib)
            if missing_libs:
                missing = ", ".join(missing_libs)
                raise ImportError(
                    f"Missing Python library dependencies: {missing}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
