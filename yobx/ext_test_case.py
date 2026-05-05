"""
The module contains the main class ``ExtTestCase`` which adds
specific functionalities to this project.
"""

import glob
import itertools
import logging
import os
import re
import shutil
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout, contextmanager
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy
from numpy.testing import assert_allclose
from .pv_version import PvVersion

BOOLEAN_VALUES = (1, "1", True, "True", "true", "TRUE")


def _msg(msg: Union[Callable[[], str], str], add_bracket: bool = True) -> str:
    if add_bracket:
        m = _msg(msg, add_bracket=False)
        if m:
            if "\n" in m:
                return f"\n----\n{m}\n---\n"
            return f" ({m})"
        return ""
    if callable(msg):
        return msg()
    return msg or ""


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform == "linux"


def skipif_ci_windows(msg: str) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if is_windows():
        msg = f"Test does not work on azure pipeline (Windows). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_linux(msg: str) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Linux`."""
    if is_linux():
        msg = f"Takes too long (Linux). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_apple(msg: str) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if is_apple():
        msg = f"Test does not work on azure pipeline (Apple). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def unit_test_going() -> bool:
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """
    if not isinstance(warns, (tuple, list)):
        warns = (warns,)
    new_list = []
    for w in warns:
        if w == "TracerWarning":
            from torch.jit import TracerWarning

            new_list.append(TracerWarning)
        else:
            new_list.append(w)
    warns = tuple(new_list)

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:  # pragma: no cover
            pass
        return call_f

    return wrapper


def ignore_errors(errors: Union[Exception, Tuple[Exception]]) -> Callable:
    """
    Catches exception, skip the test if the error is expected sometimes.

    :param errors: errors to ignore
    """

    def wrapper(fct):
        if errors is None:
            raise AssertionError(f"errors cannot be None for '{fct}'.")

        def call_f(self):
            try:
                return fct(self)
            except errors as e:
                raise unittest.SkipTest(  # noqa: B904
                    f"expecting error {e.__class__.__name__}: {e}"
                )

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:  # pragma: no cover
            pass
        return call_f

    return wrapper


def hide_stdout(f: Optional[Callable] = None) -> Callable:
    """
    Catches warnings, hides standard output.
    The function may be disabled by setting ``UNHIDE=1``
    before running the unit test.

    :param f: the function is called with the stdout as an argument
    """

    def wrapper(fct):
        def call_f(self):
            if os.environ.get("UNHIDE", "") in (1, "1", "True", "true"):
                fct(self)
                return
            st = StringIO()
            with redirect_stdout(st), warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, DeprecationWarning))
                try:
                    fct(self)
                except AssertionError as e:  # pragma: no cover
                    if "torch is not recent enough, file" in str(e):
                        raise unittest.SkipTest(str(e))  # noqa: B904
                    raise
            if f is not None:
                f(st.getvalue())
            return None

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:  # pragma: no cover
            pass
        return call_f

    return wrapper


def long_test(msg: Optional[Union[Callable[[], str], str]] = None) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if os.environ.get("LONGTEST", "0") in ("0", 0, False, "False", "false"):
        msg = f"Skipped (set LONGTEST=1 to run it. {_msg(msg)}"
        return unittest.skip(msg)
    return lambda x: x


def never_test(msg: Optional[Union[Callable[[], str], str]] = None) -> Callable:
    """Skips a unit test."""
    if os.environ.get("NEVERTEST", "0") in ("0", 0, False, "False", "false"):
        msg = f"Skipped (set NEVERTEST=1 to run it. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def measure_time(
    stmt: Union[str, Callable],
    context: Optional[Dict[str, Any]] = None,
    repeat: int = 10,
    number: int = 50,
    warmup: int = 1,
    div_by_number: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Union[str, int, float]]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string or callable
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param warmup: number of iteration to do before starting the
        real measurement
    :param div_by_number: divide by the number of executions
    :param max_time: execute the statement until the total goes
        beyond this time (approximately), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    .. runpython::
        :showcode:

        from pprint import pprint
        from math import cos
        from yobx.ext_test_case import measure_time

        res = measure_time(lambda: cos(0.5))
        pprint(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(f"stmt is not callable or a string but is of type {type(stmt)!r}.")
    if context is None:
        context = {}

    if isinstance(stmt, str):
        tim = Timer(stmt, globals=context)
    else:
        tim = Timer(stmt)

    if warmup > 0:
        warmup_time = tim.timeit(warmup)
    else:
        warmup_time = 0

    if max_time is not None:
        if not div_by_number:
            raise ValueError("div_by_number must be set to True of max_time is defined.")
        i = 1
        total_time = 0.0
        results = []
        while True:
            for j in (1, 2):
                number = i * j
                time_taken = tim.timeit(number)
                results.append((number, time_taken))
                total_time += time_taken
                if total_time >= max_time:
                    break
            if total_time >= max_time:
                break
            ratio = (max_time - total_time) / total_time
            ratio = max(ratio, 1)
            i = int(i * ratio)

        res = numpy.array(results)
        tw = res[:, 0].sum()
        ttime = res[:, 1].sum()
        mean = ttime / tw
        ave = res[:, 1] / res[:, 0]
        dev = (((ave - mean) ** 2 * res[:, 0]).sum() / tw) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(ave),
            max_exec=numpy.max(ave),
            repeat=1,
            number=tw,
            ttime=ttime,
        )
    else:
        res = numpy.array(tim.repeat(repeat=repeat, number=number))
        if div_by_number:
            res /= number

        mean = numpy.mean(res)
        dev = numpy.mean(res**2)
        dev = (dev - mean**2) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(res),
            max_exec=numpy.max(res),
            repeat=repeat,
            number=number,
            ttime=res.sum(),
        )

    if "values" in context:
        if hasattr(context["values"], "shape"):
            mes["size"] = context["values"].shape[0]
        else:
            mes["size"] = len(context["values"])
    else:
        mes["context_size"] = sys.getsizeof(context)
    mes["warmup_time"] = warmup_time
    return mes


def statistics_on_folder(
    folder: Union[str, List[str]], pattern: str = ".*[.]((py|rst))$", aggregation: int = 0
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Computes statistics on files in a folder.

    :param folder: folder or folders to investigate
    :param pattern: file pattern
    :param aggregation: show the first subfolders
    :return: list of dictionaries

    .. runpython::
        :showcode:
        :toggle:

        import os
        import pprint
        from yobx.ext_test_case import statistics_on_folder, __file__

        pprint.pprint(statistics_on_folder(os.path.dirname(__file__)))

    Aggregated:

    .. runpython::
        :showcode:
        :toggle:

        import os
        import pprint
        from yobx.ext_test_case import statistics_on_folder, __file__

        pprint.pprint(statistics_on_folder(os.path.dirname(__file__), aggregation=1))
    """
    if isinstance(folder, list):
        rows = []
        for fold in folder:
            last = fold.replace("\\", "/").split("/")[-1]
            r = statistics_on_folder(fold, pattern=pattern, aggregation=max(aggregation - 1, 0))
            if aggregation == 0:
                rows.extend(r)
                continue
            for line in r:
                line["dir"] = os.path.join(last, line["dir"])
            rows.extend(r)
        return rows

    rows = []
    reg = re.compile(pattern)
    for name in glob.glob("**/*", root_dir=folder, recursive=True):
        if not reg.match(name):
            continue
        if os.path.isdir(os.path.join(folder, name)):
            continue
        n = name.replace("\\", "/")
        spl = n.split("/")
        level = len(spl)
        stat = statistics_on_file(os.path.join(folder, name))
        stat["name"] = name
        stat["files"] = 1
        if aggregation <= 0:
            rows.append(stat)
            continue
        spl = os.path.dirname(name).replace("\\", "/").split("/")
        level = "/".join(spl[:aggregation])
        stat["dir"] = level
        rows.append(stat)
    return rows


def get_figure(ax):
    """Returns the figure of a matplotlib figure."""
    if hasattr(ax, "get_figure"):
        return ax.get_figure()
    if len(ax.shape) == 0:
        return ax.get_figure()
    if len(ax.shape) == 1:
        return ax[0].get_figure()
    if len(ax.shape) == 2:
        return ax[0, 0].get_figure()
    raise RuntimeError(f"Unexpected shape {ax.shape} for axis.")


def has_cuda() -> bool:
    """Returns ``torch.cuda.device_count() > 0``."""
    if not has_torch():
        return False
    import torch

    return torch.cuda.device_count() > 0


def requires_python(version: Tuple[int, ...], msg: str = ""):
    """
    Skips a test if python is too old.

    :param msg: to overwrite the message
    :param version: minimum version
    """
    if sys.version_info[: len(version)] < version:
        return unittest.skip(msg or f"python not recent enough {sys.version_info} < {version}")
    return lambda x: x


def requires_cuda(version: str = "", msg: str = "", memory: int = 0):
    """
    Skips a test if cuda is not available.

    :param version: minimum version
    :param msg: to overwrite the message
    :param memory: minimum number of Gb to run the test
    """
    if not has_torch():
        return unittest.skip(msg or "cuda not installed")

    import torch

    if torch.cuda.device_count() == 0:
        msg = msg or "only runs on CUDA but torch does not have it"
        return unittest.skip(msg or "cuda not installed")

    if version:
        if PvVersion(torch.version.cuda) < PvVersion(version):
            msg = msg or f"CUDA older than {version}"
        return unittest.skip(msg or f"cuda not recent enough {torch.version.cuda} < {version}")

    if memory:
        m = torch.cuda.get_device_properties(0).total_memory / 2**30
        if m < memory:
            msg = msg or f"available memory is not enough {m} < {memory} (Gb)"
            return unittest.skip(msg)

    return lambda x: x


def requires_onnxir(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnx-ir` is not recent enough."""
    try:
        import onnx_ir
    except ImportError:
        return unittest.skip(msg or "onnx-ir not installed")

    if not hasattr(onnx_ir, "__version__"):
        # development version
        return lambda x: x

    if PvVersion(onnx_ir.__version__) < PvVersion(version):
        msg = f"onnx_ir version {onnx_ir.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_sklearn(version: str = "") -> bool:
    "Returns True if torch transformers is available and recent enough."
    try:
        import sklearn
    except (ImportError, AttributeError):
        return False
    if not hasattr(sklearn, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(sklearn.__version__) >= PvVersion(version)


def requires_sklearn(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`scikit-learn` is not recent enough."""
    try:
        import sklearn
    except (AttributeError, ImportError):
        return unittest.skip(msg or "scikit-learn not installed")

    if not hasattr(sklearn, "__version__"):
        return unittest.skip(msg or "scikit-learn not installed")
    if not version:
        return lambda x: x
    if PvVersion(sklearn.__version__) < PvVersion(version):
        msg = f"scikit-learn version {sklearn.__version__} < {version}: {msg}"
        return unittest.skip(msg or f"scikit-learn version < {version}")
    return lambda x: x


def has_torch(version: str = "") -> bool:
    "Returns True if torch transformers is available and recent enough."
    try:
        import torch
    except (ImportError, AttributeError):
        return False
    if not hasattr(torch, "__version__") or os.environ.get("NOTORCH", "0") == "1":
        return False
    if not version:
        return True
    return PvVersion(torch.__version__) >= PvVersion(version)


def has_transformers(version: str = "") -> bool:
    "Returns True if transformers version is available and recent enough."
    try:
        import torch  # noqa: F401
        import transformers
    except (ImportError, AttributeError):
        return False
    if not hasattr(transformers, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(transformers.__version__) >= PvVersion(version)


def requires_torch(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`pytorch` is not recent enough."""
    try:
        import torch
    except (ImportError, AttributeError, NameError):
        return unittest.skip(msg or "torch not installed")

    if not hasattr(torch, "__version__") or os.environ.get("NOTORCH", "0") == "1":
        return unittest.skip(msg or "torch not installed")
    if not version:
        return lambda x: x

    if PvVersion(torch.__version__) < PvVersion(version):
        msg = f"torch version {torch.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_tensorflow(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`tensorflow` is not recent enough."""
    try:
        import tensorflow
    except (ImportError, AttributeError):
        return unittest.skip(msg or "tensorflow not installed")

    if not version:
        return lambda x: x

    if PvVersion(tensorflow.__version__) < PvVersion(version):
        msg = f"tensorflow version {tensorflow.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_jax(version: str = "") -> bool:
    """
    Returns True if JAX is installed and the installed
    version is >= the given version (if specified).
    """
    try:
        import jax  # noqa: F401
    except (ImportError, AttributeError):
        return False
    if not hasattr(jax, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(jax.__version__) >= PvVersion(version)


def requires_jax(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`jax` or :mod:`jax.experimental.jax2tf` is not available."""
    try:
        import jax
        from jax.experimental import jax2tf  # noqa: F401
    except (ImportError, AttributeError):
        return unittest.skip(msg or "jax[tensorflow] not installed")

    try:
        import tensorflow  # noqa: F401
    except (ImportError, AttributeError):
        return unittest.skip(msg or "tensorflow not installed (required for jax2tf)")

    if not version:
        return lambda x: x

    if PvVersion(jax.__version__) < PvVersion(version):
        msg = f"jax version {jax.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_xgboost(version: str = "") -> bool:
    "Returns True if XGBoost is installed and its version is high enough."
    try:
        import xgboost
    except (ImportError, AttributeError):
        return False
    if not hasattr(xgboost, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(xgboost.__version__) >= PvVersion(version)


def requires_xgboost(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`xgboost` is not recent enough."""
    try:
        import xgboost
    except (AttributeError, ImportError):
        return unittest.skip(msg or "xgboost not installed")

    if not hasattr(xgboost, "__version__"):
        return unittest.skip(msg or "xgboost not installed")

    if not version:
        return lambda x: x

    if PvVersion(xgboost.__version__) < PvVersion(version):
        msg = f"xgboost version {xgboost.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_category_encoders(version: str = "") -> bool:
    "Returns True if category_encoders is installed and its version is high enough."
    try:
        import category_encoders
    except (ImportError, AttributeError):
        return False
    if not hasattr(category_encoders, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(category_encoders.__version__) >= PvVersion(version)


def requires_category_encoders(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`category_encoders` is not recent enough."""
    try:
        import category_encoders
    except (AttributeError, ImportError):
        return unittest.skip(msg or "category_encoders not installed (1)")

    if not hasattr(category_encoders, "__version__"):
        return unittest.skip(msg or "category_encoders not installed (2)")

    if not version:
        return lambda x: x

    if PvVersion(category_encoders.__version__) < PvVersion(version):
        msg = f"category_encoders version {category_encoders.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_lightgbm(version: str = "") -> bool:
    "Returns True if LightGBM is installed and its version is high enough."
    try:
        import lightgbm
    except (ImportError, AttributeError):
        return False
    if not hasattr(lightgbm, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(lightgbm.__version__) >= PvVersion(version)


def requires_lightgbm(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`lightgbm` is not recent enough."""
    try:
        import lightgbm
    except (AttributeError, ImportError):
        return unittest.skip(msg or "lightgbm not installed")

    if not hasattr(lightgbm, "__version__"):
        return unittest.skip(msg or "lightgbm not installed")

    if not version:
        return lambda x: x

    if PvVersion(lightgbm.__version__) < PvVersion(version):
        msg = f"lightgbm version {lightgbm.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_sksurv(version: str = "") -> bool:
    "Returns True if :epkg:`scikit-survival` is available and recent enough."
    try:
        import sksurv
    except (ImportError, AttributeError):
        return False
    if not hasattr(sksurv, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(sksurv.__version__) >= PvVersion(version)


def requires_sksurv(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`scikit-survival` is not recent enough."""
    try:
        import sksurv
    except (AttributeError, ImportError):
        return unittest.skip(msg or "scikit-survival not installed")

    if not hasattr(sksurv, "__version__"):
        return unittest.skip(msg or "scikit-survival not installed")

    if not version:
        return lambda x: x

    if PvVersion(sksurv.__version__) < PvVersion(version):
        msg = f"scikit-survival version {sksurv.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_statsmodels(version: str = "") -> bool:
    "Returns True if :epkg:`statsmodels` is available and recent enough."
    try:
        import statsmodels
    except (ImportError, AttributeError):
        return False
    if not hasattr(statsmodels, "__version__"):
        return False
    if not version:
        return True
    return PvVersion(statsmodels.__version__) >= PvVersion(version)


def requires_statsmodels(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`statsmodels` is not installed or not recent enough."""
    try:
        import statsmodels
    except (AttributeError, ImportError):
        return unittest.skip(msg or "statsmodels not installed")

    if not hasattr(statsmodels, "__version__"):
        return unittest.skip(msg or "statsmodels not installed")

    if not version:
        return lambda x: x

    if PvVersion(statsmodels.__version__) < PvVersion(version):
        msg = f"statsmodels version {statsmodels.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_matplotlib(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`pytorch` is not recent enough."""
    try:
        import matplotlib
    except ImportError:
        return unittest.skip(msg or "matplotlib not installed")

    if not version:
        return lambda x: x

    if PvVersion(matplotlib.__version__) < PvVersion(version):
        msg = f"matplotlib version {matplotlib.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_cairosvg(msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`cairosvg` is not installed."""
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        return unittest.skip(msg or "cairosvg not installed")
    return lambda x: x


def requires_numpy(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`numpy` is not recent enough."""
    try:
        import numpy
    except ImportError:
        return unittest.skip(msg or "numpy not installed")

    if PvVersion(numpy.__version__) < PvVersion(version):
        msg = f"numpy version {numpy.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_pandas(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`pandas` is not installed or not recent enough."""
    try:
        import pandas
    except ImportError:
        return unittest.skip(msg or "pandas not installed")

    if not hasattr(pandas, "__version__"):
        return lambda x: x
    if not version:
        return lambda x: x
    if PvVersion(pandas.__version__) < PvVersion(version):
        msg = f"pandas version {pandas.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_transformers(
    version: str, msg: str = "", or_older_than: Optional[str] = None
) -> Callable:
    """Skips a unit test if :epkg:`transformers` is not recent enough."""
    if not has_torch():
        return unittest.skip(msg or "torch not installed")
    try:
        import transformers
    except (AttributeError, ImportError, NameError):
        return unittest.skip(msg or "transformers not installed")

    if not version:
        return lambda x: x

    v = PvVersion(transformers.__version__.replace(".dev0", ""))
    if v < PvVersion(version):
        msg = f"transformers version {transformers.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    if or_older_than and v > PvVersion(or_older_than):
        msg = (
            f"transformers version {or_older_than} < "
            f"{transformers.__version__} < {version}: {msg}"
        )
        return unittest.skip(msg)
    return lambda x: x


def requires_diffusers(
    version: str, msg: str = "", or_older_than: Optional[str] = None
) -> Callable:
    """Skips a unit test if :epkg:`transformers` is not recent enough."""

    try:
        import torch  # noqa: F401
        import diffusers
    except (ImportError, AttributeError):
        return unittest.skip(msg or "diffusers not installed")

    if not version:
        return lambda x: x

    v = PvVersion(diffusers.__version__)
    if v < PvVersion(version):
        msg = f"diffusers version {diffusers.__version__} < {version} {msg}"
        return unittest.skip(msg)
    if or_older_than and v > PvVersion(or_older_than):
        msg = f"diffusers version {or_older_than} < {diffusers.__version__} < {version} {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_onnxscript(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxscript` is not recent enough."""
    try:
        import onnxscript
    except ImportError:
        return unittest.skip(msg or "onnxscript not installed")

    if not version:
        return lambda x: x

    if not hasattr(onnxscript, "__version__"):
        # development version
        return lambda x: x

    if PvVersion(onnxscript.__version__) < PvVersion(version):
        msg = f"onnxscript version {onnxscript.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_onnxscript(version: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxscript` is not recent enough."""
    try:
        import onnxscript
    except ImportError:
        return False

    if not hasattr(onnxscript, "__version__"):
        # development version
        return True

    if not version:
        return True

    if PvVersion(onnxscript.__version__) < PvVersion(version):
        return False
    return True


def has_onnx_ir(version: str = "") -> Callable:
    """Skips a unit test if `ir-py` is not recent enough."""
    try:
        import onnx_ir
    except ImportError:
        return False

    if not version:
        return True

    if not hasattr(onnx_ir, "__version__"):
        # development version
        return True

    if PvVersion(onnx_ir.__version__) < PvVersion(version):
        return False
    return True


def has_onnx_shape_inference(version: str = "") -> Callable:
    """Skips a unit test if `onnx-shape-inference`."""
    try:
        import onnx_shape_inference
    except ImportError:
        return False

    if not version:
        return True

    if not hasattr(onnx_shape_inference, "__version__"):
        # development version
        return True

    if PvVersion(onnx_shape_inference.__version__) < PvVersion(version):
        return False
    return True


def requires_spox(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`spox` is not recent enough."""
    try:
        import spox
    except ImportError:
        return unittest.skip(msg or "spox not installed")

    if not version:
        return lambda x: x

    if not hasattr(spox, "__version__"):
        # development version
        return lambda x: x

    if PvVersion(spox.__version__) < PvVersion(version):
        msg = f"spox version {spox.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_spox(version: str = "") -> bool:
    """Returns ``True`` if :epkg:`spox` is installed and recent enough."""
    try:
        import spox
    except (ImportError, AttributeError):
        return False

    if not version:
        return True

    if not hasattr(spox, "__version__"):
        # development version
        return True

    return PvVersion(spox.__version__) >= PvVersion(version)


def has_tensorflow(version: str = "") -> bool:
    """Returns ``True`` if :epkg:`tensorflow` is installed and recent enough."""
    try:
        import tensorflow
    except (ImportError, AttributeError):
        return False

    if not hasattr(tensorflow, "__version__"):
        return False

    if not version:
        return True

    return PvVersion(tensorflow.__version__) >= PvVersion(version)


def has_tf2onnx(version: str = "") -> bool:
    """Returns ``True`` if :epkg:`tf2onnx` is installed and recent enough."""
    try:
        import tf2onnx
    except (ImportError, AttributeError):
        return False

    if not version:
        return True

    if not hasattr(tf2onnx, "__version__"):
        # development version
        return True

    return PvVersion(tf2onnx.__version__) >= PvVersion(version)


def requires_tf2onnx(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`tf2onnx` is not installed or not recent enough."""
    try:
        import tf2onnx
    except (ImportError, AttributeError):
        return unittest.skip(msg or "tf2onnx not installed")

    if not version:
        return lambda x: x

    if not hasattr(tf2onnx, "__version__"):
        # development version
        return lambda x: x

    if PvVersion(tf2onnx.__version__) < PvVersion(version):
        msg = f"tf2onnx version {tf2onnx.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_onnxruntime(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxruntime` is not recent enough."""
    try:
        import onnxruntime
    except ImportError:
        return unittest.skip(msg or "onnxruntime not installed")

    if PvVersion(onnxruntime.__version__) < PvVersion(version):
        msg = f"onnxruntime version {onnxruntime.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_onnxruntime(version: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxruntime` is not recent enough."""
    try:
        import onnxruntime
    except ImportError:
        return False

    if not hasattr(onnxruntime, "__version__"):
        # development version
        return True

    if version and PvVersion(onnxruntime.__version__) < PvVersion(version):
        return False
    return True


def has_onnxruntime_training(push_back_batch: bool = False):
    """Tells if onnxruntime_training is installed."""
    try:
        from onnxruntime import training
    except ImportError:
        # onnxruntime not training
        training = None
    if training is None:
        return False

    if push_back_batch:
        try:
            from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
        except ImportError:
            return False

        if not hasattr(OrtValueVector, "push_back_batch"):
            return False
    return True


def has_onnxruntime_genai():
    """Tells if onnxruntime_genai is installed."""
    try:
        import onnxruntime_genai  # noqa: F401

        return True
    except ImportError:
        # onnxruntime not training
        return False


def requires_onnxruntime_training(
    push_back_batch: bool = False, ortmodule: bool = False, msg: str = ""
) -> Callable:
    """Skips a unit test if :epkg:`onnxruntime` is not onnxruntime_training."""
    try:
        from onnxruntime import training
    except ImportError:
        # onnxruntime not training
        training = None
    if training is None:
        msg = msg or "onnxruntime_training is not installed"
        return unittest.skip(msg)

    if push_back_batch:
        try:
            from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
        except ImportError:
            msg = msg or "OrtValue has no method push_back_batch"
            return unittest.skip(msg)

        if not hasattr(OrtValueVector, "push_back_batch"):
            msg = msg or "OrtValue has no method push_back_batch"
            return unittest.skip(msg)
    if ortmodule:
        try:
            import onnxruntime.training.ortmodule  # noqa: F401
        except (AttributeError, ImportError):  # pragma: no cover
            msg = msg or "ortmodule is missing in onnxruntime-training"
            return unittest.skip(msg)
    return lambda x: x


def has_litert(version: str = "") -> bool:
    """Returns True if ``ai_edge_litert`` or ``tensorflow.lite`` is available."""
    try:
        import ai_edge_litert  # noqa: F401

        if not version:
            return True
        return PvVersion(ai_edge_litert.__version__) >= PvVersion(version)
    except (ImportError, AttributeError):
        pass
    try:
        import tensorflow as tf  # noqa: F401

        if not version:
            return True
        return PvVersion(tf.__version__) >= PvVersion(version)
    except (ImportError, AttributeError):
        return False


def requires_litert(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if neither :epkg:`ai_edge_litert` nor
    :epkg:`tensorflow` (which ships ``tf.lite``) is installed."""
    if not has_litert(version):
        return unittest.skip(
            msg or "ai_edge_litert / tensorflow not installed or version too old"
        )
    return lambda x: x


def requires_onnx(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnx` is not recent enough."""
    try:
        import onnx
    except ImportError:
        return unittest.skip(msg or "onnx not installed")

    if PvVersion(onnx.__version__) < PvVersion(version):
        msg = f"onnx version {onnx.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_jax2onnx(version: str = "") -> bool:
    "Returns True if :epkg:`jax2onnx` is installed and its version is high enough."
    try:
        import jax2onnx  # noqa: F401
    except (ImportError, AttributeError):
        return False
    if not version:
        return True
    try:
        from importlib.metadata import version as _meta_version

        installed = _meta_version("jax2onnx")
    except Exception:
        return True
    return PvVersion(installed) >= PvVersion(version)


def requires_jax2onnx(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`jax2onnx` is not installed or not recent enough."""
    try:
        import jax2onnx  # noqa: F401
    except (ImportError, AttributeError):
        return unittest.skip(msg or "jax2onnx not installed")

    if not version:
        return lambda x: x

    try:
        from importlib.metadata import version as _meta_version

        installed = _meta_version("jax2onnx")
    except Exception:
        return lambda x: x

    if PvVersion(installed) < PvVersion(version):
        msg = f"jax2onnx version {installed} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def statistics_on_file(filename: str) -> Dict[str, Union[int, float, str]]:
    """
    Computes statistics on a file.

    .. runpython::
        :showcode:

        import pprint
        from yobx.ext_test_case import statistics_on_file, __file__

        pprint.pprint(statistics_on_file(__file__))
    """
    assert os.path.exists(filename), f"File {filename!r} does not exists."

    ext = os.path.splitext(filename)[-1]
    if ext not in {".py", ".rst", ".md", ".txt"}:
        size = os.stat(filename).st_size
        return {"size": size}
    alpha = set("abcdefghijklmnopqrstuvwxyz0123456789")
    with open(filename, "r", encoding="utf-8") as f:
        n_line = 0
        n_ch = 0
        for line in f.readlines():
            s = line.strip("\n\r\t ")
            if s:
                n_ch += len(s.replace(" ", ""))
                ch = set(s.lower()) & alpha
                if ch:
                    # It avoid counting line with only a bracket, a comma.
                    n_line += 1

    stat = dict(lines=n_line, chars=n_ch, ext=ext)
    if ext != ".py":
        return stat
    # add statistics on python syntax?
    return stat


class ExtTestCase(unittest.TestCase):
    """
    Inherits from :class:`unittest.TestCase` and adds specific comprison
    functions and other helper.
    """

    _warns: List[Tuple[str, int, Warning]] = []

    def shortDescription(self):
        # To remove annoying display on the screen every time verbosity is enabled.
        return None

    def unit_test_going(self) -> bool:
        """
        Enables a flag telling the script is running while testing it.
        Avois unit tests to be very long.
        """
        return unit_test_going()

    @property
    def verbose(self) -> int:
        "Returns the value of environment variable ``VERBOSE``."
        return int(os.environ.get("VERBOSE", "0"))

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        unittest.TestCase.setUpClass()

    @classmethod
    def ort(cls) -> unittest.__class__:
        import onnxruntime

        return onnxruntime

    @classmethod
    def to_onnx(self, *args, **kwargs) -> "ModelProto":  # noqa: F821
        from yobx.torch.interpreter import to_onnx

        return to_onnx(*args, **kwargs)

    def print_model(self, model: "ModelProto"):  # noqa: F821
        "Prints a ModelProto"
        from yobx.helpers.onnx_helper import pretty_onnx

        print(pretty_onnx(model))

    def print_onnx(self, model: "ModelProto"):  # noqa: F821
        "Prints a ModelProto"
        from yobx.helpers.onnx_helper import pretty_onnx

        print(pretty_onnx(model))

    def get_dump_file(self, name: str, folder: Optional[str] = None) -> str:
        """Returns a filename to dump a model."""
        if folder is None:
            folder = "dump_test"
        if folder and not os.path.exists(folder):
            os.mkdir(folder)
        return os.path.join(folder, name)

    def get_dump_folder(self, folder: str) -> str:
        """Returns a folder."""
        folder = os.path.join("dump_test", folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def clean_dump(self, folder: str = "dump_test"):
        """Cleans this folder."""
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    def dump_onnx(self, name: str, proto: Any, folder: Optional[str] = None) -> str:
        """Dumps an onnx file."""
        fullname = self.get_dump_file(name, folder=folder)
        with open(fullname, "wb") as f:
            f.write(proto.SerializeToString())
        return fullname

    def dump_text(self, name: str, text: str, folder: Optional[str] = None) -> str:
        """Dumps text in a file."""
        fullname = self.get_dump_file(name, folder=folder)
        with open(fullname, "w") as f:
            f.write(text)
        return fullname

    def assertExists(self, name, msg: Optional[Union[Callable[[], str], str]] = None):
        """Checks the existing of a file."""
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists{_msg(msg)}.")

    def assertGreater(self, a, b, msg: Optional[Union[Callable[[], str], str]] = None):
        """In the name"""
        if a <= b:
            raise AssertionError(f"{a} <= {b}, a not strictly greater than b{_msg(msg)}")

    def assertGreaterOrEqual(self, a, b, msg: Optional[Union[Callable[[], str], str]] = None):
        """In the name"""
        if a < b:
            raise AssertionError(f"{a} < {b}, a not greater or equal than b{_msg(msg)}")

    def assertLess(self, a, b, msg: Optional[Union[Callable[[], str], str]] = None):
        """In the name"""
        if a >= b:
            raise AssertionError(f"{a} >= {b}, a not strictly less than b{_msg(msg)}")

    def assertLessOrEqual(self, a, b, msg: Optional[Union[Callable[[], str], str]] = None):
        """In the name"""
        if a > b:
            raise AssertionError(f"{a} > {b}, a not less or equal than b{_msg(msg)}")

    def assertInOr(
        self,
        tofind: Tuple[str, ...],
        text: str,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        for tof in tofind:
            if tof in text:
                return
        raise AssertionError(
            f"Unable to find one string in the list {tofind!r} in {text!r}{_msg(msg)}"
        )

    def assertIn(
        self, tofind: str, text: str, msg: Optional[Union[Callable[[], str], str]] = None
    ):
        if tofind in text:
            return
        raise AssertionError(
            f"Unable to find the list of strings {tofind!r} in {text!r}{_msg(msg)}"
        )

    def assertHasAttr(
        self, obj: Any, name: str, msg: Optional[Union[Callable[[], str], str]] = None
    ):
        assert hasattr(
            obj, name
        ), f"Unable to find attribute {name!r} in object type {type(obj)}{_msg(msg)}"

    def assertSetContained(self, set1, set2, msg: Optional[Union[Callable[[], str], str]] = None):
        "Checks that ``set1`` is contained in ``set2``."
        set1 = set(set1)
        set2 = set(set2)
        if set1 & set2 != set1:
            raise AssertionError(f"Set {set2} does not contain set {set1}{_msg(msg)}")

    def assertEqualArrays(
        self,
        expected: Sequence[numpy.ndarray],
        value: Sequence[numpy.ndarray],
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        """In the name"""
        self.assertEqual(len(expected), len(value), msg=msg)
        for a, b in zip(expected, value):
            self.assertEqualArray(a, b, atol=atol, rtol=rtol, msg=msg)

    def to_numpy(self, tensor):
        """Converts a :class:`torch.Tensor` to :class:`numpy.ndarray`."""
        try:
            return tensor.detach().cpu().numpy()
        except TypeError:
            # We try with ml_dtypes
            pass

        import ml_dtypes
        import torch

        conv = {torch.bfloat16: ml_dtypes.bfloat16}
        assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
        return tensor.detach().to(torch.float32).cpu().numpy().astype(conv[tensor.dtype])

    def assertEqualArray(
        self,
        expected: Any,
        value: Any,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        """In the name"""
        if hasattr(expected, "detach") and hasattr(value, "detach"):
            self.assertEqual(expected.dtype, value.dtype, msg=msg)
            self.assertEqual(expected.shape, value.shape, msg=msg)

            import torch

            try:
                torch.testing.assert_close(value, expected, atol=atol, rtol=rtol)
            except AssertionError as e:
                expected_max = torch.abs(expected).max()
                expected_value = torch.abs(value).max()
                rows = [
                    f"{msg}\n{e}" if msg else str(e),
                    f"expected max value={expected_max}",
                    f"expected computed value={expected_value}",
                    _msg(msg, False),
                ]
                raise AssertionError("\n".join(rows))  # noqa: B904
            return

        if hasattr(expected, "detach"):
            expected = self.to_numpy(expected.detach().cpu())
        if hasattr(value, "detach"):
            value = self.to_numpy(value.detach().cpu())
        self.assertEqual(expected.dtype, value.dtype, msg=msg)
        self.assertEqual(expected.shape, value.shape, msg=msg)

        try:
            assert_allclose(desired=expected, actual=value, atol=atol, rtol=rtol)
        except AssertionError as e:
            expected_max = numpy.abs(expected).max()
            expected_value = numpy.abs(value).max()
            tte = expected.astype(int) if expected.dtype == numpy.bool_ else expected
            ttv = value.astype(int) if value.dtype == numpy.bool_ else value
            rows = [
                f"{msg}\n{e}" if msg else str(e),
                f"expected max value={expected_max}",
                f"expected computed value={expected_value}\n",
                f"ratio={tte / ttv}\ndiff={tte - ttv}",
                _msg(msg, False),
            ]
            raise AssertionError("\n".join(rows))  # noqa: B904

    def assertEqualDataFrame(
        self, d1, d2, msg: Optional[Union[Callable[[], str], str]] = None, **kwargs
    ):
        """
        Checks that two dataframes are equal.
        Calls :func:`pandas.testing.assert_frame_equal`.
        """
        from pandas.testing import assert_frame_equal

        if msg:
            try:
                assert_frame_equal(d1, d2, **kwargs)
            except AssertionError as e:
                raise AssertionError(_msg(msg, False)) from e
        else:
            assert_frame_equal(d1, d2, **kwargs)

    def assertEqualTrue(self, value: Any, msg: Optional[Union[Callable[[], str], str]] = None):
        if value is True:
            return
        raise AssertionError(f"value is not True: {value!r}{_msg(msg)}")

    def assertEqual(
        self, expected: Any, value: Any, msg: Optional[Union[Callable[[], str], str]] = None
    ):
        """Overwrites the error message to get a more explicit message about what is what."""
        try:
            super().assertEqual(expected, value)
        except AssertionError as e:
            raise AssertionError(  # noqa: B904
                f"expected is {expected!r}, value is {value!r}({_msg(msg)}\n{e}"
            )

    def assertTrue(self, cond: bool, msg: Optional[Union[Callable[[], str], str]] = None):
        """Overwrites the error message to get a more explicit message about what is what."""
        try:
            super().assertTrue(cond)
        except AssertionError as e:
            raise AssertionError(  # noqa: B904
                f"condition is False when it should be True({_msg(msg)}\n{e}"
            )

    def assertFalse(self, cond: bool, msg: Optional[Union[Callable[[], str], str]] = None):
        """Overwrites the error message to get a more explicit message about what is what."""
        try:
            super().assertFalse(cond)
        except AssertionError as e:
            raise AssertionError(  # noqa: B904
                f"condition is True when it should be False({_msg(msg)}\n{e}"
            )

    def assertEqualAny(
        self,
        expected: Any,
        value: Any,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        if isinstance(expected, (int, float, str)):
            self.assertEqual(expected, value, msg=msg)
        elif expected is None:
            self.assertEqual(expected, value, msg=msg)
        elif isinstance(expected, (tuple, list, dict)):
            self.assertIsInstance(value, type(expected), msg=msg)
            self.assertEqual(len(expected), len(value), msg=msg)
            if isinstance(expected, dict):
                for k in expected:
                    self.assertIn(k, value, msg=msg)
                    self.assertEqualAny(expected[k], value[k], msg=msg, atol=atol, rtol=rtol)
            else:
                for e, g in zip(expected, value):
                    self.assertEqualAny(e, g, msg=msg, atol=atol, rtol=rtol)
        elif hasattr(expected, "shape"):
            self.assertEqual(type(expected), type(value), msg=msg)
            self.assertEqualArray(expected, value, msg=msg, atol=atol, rtol=rtol)
        elif expected.__class__.__name__ in ("StaticCache", "DynamicCache"):
            import transformers

            self.assertIsInstance(
                value,
                (transformers.cache_utils.DynamicCache, transformers.cache_utils.StaticCache),
            )
            self.assertEqual(len(expected.layers), len(value.layers))
            self.assertEqual(
                [type(layer) for layer in expected.layers],
                [type(layer) for layer in value.layers],
            )
            self.assertEqualAny(
                [(layer.keys, layer.values) for layer in expected.layers],
                [(layer.keys, layer.values) for layer in value.layers],
            )
        elif expected.__class__.__name__ == "EncoderDecoderCache":
            import transformers

            self.assertIsInstance(value, transformers.cache_utils.EncoderDecoderCache)
            self.assertEqualAny(expected.self_attention_cache, value.self_attention_cache)
            self.assertEqualAny(expected.cross_attention_cache, value.cross_attention_cache)
        else:
            raise AssertionError(
                f"Comparison not implemented for types {type(expected)} and {type(value)}"
            )

    def assertEqualArrayAny(
        self,
        expected: Any,
        value: Any,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        if isinstance(expected, (tuple, list, dict)):
            self.assertIsInstance(value, type(expected), msg=msg)
            self.assertEqual(len(expected), len(value), msg=msg)
            if isinstance(expected, dict):
                for k in expected:
                    self.assertIn(k, value, msg=msg)
                    self.assertEqualArrayAny(expected[k], value[k], msg=msg, atol=atol, rtol=rtol)
            else:
                excs = []
                for i, (e, g) in enumerate(zip(expected, value)):
                    try:
                        self.assertEqualArrayAny(e, g, msg=msg, atol=atol, rtol=rtol)
                    except AssertionError as e:
                        excs.append(f"Error at position {i} due to {e}")
                if excs:
                    msg_ = "\n".join(excs)
                    msg = f"{msg}\n{msg_}" if msg else msg_
                    raise AssertionError(f"Found {len(excs)} discrepancies\n{msg}")
        elif expected.__class__.__name__ in ("DynamicCache", "StaticCache"):
            atts = {"key_cache", "value_cache"}
            self.assertEqualArrayAny(
                {k: expected.__dict__.get(k, None) for k in atts},
                {k: value.__dict__.get(k, None) for k in atts},
                atol=atol,
                rtol=rtol,
            )
        elif isinstance(expected, (int, float, str)):
            self.assertEqual(expected, value, msg=msg)
        elif hasattr(expected, "shape"):
            self.assertEqual(type(expected), type(value), msg=msg)
            self.assertEqualArray(expected, value, msg=msg, atol=atol, rtol=rtol)
        elif expected is None:
            assert value is None, f"Expected is None but value is of type {type(value)}"
        else:
            raise AssertionError(
                f"Comparison not implemented for types {type(expected)} and {type(value)}"
            )

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        """In the name"""
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol, msg=msg)

    def check_ort(
        self, onx: Union["onnx.ModelProto", str]  # noqa: F821
    ) -> "onnxruntime.InferenceSession":  # noqa: F821
        return self._check_with_ort(onx, cpu=True)

    def _check_with_ort(
        self, proto: Union["onnx.ModelProto", str], cpu: bool = False  # noqa: F821
    ) -> "onnxruntime.InferenceSession":  # noqa: F821
        from onnxruntime import InferenceSession, get_available_providers
        from .container.export_artifact import ExportArtifact

        if isinstance(proto, ExportArtifact):
            proto = proto.proto

        providers = ["CPUExecutionProvider"]
        if not cpu and "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        return InferenceSession(
            proto.SerializeToString() if hasattr(proto, "SerializeToString") else proto,
            providers=providers,
        )

    def make_inference_session(
        self, onx: Union["onnx.ModelProto", str], cpu: bool = True  # noqa: F821
    ) -> "onnxruntime.InferenceSession":  # noqa: F821
        return self._check_with_ort(onx, cpu=cpu)

    def assertRaise(
        self,
        fct: Callable,
        exc_type: type[Exception],
        look_for: str = "",
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):
        """In the name"""
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}{_msg(msg)}")  # noqa: B904
            if look_for and look_for not in str(e):
                raise AssertionError(  # noqa: B904
                    f"Unexpected exception message {e!r}{_msg(msg)}"
                )
            return
        raise AssertionError("No exception was raised.")  # noqa: B904

    def assertEmpty(self, value: Any, msg: Optional[Union[Callable[[], str], str]] = None):
        """In the name"""
        if value is None:
            return
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}{_msg(msg)}")

    def assertNotEmpty(self, value: Any, msg: Optional[Union[Callable[[], str], str]] = None):
        """In the name"""
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(f"value is empty: {value!r}{_msg(msg)}")

    def assertStartsWith(
        self, prefix: str, full: str, msg: Optional[Union[Callable[[], str], str]] = None
    ):
        """In the name"""
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string {full!r}{_msg(msg)}")

    def assertEndsWith(
        self, suffix: str, full: str, msg: Optional[Union[Callable[[], str], str]] = None
    ):
        """In the name"""
        if not full.endswith(suffix):
            raise AssertionError(f"suffix={suffix!r} does not end string {full!r}{_msg(msg)}")

    def capture(self, fct: Callable) -> Tuple[Any, str, str]:
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
            try:
                res = fct()
            except Exception as e:
                raise AssertionError(
                    f"function {fct} failed, stdout="
                    f"\n{sout.getvalue()}\n---\nstderr=\n{serr.getvalue()}"
                ) from e
        return res, sout.getvalue(), serr.getvalue()

    def tryCall(
        self,
        fct: Callable,
        msg: Optional[Union[Callable[[], str], str]] = None,
        none_if: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Calls the function, catch any error.

        :param fct: function to call
        :param msg: error message to display if failing
        :param none_if: returns None if this substring is found in the error message
        :return: output of *fct*
        """
        try:
            return fct()
        except Exception as e:
            if none_if is not None and none_if in str(e):
                return None
            if msg is None:
                raise
            raise AssertionError(msg) from e

    def _debug(self):
        "Tells if DEBUG=1 is set up."
        return os.environ.get("DEBUG") in BOOLEAN_VALUES

    def subloop(self, *args, verbose: int = 0):
        "Loops over elements and calls :meth:`unittests.TestCase.subTest`."
        if len(args) == 1:
            for it in args[0]:
                with self.subTest(case=it):
                    if verbose:
                        print(f"[subloop] it={it!r}")
                    yield it
        else:
            for it in itertools.product(*args):
                with self.subTest(case=it):
                    if verbose:
                        print(f"[subloop] it={it!r}")
                    yield it

    @contextmanager
    def set_env(self, varname: str, value: str):
        """
        Sets environment variable `varname` to `value`
        and sets it back.
        """
        old_value = os.environ.get(varname, None)
        os.environ[varname] = value
        try:
            yield
        finally:
            os.environ[varname] = old_value or ""

    def string_type(self, *args, **kwargs):
        from .helpers import string_type

        return string_type(*args, **kwargs)

    def max_diff(self, *args, **kwargs):
        from .helpers import max_diff

        return max_diff(*args, **kwargs)

    def assert_conversion_with_ort_on_cpu(
        self,
        onx: "onnx.ModelProto",  # noqa: F821
        expected: Tuple["torch.Tensor", ...],  # noqa: F821
        inputs: Tuple["torch.Tensor", ...],  # noqa: F821
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[Union[Callable[[], str], str]] = None,
        use_python: bool = False,
    ):
        import onnxruntime
        from .reference import ExtendedReferenceEvaluator

        if use_python:
            sess = ExtendedReferenceEvaluator(onx, verbose=10)
            feeds = dict(zip(sess.input_names, [x.detach().numpy() for x in inputs]))
        else:
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            feeds = dict(
                zip([i.name for i in sess.get_inputs()], [x.detach().numpy() for x in inputs])
            )
        got = sess.run(None, feeds)
        if len(got) == 1 and hasattr(expected, "shape"):
            self.assertEqualArray(expected, got[0], atol=atol, rtol=rtol, msg=msg)
        else:
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=atol, rtol=rtol, msg=msg)
