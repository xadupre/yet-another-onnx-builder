import os
import tempfile
import subprocess
import sys
from typing import Optional, List, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from .helpers.dot_helper import to_dot
from .container import ExportArtifact


def get_latest_pypi_version(package_name="yet-another-onnx-builder") -> str:  # pragma: no cover
    """Returns the latest published version."""

    import requests

    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url, timeout=10)

    assert response.status_code == 200, f"Unable to retrieve the version response={response}"
    data = response.json()
    version = data["info"]["version"]
    return version


def update_version_package(version: str, package_name="yet-another-onnx-builder") -> str:
    "Adds dev if the major version is different from the latest published one."
    released = get_latest_pypi_version(package_name)
    shorten_r = ".".join(released.split(".")[:2])
    shorten_v = ".".join(version.split(".")[:2])
    return version if shorten_r == shorten_v else f"{shorten_v}.dev"


def reset_torch_transformers(gallery_conf, fname):  # pragma: no cover
    "Resets torch dynamo for :epkg:`sphinx-gallery`."
    import matplotlib.pyplot as plt
    import torch

    plt.style.use("ggplot")
    torch._dynamo.reset()


def reset_tensorflow(gallery_conf, fname):  # pragma: no cover
    """Suppresses TensorFlow logging for :epkg:`sphinx-gallery`."""
    import logging

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def plot_legend(
    text: str, text_bottom: str = "", color: str = "green", fontsize: int = 15
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Plots a graph with only text (for :epkg:`sphinx-gallery`).

    :param text: legend
    :param text_bottom: text at the bottom
    :param color: color
    :param fontsize: font size
    :return: axis
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot()
    ax.axis([0, 5, 0, 5])
    ax.text(2.5, 4, "END", fontsize=10, horizontalalignment="center")
    ax.text(
        2.5,
        2.5,
        text,
        fontsize=fontsize,
        bbox={"facecolor": color, "alpha": 0.5, "pad": 10},
        horizontalalignment="center",
        verticalalignment="center",
    )
    if text_bottom:
        ax.text(4.5, 0.5, text_bottom, fontsize=7, horizontalalignment="right")
    ax.grid(False)
    ax.set_axis_off()
    return ax


def rotate_align(ax, angle=15, align="right"):
    """Rotates x-label and align them to thr right. Returns ax."""
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_horizontalalignment(align)
    return ax


def save_fig(ax, name: str, **kwargs) -> "matplotlib.axis.Axis":  # noqa: F821
    """Applies ``tight_layout`` and saves the figures. Returns ax."""
    fig = ax.get_figure()
    fig.savefig(name, **kwargs)
    return ax


def title(ax: "plt.axes", title: str) -> "matplotlib.axis.Axis":  # noqa: F821
    "Adds a title to axes and returns them."
    ax.set_title(title)
    return ax


def plot_histogram(
    tensor: np.ndarray,
    ax: Optional["plt.axes"] = None,  # noqa: F821
    bins: int = 30,
    color: str = "orange",
    alpha: float = 0.7,
) -> "matplotlib.axis.Axis":  # noqa: F821
    "Computes the distribution for a tensor."
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.cla()
    ax.hist(tensor, bins=30, color="orange", alpha=0.7)
    ax.set_yscale("log")
    return ax


def _find_in_PATH(prog: str) -> Optional[str]:
    """
    Looks into every path mentioned in ``%PATH%`` a specific file,
    it raises an exception if not found.

    :param prog: program to look for
    :return: path
    """
    sep = ";" if sys.platform.startswith("win") else ":"
    path = os.environ["PATH"]
    for p in path.split(sep):
        f = os.path.join(p, prog)
        if os.path.exists(f):
            return p
    return None


def _run_subprocess(args: List[str], cwd: Optional[str] = None):
    assert not isinstance(args, str), "args should be a sequence of strings, not a string."

    p = subprocess.Popen(
        args, cwd=cwd, shell=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Use communicate() to read stdout and stderr concurrently, avoiding the
    # deadlock that occurs when the subprocess fills its stderr pipe buffer
    # while the main process is blocked waiting for stdout data.
    stdout_data, stderr_data = p.communicate()
    stdout_output = stdout_data.decode(errors="ignore")
    error = stderr_data.decode(errors="ignore")
    raise_exception = any(
        phrase in stdout_output
        for phrase in ("fatal error", "CMake Error", "gmake: ***", "): error C", ": error: ")
    )
    if error and raise_exception:
        raise RuntimeError(
            f"An error was found in the output. The build is stopped."
            f"\n{stdout_output}\n---\n{error}"
        )
    return stdout_output + "\n" + error


def _run_graphviz(filename: str, image: str, engine: str = "dot") -> str:
    """
    Run :epkg:`Graphviz`.

    :param filename: filename which contains the graph definition
    :param image: output image
    :param engine: *dot* or *neato*
    :return: output of graphviz
    """
    ext = os.path.splitext(image)[-1]
    assert ext in {
        ".png",
        ".bmp",
        ".fig",
        ".gif",
        ".ico",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".ps",
        ".svg",
        ".vrml",
        ".tif",
        ".tiff",
        ".wbmp",
    }, f"Unexpected extension {ext!r} for {image!r}."
    assert not sys.platform.startswith("win"), "this is not working on Windows"
    exe = engine
    if os.path.exists(image):
        os.remove(image)
    cmd = [exe, f"-T{ext[1:]}", filename, "-o", image]
    output = _run_subprocess(cmd)
    assert os.path.exists(image), (
        f"Unable to find {image!r}, command line is "
        f"{' '.join(cmd)!r}, Graphviz failed due to\n{output}"
    )
    return output


def draw_graph_graphviz(dot: Union[str, onnx.ModelProto], image: str, engine: str = "dot") -> str:
    """
    Draws a graph using :epkg:`Graphviz`.

    :param dot: dot graph or ModelProto
    :param image: output image file path
    :param engine: *dot* or *neato*
    :return: :epkg:`Graphviz` output

    The function creates a temporary file to store the dot file.
    """
    if isinstance(dot, onnx.ModelProto):
        sdot = to_dot(dot)
    elif isinstance(dot, ExportArtifact):
        sdot = to_dot(dot.get_proto(include_weights=False))
    else:
        if "{" not in dot:
            assert dot.endswith(".onnx"), f"Unexpected file extension for {dot!r}"
            proto = onnx.load(dot)
            sdot = to_dot(proto)
        else:
            sdot = dot
        assert "{" in sdot, f"This string is not a dot string\n{sdot}"
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(sdot.encode("utf-8"))
        fp.close()

        filename = fp.name
        assert os.path.exists(
            filename
        ), f"File {filename!r} cannot be created to store the graph."
        out = _run_graphviz(filename, image, engine=engine)
        assert os.path.exists(
            image
        ), f"Graphviz failed with no reason, {image!r} not found, output is {out}."
        os.remove(filename)
        return out


def plot_dot(
    dot: Union[str, onnx.ModelProto],
    ax: Optional["matplotlib.axis.Axis"] = None,  # noqa: F821
    engine: str = "dot",
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.axis.Axis":  # noqa: F821
    """
    Draws a dot graph into a matplotlib axis.

    :param dot: dot graph or ModelProto
    :param ax: optional matplotlib axis; if None, a new figure and axis are created
    :param engine: *dot* or *neato*
    :param figsize: size of the figure if *ax* is None
    :return: matplotlib axis containing the rendered graph image

    .. plot::

        import matplotlib.pyplot as plt
        import onnx.parser
        from yobx.doc import plot_dot

        model = onnx.parser.parse_model(
            '''
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(four, four)
            }
        ''')

        ax = plot_dot(model)
        ax.set_title("Dummy graph")
        plt.show()
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 1, figsize=figsize)
        clean = True
    else:
        clean = False

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        fp.close()

        exc = None
        try:
            draw_graph_graphviz(dot, fp.name, engine=engine)
        except Exception as e:
            exc = e

        if not exc:
            img = np.asarray(Image.open(fp.name))
            os.remove(fp.name)
            ax.imshow(img)
        elif os.path.exists(fp.name):
            os.remove(fp.name)
            ax.text(
                0.5, 0.5, f"graphviz not available\n{exc}", ha="center", va="center", fontsize=10
            )

    if clean:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()
        ax.get_figure().tight_layout()
    return ax


def draw_graph_mermaid(mermaid: Union[str, onnx.ModelProto], image: str = "") -> Optional[str]:
    """
    Draws a Mermaid flowchart to an image file using the :epkg:`mermaid-py` library.

    :param mermaid: Mermaid flowchart string or :class:`onnx.ModelProto`
    :param image: output image file path (``.svg`` or ``.png``)
    :return: the SVG if image is empty, None otherwise.

    The diagram is rendered via the ``mermaid.ink`` online service through
    :class:`mermaid.Mermaid`.
    """
    from mermaid import Mermaid

    from .helpers.mermaid_helper import to_mermaid

    if isinstance(mermaid, onnx.ModelProto):
        smmd = to_mermaid(mermaid)
    else:
        smmd = mermaid

    renderer = Mermaid(smmd)
    if not image:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "diagram.svg")
            renderer.to_svg(path)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    if image.endswith(".svg"):
        renderer.to_svg(image)
    else:
        renderer.to_png(image)


def plot_text(
    text: str,
    ax: Optional["plt.axes"] = None,  # noqa: F821
    title: str = "",
    fontsize: int = 6,
    line_color_map: Optional[dict] = None,
    default_color: str = "#333333",
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.axis.Axis":  # noqa: F821
    """
    Renders a block of text as a matplotlib figure, with optional per-line
    colour coding based on the first character of each line.

    Typical use-case: rendering a unified diff so that sphinx-gallery captures
    the example as a figure.

    :param text: the text to render (newlines split into rows)
    :param ax: optional matplotlib axis; if *None* a new figure and axis are created
    :param title: optional axis title
    :param fontsize: font size for the rendered text
    :param line_color_map: mapping from a line's first character to a colour
        string (e.g. ``{"+": "green", "-": "red", "@": "blue"}``).
        Lines whose first character is not in the map use *default_color*.
    :param default_color: colour for lines not matched by *line_color_map*
    :param figsize: ``(width, height)`` in inches; only used when *ax* is *None*
    :return: the matplotlib axis

    Example rendering a unified diff::

        from yobx import doc
        diff_text = some_patch.make_diff()
        ax = doc.plot_text(
            diff_text,
            title="my patch",
            line_color_map={"+": "#2a9d2a", "-": "#cc2222", "@": "#1a6fbf"},
        )

    .. plot::

        import textwrap
        import matplotlib.pyplot as plt
        from yobx.doc import plot_text

        sample = textwrap.dedent(
            \"\"\"
            --- a/foo.py
            +++ b/foo.py
            @@ -1,3 +1,3 @@
            def foo():
            -    return 1
            +    return 2
            \"\"\")
        ax = plot_text(
            sample,
            title="sample diff",
            line_color_map={"+": "#2a9d2a", "-": "#cc2222", "@": "#1a6fbf"},
        )
        plt.show()
    """
    import matplotlib.pyplot as plt

    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    if ax is None:
        if figsize is None:
            figsize = (10, max(2, n_lines * 0.18 + 0.5))
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_lines)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 2, loc="left", pad=3)

    color_map = line_color_map or {}
    for i, line in enumerate(lines):
        color = color_map.get(line[:1], default_color)
        ax.text(
            0.01,
            n_lines - i - 0.5,
            line,
            fontsize=fontsize,
            color=color,
            fontfamily="monospace",
            va="center",
            transform=ax.transData,
        )

    return ax


def demo_mlp_model(filename: str) -> onnx.ModelProto:
    """
    One model obtained with the following.

    .. code-block:: python

        import torch
        from yobx.helpers.onnx_helper import pretty_onnx
        from yobx.xbuilder import OptimizationOptions
        from yobx.torch.interpreter import to_onnx
        from yobx.translate import translate


        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)


        x = torch.rand(3, 10)
        onx = to_onnx(
            MLP(), (x,), input_names=["x"], options=OptimizationOptions(patterns=None)
        )
        print(pretty_onnx(onx))
        print(translate(onx, api="onnx-short"))
    """
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["x", "p_layers_0_weight::T10"], ["_onx_matmul_x"]),
                oh.make_node("Add", ["_onx_matmul_x", "layers.0.bias"], ["linear"]),
                oh.make_node("Relu", ["linear"], ["relu"]),
                oh.make_node("MatMul", ["relu", "p_layers_2_weight::T10"], ["_onx_matmul_relu"]),
                oh.make_node("Add", ["_onx_matmul_relu", "layers.2.bias"], ["output_0"]),
            ],
            "experiment",
            [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (3, 10))],
            [oh.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (3, 1))],
            [
                onh.from_array(
                    np.random.randn(10, 32).astype(np.float32), name="p_layers_0_weight::T10"
                ),
                onh.from_array(
                    np.random.randn(32, 1).astype(np.float32), name="p_layers_2_weight::T10"
                ),
                onh.from_array(np.random.randn(32).astype(np.float32), name="layers.0.bias"),
                onh.from_array(
                    np.array([-0.1422213315963745], dtype=np.float32), name="layers.2.bias"
                ),
            ],
        ),
        functions=[],
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=8,
    )
