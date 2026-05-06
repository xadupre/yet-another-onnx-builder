import json
import logging
import os
import shutil
import sys
import time
from sphinx_runpython.github_link import make_linkcode_resolve
import yobx

# Suppress TensorFlow C++ and Python logging before any TF import occurs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Per-page build duration tracking (writes top-5 to a JSON file so that the
# doc_build_durations.rst page can display it in the generated documentation).
# ---------------------------------------------------------------------------

_PAGE_TIMINGS_START: dict[str, float] = {}
_PAGE_DURATIONS: dict[str, float] = {}
_DOC_BUILD_DURATIONS_JSON = os.path.join(
    os.path.dirname(__file__), "_static", "doc_build_durations.json"
)
os.environ["YOBX_DOC_BUILD_DURATIONS_JSON"] = _DOC_BUILD_DURATIONS_JSON


def _on_source_read(app, docname, source):
    """Records the start time when a source file begins to be read."""
    _PAGE_TIMINGS_START[docname] = time.monotonic()


def _on_doctree_read(app, doctree):
    """Records the elapsed time after a doctree has been parsed."""
    docname = app.env.docname
    start = _PAGE_TIMINGS_START.pop(docname, None)
    if start is not None:
        _PAGE_DURATIONS[docname] = time.monotonic() - start


def _on_build_finished(app, exception):
    """Writes the 5 slowest pages (by build duration) to a JSON file."""
    if exception or not _PAGE_DURATIONS:
        return
    top5 = sorted(_PAGE_DURATIONS.items(), key=lambda kv: kv[1], reverse=True)[:5]
    static_dir = os.path.dirname(_DOC_BUILD_DURATIONS_JSON)
    if os.path.exists(static_dir) and not os.path.isdir(static_dir):
        os.remove(static_dir)
    os.makedirs(static_dir, exist_ok=True)
    with open(_DOC_BUILD_DURATIONS_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            [{"docname": name, "duration_s": round(dur, 3)} for name, dur in top5], fh, indent=2
        )


def _on_builder_inited(app):
    """Removes any non-directory file named '_static' in the output directory.

    Sphinx 9.1 raises an error when it tries to create the ``_static`` subfolder
    but finds a plain file with that name instead of a directory.  This hook
    runs at builder-init time and removes such a stale file so that Sphinx can
    proceed normally.
    """
    outdir_static = os.path.join(str(app.outdir), "_static")
    if os.path.exists(outdir_static) and not os.path.isdir(outdir_static):
        os.remove(outdir_static)


def setup(app):
    """Connects duration-tracking hooks to Sphinx events."""
    app.connect("builder-inited", _on_builder_inited)
    app.connect("source-read", _on_source_read)
    app.connect("doctree-read", _on_doctree_read)
    app.connect("build-finished", _on_build_finished)


project = "yet-another-onnx-builder"
author = "yet-another-onnx-builder contributors"
release = yobx.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinxcontrib.mermaid",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runmermaid",
    "sphinx_runpython.runpython",
]
if shutil.which("latex"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
graphviz_output_format = "svg"
graphviz_dot_args = ["-Gbgcolor=transparent"]

autodoc_typehints_format = "short"

mermaid_init_js = """
document.addEventListener("DOMContentLoaded", function () {
    // pydata_sphinx_theme sets data-bs-theme="dark" on <html> for dark mode.
    function getMermaidTheme() {
        return document.documentElement.getAttribute("data-bs-theme") === "dark"
            ? "dark" : "default";
    }
    mermaid.initialize({ startOnLoad: true, theme: getMermaidTheme() });
    new MutationObserver(function () {
        const theme = getMermaidTheme();
        document.querySelectorAll(".mermaid[data-processed]").forEach(function (el) {
            el.removeAttribute("data-processed");
        });
        mermaid.initialize({ startOnLoad: false, theme: theme });
        mermaid.run();
    }).observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-bs-theme"],
    });
});
"""

# templates_path = ["_templates"]
exclude_patterns = ["_build"]
if int(os.environ.get("UNITTEST_GOING", "0")):
    exclude_patterns.append("ci_durations.rst")
    exclude_patterns.append("commits_per_week.rst")
    exclude_patterns.append("pypi_downloads.rst")
    exclude_patterns.append("design/torch/case_coverage.rst")
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "github_url": "https://github.com/xadupre/yet-another-onnx-builder",
    "logo": {"image_light": "_static/logo.svg", "image_dark": "_static/logo.svg"},
}


def linkcode_resolve(domain, info):
    return _linkcode_resolve(domain, info)


_linkcode_resolve = make_linkcode_resolve(
    "yobx",
    (
        "https://github.com/xadupre/yet-another-onnx-builder/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

intersphinx_mapping = {
    # Not a sphinx documentation
    # "diffusers": ("https://huggingface.co/docs/diffusers/index", None),
    "category_encoders": ("https://contrib.scikit-learn.org/category_encoders/", None),
    "imblearn": ("https://imbalanced-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "onnxscript": ("https://microsoft.github.io/onnxscript/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skl2onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "sktorch": ("https://skorch.readthedocs.io/en/latest/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/mr-ubik/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

suppress_warnings = ["intersphinx.external"]
if int(os.environ.get("UNITTEST_GOING", "0")):
    # Suppress toctree and cross-reference warnings for pages intentionally
    # excluded on CI (e.g. case_coverage.rst, ci_durations.rst).
    suppress_warnings += ["toc.excluded", "ref.ref"]


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        os.path.join(os.path.dirname(__file__), "examples", "core"),
        os.path.join(os.path.dirname(__file__), "examples", "sql"),
        os.path.join(os.path.dirname(__file__), "examples", "sklearn"),
        os.path.join(os.path.dirname(__file__), "examples", "torch"),
        os.path.join(os.path.dirname(__file__), "examples", "tensorflow"),
        os.path.join(os.path.dirname(__file__), "examples", "litert"),
        os.path.join(os.path.dirname(__file__), "examples", "transformers"),
    ],
    # path where to save gallery generated examples
    "gallery_dirs": [
        "auto_examples_core",
        "auto_examples_sql",
        "auto_examples_sklearn",
        "auto_examples_torch",
        "auto_examples_tensorflow",
        "auto_examples_litert",
        "auto_examples_transformers",
    ],
    # no parallelization to avoid conflict with environment variables
    "parallel": 1,
    # sorting
    "within_subsection_order": "ExampleTitleSortKey",
    # errors
    "abort_on_example_error": True,
    # recommendation
    "recommender": {"enable": True, "n_examples": 3, "min_df": 3, "max_df": 0.9},
    # ignore capture for matplotib axes
    "ignore_repr_types": "matplotlib\\.(text|axes)",
    # robubstness
    "reset_modules_order": "both",
    "reset_modules": (
        "matplotlib",
        "yobx.doc.reset_torch_transformers",
        "yobx.doc.reset_tensorflow",
    ),
}

substring_to_disable = []
if int(os.environ.get("UNITTEST_GOING", "0")):
    substring_to_disable = ["tiny_llm", "examples.transformers.plot_"]
substring = "|".join(f"({s})" for s in substring_to_disable)
if substring:
    sphinx_gallery_conf["ignore_pattern"] = f".*({substring}).*"

epkg_dictionary = {
    "aten functions": "https://pytorch.org/cppdocs/api/namespace_at.html#functions",
    "azure pipeline": "https://azure.microsoft.com/en-us/products/devops/pipelines",
    "guard_size_oblivious": "https://docs.pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html",
    "black": "https://github.com/psf/black",
    "category_encoders": "https://contrib.scikit-learn.org/category_encoders/",
    "Custom Backends": "https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html",
    "diffusers": "https://github.com/huggingface/diffusers",
    "dot": "https://graphviz.org/doc/info/lang.html",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "executorch": "https://pytorch.org/executorch/stable/intro-overview.html",
    "ExecuTorch": "https://pytorch.org/executorch/stable/intro-overview.html",
    "ExecuTorch Runtime Python API Reference": "https://docs.pytorch.org/executorch/main/runtime-python-api-reference.html",
    "ExecuTorch Tutorial": "https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "FunctionProto": "https://onnx.ai/onnx/api/classes.html#functionproto",
    "graph break": "https://docs.pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks",
    "Graphviz": "https://graphviz.org/",
    "GraphModule": "https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule",
    "HuggingFace": "https://huggingface.co/docs/hub/en/index",
    "huggingface_hub": "https://github.com/huggingface/huggingface_hub",
    "imbalanced-learn": "https://imbalanced-learn.org/stable/",
    "ai_edge_litert": "https://pypi.org/project/ai-edge-litert/",
    "flax": "https://flax.readthedocs.io/en/latest/",
    "equinox": "https://docs.kidger.site/equinox/",
    "IPython": "https://ipython.org/",
    "ir-py": "https://onnx.ai/ir-py/",
    "jax": "https://docs.jax.dev/en/latest/",
    "JAX": "https://docs.jax.dev/en/latest/",
    "jax2onnx": "https://github.com/enpasos/jax2onnx",
    "jax2tf": "https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md",
    "lightgbm": "https://lightgbm.readthedocs.io/en/latest/",
    "LightGBM": "https://lightgbm.readthedocs.io/en/latest/",
    "Linux": "https://www.linux.org/",
    "LiteRT": "https://ai.google.dev/edge/litert/",
    "mermaid-py": "https://pypi.org/project/mermaid-py/",
    "ml_dtypes": "https://github.com/jax-ml/ml_dtypes",
    "ModelBuilder": "https://onnxruntime.ai/docs/genai/howto/build-model.html",
    "monai": "https://github.com/Project-MONAI/MONAI",
    "numpy": "https://numpy.org/",
    "onnx": "https://onnx.ai/onnx/",
    "onnx-ir": "https://github.com/onnx/ir-py",
    "onnx.helper": "https://onnx.ai/onnx/api/helper.html",
    "ONNX": "https://onnx.ai/",
    "ONNX Operators": "https://onnx.ai/onnx/operators/",
    "onnxmltools": "https://github.com/onnx/onnxmltools",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-genai": "https://github.com/microsoft/onnxruntime-genai",
    "onnxruntime-training": "https://onnxruntime.ai/docs/get-started/training-on-device.html",
    "onnxruntime kernels": "https://onnxruntime.ai/docs/reference/operators/OperatorKernels.html",
    "onnx-script": "https://github.com/microsoft/onnxscript",
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "onnxscript Tutorial": "https://microsoft.github.io/onnxscript/tutorial/index.html",
    "optree": "https://github.com/metaopt/optree",
    "pandas": "https://pandas.pydata.org/",
    "Pattern-based Rewrite Using Rules With onnxscript": "https://microsoft.github.io/onnxscript/tutorial/rewriter/rewrite_patterns.html",
    "polars.LazyFrame": "https://docs.polars.io/api/python/stable/reference/lazyframe/api/polars.LazyFrame.html",
    "opsets": "https://onnx.ai/onnx/intro/concepts.html#what-is-an-opset-version",
    "pyinstrument": "https://pyinstrument.readthedocs.io/en/latest/",
    "psutil": "https://psutil.readthedocs.io/en/latest/",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "run_with_ortvaluevector": "https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_inference_collection.py#L339",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "StableHLO": "https://openxla.org/stablehlo/spec",
    "scikit-survival": "https://scikit-survival.readthedocs.io/en/stable/",
    "scipy": "https://scipy.org/",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "sktorch": "https://skorch.readthedocs.io/en/latest/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "sphinx-runpython": "https://github.com/sdpython/sphinx-runpython",
    "spox": "https://spox.readthedocs.io/en/latest/",
    "Supported Operators and Data Types": "https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md",
    "sympy": "https://www.sympy.org/en/index.html",
    "statsmodels": "https://www.statsmodels.org/",
    "Keras": "https://keras.io/",
    "tensorflow": "https://www.tensorflow.org/",
    "TensorFlow": "https://www.tensorflow.org/",
    "tensorflow-onnx": "https://github.com/onnx/tensorflow-onnx",
    "tf2onnx": "https://github.com/onnx/tensorflow-onnx",
    "TFLite": "https://ai.google.dev/edge/litert",
    "timm": "https://github.com/huggingface/pytorch-image-models",
    "torch": "https://docs.pytorch.org/docs/stable/torch.html",
    "torchbench": "https://github.com/pytorch/benchmark",
    "torch.compile": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
    "torch.compiler": "https://docs.pytorch.org/docs/stable/torch.compiler.html",
    "torch.export.export": "https://docs.pytorch.org/docs/stable/export.html#torch.export.export",
    "torch.onnx": "https://docs.pytorch.org/docs/stable/onnx.html",
    "tqdm": "https://github.com/tqdm/tqdm",
    "transformers": "https://huggingface.co/docs/transformers/en/index",
    "vibe coding": "https://en.wikipedia.org/wiki/Vibe_coding",
    "vocos": "https://github.com/gemelo-ai/vocos",
    "Windows": "https://www.microsoft.com/windows",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/get_started.html",
    "XGBoost": "https://xgboost.readthedocs.io/en/stable/get_started.html",
    "yet-another-onnxruntime-extensions": "https://sdpython.github.io/doc/yet-another-onnxruntime-extensions/dev/",
}

# models
epkg_dictionary.update(
    {
        "arnir0/Tiny-LLM": "https://huggingface.co/arnir0/Tiny-LLM",
        "microsoft/Phi-1.5": "https://huggingface.co/microsoft/phi-1_5",
        "microsoft/phi-2": "https://huggingface.co/microsoft/phi-2",
        "microsoft/Phi-3.5-mini-instruct": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-vision-instruct": "https://huggingface.co/microsoft/Phi-3.5-vision-instruct",
    }
)
