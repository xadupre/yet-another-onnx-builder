import os
import shutil
import sys
from sphinx_runpython.github_link import make_linkcode_resolve
import yobx

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
    "sphinx_runpython.runpython",
]
if shutil.which("latex"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
graphviz_output_format = "svg"
graphviz_dot_args = ["-Gbgcolor=transparent"]

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
    "reset_modules": ("matplotlib", "yobx.doc.reset_torch_transformers"),
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
    "ir-py": "https://onnx.ai/ir-py/",
    "jax": "https://docs.jax.dev/en/latest/",
    "JAX": "https://docs.jax.dev/en/latest/",
    "jax2onnx": "https://github.com/enpasos/jax2onnx",
    "jax2tf": "https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md",
    "lightgbm": "https://lightgbm.readthedocs.io/en/latest/",
    "LightGBM": "https://lightgbm.readthedocs.io/en/latest/",
    "Linux": "https://www.linux.org/",
    "LiteRT": "https://ai.google.dev/edge/litert/",
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
    "onnx-diagnostic": "https://sdpython.github.io/doc/onnx-diagnostic/dev/",
    "onnx-extended": "https://sdpython.github.io/doc/onnx-extended/dev/",
    "onnx-script": "https://github.com/microsoft/onnxscript",
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "onnxscript Tutorial": "https://microsoft.github.io/onnxscript/tutorial/index.html",
    "optree": "https://github.com/metaopt/optree",
    "pandas": "https://pandas.pydata.org/",
    "Pattern-based Rewrite Using Rules With onnxscript": "https://microsoft.github.io/onnxscript/tutorial/rewriter/rewrite_patterns.html",
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
