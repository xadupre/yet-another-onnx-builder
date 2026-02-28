import os
import sys
from sphinx_runpython.github_link import make_linkcode_resolve
import yobx

project = "yet-another-onnx-builder"
author = "yet-another-onnx-builder contributors"
release = yobx.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]
extensions.append("sphinx.ext.imgmath")
imgmath_image_format = "svg"

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "source_repository": "https://github.com/xadupre/yet-another-onnx-builder",
    "source_branch": "main",
    "source_directory": "docs/",
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
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "onnxscript": ("https://microsoft.github.io/onnxscript/", None),
    "onnx_diagnostic": ("https://sdpython.github.io/doc/onnx-diagnostic/dev/", None),
    "onnx_extended": ("https://sdpython.github.io/doc/onnx-extended/dev/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skl2onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [os.path.join(os.path.dirname(__file__), "examples")],
    # path where to save gallery generated examples
    "gallery_dirs": ["auto_examples"],
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

if int(os.environ.get("UNITTEST_GOING", "0")):
    sphinx_gallery_conf["ignore_pattern"] = (
        ".*((tiny_llm)|(dort)|(draft_mode)|(hub_codellama.py)|(whisper)|(optimind)|(export_with_modelbuilder)).*"
    )

epkg_dictionary = {
    "aten functions": "https://pytorch.org/cppdocs/api/namespace_at.html#functions",
    "azure pipeline": "https://azure.microsoft.com/en-us/products/devops/pipelines",
    "black": "https://github.com/psf/black",
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
    "GraphModule": "https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule",
    "HuggingFace": "https://huggingface.co/docs/hub/en/index",
    "huggingface_hub": "https://github.com/huggingface/huggingface_hub",
    "Linux": "https://www.linux.org/",
    "ml_dtypes": "https://github.com/jax-ml/ml_dtypes",
    "ModelBuilder": "https://onnxruntime.ai/docs/genai/howto/build-model.html",
    "monai": "https://github.com/Project-MONAI/MONAI",
    "numpy": "https://numpy.org/",
    "onnx": "https://onnx.ai/onnx/",
    "onnx-ir": "https://github.com/onnx/ir-py",
    "onnx.helper": "https://onnx.ai/onnx/api/helper.html",
    "ONNX": "https://onnx.ai/",
    "ONNX Operators": "https://onnx.ai/onnx/operators/",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-genai": "https://github.com/microsoft/onnxruntime-genai",
    "onnxruntime-training": "https://onnxruntime.ai/docs/get-started/training-on-device.html",
    "onnxruntime kernels": "https://onnxruntime.ai/docs/reference/operators/OperatorKernels.html",
    "onnx-array-api": "https://sdpython.github.io/doc/onnx-array-api/dev/",
    "onnx-diagnostic": "https://sdpython.github.io/doc/onnx-diagnostic/dev/",
    "onnx-extended": "https://sdpython.github.io/doc/onnx-extended/dev/",
    "onnx-script": "https://github.com/microsoft/onnxscript",
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "onnxscript Tutorial": "https://microsoft.github.io/onnxscript/tutorial/index.html",
    "optree": "https://github.com/metaopt/optree",
    "Pattern-based Rewrite Using Rules With onnxscript": "https://microsoft.github.io/onnxscript/tutorial/rewriter/rewrite_patterns.html",
    "opsets": "https://onnx.ai/onnx/intro/concepts.html#what-is-an-opset-version",
    "pyinstrument": "https://pyinstrument.readthedocs.io/en/latest/",
    "psutil": "https://psutil.readthedocs.io/en/latest/",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "run_with_ortvaluevector": "https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_inference_collection.py#L339",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "sphinx-runpython": "https://github.com/sdpython/sphinx-runpython",
    "Supported Operators and Data Types": "https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md",
    "sympy": "https://www.sympy.org/en/index.html",
    "timm": "https://github.com/huggingface/pytorch-image-models",
    "torch": "https://docs.pytorch.org/docs/stable/torch.html",
    "torchbench": "https://github.com/pytorch/benchmark",
    "torch.compile": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
    "torch.compiler": "https://docs.pytorch.org/docs/stable/torch.compiler.html",
    "torch.export.export": "https://docs.pytorch.org/docs/stable/export.html#torch.export.export",
    "torch.onnx": "https://docs.pytorch.org/docs/stable/onnx.html",
    "tqdm": "https://github.com/tqdm/tqdm",
    "transformers": "https://huggingface.co/docs/transformers/en/index",
    "vocos": "https://github.com/gemelo-ai/vocos",
    "Windows": "https://www.microsoft.com/windows",
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
