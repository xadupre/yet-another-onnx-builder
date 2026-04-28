import argparse
import os
import re
import sys
import textwrap
import onnx
from typing import Any, Dict, List, Optional
from argparse import ArgumentParser, RawTextHelpFormatter, BooleanOptionalAction


def process_outputname(output_name: str, input_name: str) -> str:
    """
    If 'output_name' starts with '+', then it is modified into
    ``<input_name_no_extension><output_name>.extension``.
    """
    if not output_name.startswith("+"):
        return output_name
    name, ext = os.path.splitext(input_name)
    return f"{name}{output_name[1:]}{ext}"


class _ParseNamedDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert ":" in values, f"':' missing from {values!r}"
        namespace_key, rest = values.split(":", 1)
        pairs = rest.split(",")
        inner_dict = {}

        for pair in pairs:
            if "=" not in pair:
                raise argparse.ArgumentError(self, f"Expected '=' in pair '{pair}'")
            key, value = pair.split("=", 1)
            inner_dict[key] = value
        assert inner_dict, f"Unable to parse {rest!r} into a dictionary"
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        assert isinstance(
            getattr(namespace, self.dest), dict
        ), f"Unexpected type for namespace.{self.dest}={getattr(namespace, self.dest)}"
        getattr(namespace, self.dest).update({namespace_key: inner_dict})


def get_parser_dot() -> ArgumentParser:
    parser = ArgumentParser(
        prog="dot",
        description=textwrap.dedent("""
            Converts a model into a dot file dot can draw into a graph.
            """),
    )
    parser.add_argument("input", type=str, help="onnx model to lighten")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        type=str,
        required=False,
        help="dot model to output or empty to print out the result",
    )
    parser.add_argument("-v", "--verbose", type=int, default=0, required=False, help="verbosity")
    parser.add_argument(
        "-r",
        "--run",
        default="",
        required=False,
        help="run dot, in that case, format must be given (svg, png)",
    )
    return parser


def _cmd_dot(argv: List[Any]):
    import subprocess
    from .helpers.dot_helper import to_dot

    parser = get_parser_dot()
    args = parser.parse_args(argv[1:])
    if args.verbose:
        print(f"-- loads {args.input!r}")
    onx = onnx.load(args.input, load_external_data=False)
    if args.verbose:
        print("-- converts into dot")
    dot = to_dot(onx)
    if args.output:
        outname = process_outputname(args.output, args.input)
        if args.verbose:
            print(f"-- saves into {outname!r}")
        with open(outname, "w") as f:
            f.write(dot)
    else:
        outname = None
        print(dot)
    if args.run:
        assert outname, f"Cannot run dot without an output file but outname={outname!r}."
        cmds = ["dot", f"-T{args.run}", outname, "-o", f"{outname}.{args.run}"]
        if args.verbose:
            print(f"-- run {' '.join(cmds)}")
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        out, err = res
        if out:
            print("--")
            print(out)
        if err:
            print("--")
            print(err)


def get_parser_find() -> ArgumentParser:
    parser = ArgumentParser(
        prog="find",
        description=textwrap.dedent("""
            Look into a model and search for a set of names,
            tells which node is consuming or producing it.
            """),
        epilog="Enables some quick validation.",
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="onnx model to search")
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        required=False,
        help="Names to look at comma separated values, if 'SHADOW', "
        "search for shadowing names.",
    )
    parser.add_argument("-v", "--verbose", default=0, type=int, required=False, help="verbosity")
    parser.add_argument(
        "--v2",
        default=False,
        action=BooleanOptionalAction,
        help="Uses enumerate_results instead of onnx_find.",
    )
    return parser


def _cmd_find(argv: List[Any]):
    from .helpers.onnx_helper import onnx_find, enumerate_results, shadowing_names

    parser = get_parser_find()
    args = parser.parse_args(argv[1:])
    if args.names == "SHADOW":
        onx = onnx.load(args.input, load_external_data=False)
        s, ps = shadowing_names(onx)[:2]
        print(f"shadowing names: {s}")
        print(f"post-shadowing names: {ps}")
    elif args.v2:
        onx = onnx.load(args.input, load_external_data=False)
        names = set(args.names.split(",")) if args.names is not None else set()
        res = list(enumerate_results(onx, name=names, verbose=args.verbose))
        if not args.verbose:
            print("\n".join(map(str, res)))
    else:
        watch = set(args.names.split(",")) if args.names is not None else None
        onnx_find(args.input, verbose=args.verbose, watch=watch)


def get_parser_agg() -> ArgumentParser:
    parser = ArgumentParser(
        prog="agg",
        description=textwrap.dedent("""
            Aggregates statistics coming from benchmarks.
            Every run is a row. Every row is indexed by some keys,
            and produces values. Every row has a date.
            The data can come any csv files produces by benchmarks,
            it can concatenates many csv files, or csv files inside zip files.
            It produces an excel file with many tabs, one per view.
            """),
        epilog=textwrap.dedent("""
            examples:

                python -m yobx agg test_agg.xlsx raw/*.zip -v 1
                python -m yobx agg agg.xlsx raw/*.zip raw/*.csv -v 1 \\
                    --no-raw  --keep-last-date --filter-out "exporter:test-exporter"

            Another to create timeseries:

                python -m yobx agg history.xlsx raw/*.csv -v 1 --no-raw \\
                    --no-recent
            """),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("output", help="output excel file")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="input csv or zip files, at least 1, it can be a name, or search path",
    )
    parser.add_argument(
        "--filter", default="rawdata_.*.csv", help="filter for input files inside zip files"
    )
    parser.add_argument(
        "--recent",
        default=True,
        action=BooleanOptionalAction,
        help="Keeps only the most recent experiment for the same set of keys.",
    )
    parser.add_argument(
        "--keep-last-date",
        default=False,
        action=BooleanOptionalAction,
        help="Rewrite all dates to the last one to simplify the analysis, "
        "this assume changing the date does not add ambiguity, if any, option "
        "--recent should be added.",
    )
    parser.add_argument(
        "--raw", default=True, action=BooleanOptionalAction, help="Keeps the raw data in a sheet."
    )
    parser.add_argument("-t", "--time", default="DATE", help="Date or time column")
    parser.add_argument(
        "-k",
        "--keys",
        default="^version_.*,^model_.*,device,opt_patterns,suite,memory_peak,"
        "machine,exporter,dynamic,rtopt,dtype,device,architecture",
        help="List of columns to consider as keys, "
        "multiple values are separated by `,`\n"
        "regular expressions are allowed",
    )
    parser.add_argument(
        "--drop-keys",
        default="",
        help="Drops keys from the given list. Something it is faster "
        "to remove one than to select all the remaining ones.",
    )
    parser.add_argument(
        "-w",
        "--values",
        default="^time_.*,^disc.*,^ERR_.*,CMD,^ITER.*,^onnx_.*,^op_onnx_.*,^peak_gpu_.*",
        help="List of columns to consider as values, "
        "multiple values are separated by `,`\n"
        "regular expressions are allowed",
    )
    parser.add_argument(
        "-i", "--ignored", default="^version_.*", help="List of columns to ignore"
    )
    parser.add_argument(
        "-f",
        "--formula",
        default="speedup,bucket[speedup],ERR1,n_models,n_model_eager,"
        "n_model_running,n_model_acc01,n_model_acc001,n_model_dynamic,"
        "n_model_pass,n_model_faster,"
        "n_model_faster2x,n_model_faster3x,n_model_faster4x,n_node_attention,"
        "n_node_attention23,n_node_rotary_embedding,n_node_rotary_embedding23,"
        "n_node_gqa,n_node_layer_normalization,n_node_layer_normalization23,"
        "peak_gpu_torch,peak_gpu_nvidia,n_node_control_flow,n_node_random,"
        "n_node_constant,n_node_shape,n_node_expand,"
        "n_node_function,n_node_initializer,n_node_scatter,"
        "time_export_unbiased,onnx_n_nodes_no_cst,n_node_initializer_small",
        help="Columns to compute after the aggregation was done.",
    )
    parser.add_argument(
        "--views",
        default="agg-suite,agg-all,disc,speedup,time,time_export,err,cmd,"
        "bucket-speedup,raw-short,counts,peak-gpu,onnx",
        help=textwrap.dedent("""
            Views to add to the output files. Each view becomes a tab.
            A view is defined by its name, among
            agg-suite, agg-all, disc, speedup, time, time_export, err,
            cmd, bucket-speedup, raw-short, counts, peak-gpu, onnx.
            Their definition is part of class CubeLogsPerformance.
            """),
    )
    parser.add_argument("--csv", default="raw-short", help="Views to dump as csv files.")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="verbosity")
    parser.add_argument(
        "--filter-in",
        default="",
        help="adds a filter to filter in data, syntax is\n"
        '``"<column1>:<value1>;<value2>//<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--filter-out",
        default="",
        help="adds a filter to filter out data, syntax is\n"
        '``"<column1>:<value1>;<value2>//<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--sbs",
        help=textwrap.dedent("""
            Defines an exporter to compare to another, there must be at least
            two arguments defined with --sbs. Example:
                --sbs dynamo:exporter=onnx-dynamo,opt=ir,attn_impl=eager
                --sbs custom:exporter=custom,opt=default,attn_impl=eager
            """),
        action=_ParseNamedDict,
    )
    return parser


def _cmd_agg(argv: List[Any]):
    from .helpers._log_helper import open_dataframe, enumerate_csv_files, filter_data
    from .helpers.cube_helper import CubeLogsPerformance

    parser = get_parser_agg()
    args = parser.parse_args(argv[1:])
    reg = re.compile(args.filter)

    csv = list(
        enumerate_csv_files(
            args.inputs, verbose=args.verbose, filtering=lambda name: bool(reg.search(name))
        )
    )
    assert csv, f"No csv files in {args.inputs}, args.filter={args.filter!r}, csv={csv}"
    if args.verbose:
        try:
            from tqdm import tqdm

            loop = tqdm(csv)
        except ImportError:
            loop = csv
    else:
        loop = csv
    dfs = []
    for c in loop:
        df = open_dataframe(c)
        assert (
            args.time in df.columns
        ), f"Missing time column {args.time!r} in {c!r}\n{df.head()}\n{sorted(df.columns)}"
        dfs.append(filter_data(df, filter_in=args.filter_in, filter_out=args.filter_out))

    drop_keys = set(args.drop_keys.split(","))
    cube = CubeLogsPerformance(
        dfs,
        time=args.time,
        keys=[a for a in args.keys.split(",") if a and a not in drop_keys],
        values=[a for a in args.values.split(",") if a],
        ignored=[a for a in args.ignored.split(",") if a],
        recent=args.recent,
        formulas={k: k for k in args.formula.split(",")},
        keep_last_date=args.keep_last_date,
    )
    cube.load(verbose=max(args.verbose - 1, 0))
    if args.verbose:
        print(f"Dumps final file into {args.output!r}")
    cube.to_excel(
        args.output,
        {k: k for k in args.views.split(",")},
        verbose=args.verbose,
        csv=args.csv.split(","),
        raw=args.raw,
        time_mask=True,
        sbs=args.sbs,
    )
    if args.verbose:
        print(f"Wrote {args.output!r}")


def get_parser_partition() -> ArgumentParser:
    parser = ArgumentParser(
        prog="partition",
        formatter_class=RawTextHelpFormatter,
        description=textwrap.dedent("""
            Partitions an onnx model by moving nodes into local functions.
            Exporters may add metadata to the onnx nodes telling which part
            of the model it comes from (namespace, source, ...).
            This nodes are moved into local functions.
            """),
        epilog=textwrap.dedent("""
            The regular may match the following values,
            'model.layers.0.forward', 'model.layers.1.forward', ...
            A local function will be created for each distinct layer.

            Example:

                python -m yobx partition \\
                        model.onnx +.part -v 1 -r "model.layers.0.s.*"
            """),
    )
    parser.add_argument("input", help="input model")
    parser.add_argument(
        "output",
        help=textwrap.dedent("""
            output model, an expression like '+.part'
            inserts '.part' just before the extension"
            """).strip("\n"),
    )
    parser.add_argument(
        "-r",
        "--regex",
        default=".*[.]layers[.][0-9]+[.]forward$",
        help=textwrap.dedent("""
            merges all nodes sharing the same value in node metadata,
            these values must match the regular expression specified by
            this parameter, the default value matches what transformers
            usually to define a layer
            """).strip("\n"),
    )
    parser.add_argument(
        "-p",
        "--meta-prefix",
        default="namespace,source[",
        help="allowed prefixes for keys in the metadata",
    )
    parser.add_argument("-v", "--verbose", default=0, required=False, type=int, help="verbosity")
    return parser


def _cmd_partition(argv: List[Any]):
    from .helpers.onnx_helper import make_model_with_local_functions

    parser = get_parser_partition()
    args = parser.parse_args(argv[1:])

    if args.verbose:
        print(f"-- load {args.input!r}")
    onx = onnx.load(args.input, load_external_data=False)
    if args.verbose:
        print("-- partition")
    onx2 = make_model_with_local_functions(
        onx,
        regex=args.regex,
        metadata_key_prefix=tuple(args.meta_prefix.split(",")),
        verbose=args.verbose,
    )
    outname = process_outputname(args.output, args.input)
    if args.verbose:
        print(f"-- save into {outname!r}")
    onnx.save(onx2, outname)
    if args.verbose:
        print("-- done")


def get_parser_print() -> ArgumentParser:
    parser = ArgumentParser(
        prog="print",
        description="Prints the model on the standard output.",
        epilog="To show a model.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "fmt",
        choices=["dot", "onnx-compact", "mermaid", "pretty", "printer", "raw", "shape"],
        default="pretty",
        help=textwrap.dedent("""
            Prints out a model on the standard output.

            dot          - converts the graph into dot
            onnx-compact - translates the model into compact Python code
            pretty       - an improved rendering
            printer      - onnx.printer.to_text(...)
            raw          - just prints the model with print(...)
            shape        - prints every node with input and output shapes

            """.strip("\n")),
    )
    parser.add_argument("input", type=str, help="onnx model to load")
    return parser


def _cmd_print(argv: List[Any]):
    parser = get_parser_print()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.input)
    if args.fmt == "onnx-compact":
        from .translate import translate

        print(translate(onx, api="onnx-compact"))
    elif args.fmt == "raw":
        print(onx)
    elif args.fmt == "pretty":
        from .helpers.onnx_helper import pretty_onnx

        print(pretty_onnx(onx))
    elif args.fmt == "printer":
        print(onnx.printer.to_text(onx))
    elif args.fmt == "shape":
        from .xshape import BasicShapeBuilder

        bs = BasicShapeBuilder()
        bs.run_model(onx)
        print(bs.get_debug_msg())
    elif args.fmt == "mermaid":
        from .translate import translate

        print(translate(onx, api="mermaid"))
    elif args.fmt == "dot":
        from .helpers.dot_helper import to_dot

        print(to_dot(onx))
    else:
        raise ValueError(f"Unexpected value fmt={args.fmt!r}")


def get_parser_run_doc_examples() -> ArgumentParser:
    parser = ArgumentParser(
        prog="run-doc-examples",
        description=textwrap.dedent("""
            Extracts all ``.. runpython::`` and ``.. gdot::`` code blocks from
            RST documentation files or Python source files (docstrings) and
            executes each one in an isolated subprocess. Exits with a non-zero
            status when at least one block fails.
            """),
        epilog=textwrap.dedent("""
            examples:

                # Check every runpython block in a single RST file
                python -m yobx run-doc-examples docs/design/misc/helpers.rst

                # Check all RST files in a directory tree
                python -m yobx run-doc-examples docs/ -v 1

                # Check Python source files (docstring examples)
                python -m yobx run-doc-examples yobx/helpers/helper.py -v 2

                # Multiple paths at once
                python -m yobx run-doc-examples \\
                    docs/design/misc/helpers.rst \\
                    yobx/helpers/helper.py \\
                    -v 1
            """),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="RST or Python files (or directories) to scan for runpython:: blocks.",
    )
    parser.add_argument(
        "--ext",
        default=".rst,.py",
        help="Comma-separated list of file extensions to include when a directory "
        "is given (default: '.rst,.py').",
    )
    parser.add_argument(
        "--timeout",
        default=None,
        type=int,
        required=False,
        help="Per-block execution timeout in seconds (no limit by default).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        required=False,
        help="Verbosity level: 0=failures only, 1=each block status, 2=also show code.",
    )
    return parser


def _collect_files(inputs: List[str], extensions: List[str]) -> List[str]:
    """Expand a mixed list of file paths and directories into individual file paths."""
    collected: List[str] = []
    for inp in inputs:
        if os.path.isfile(inp):
            collected.append(inp)
        elif os.path.isdir(inp):
            for dirpath, _dirnames, filenames in os.walk(inp):
                for fname in sorted(filenames):
                    if any(fname.endswith(ext) for ext in extensions):
                        collected.append(os.path.join(dirpath, fname))
        else:
            # Try glob expansion so the caller can pass patterns like "docs/**/*.rst"
            import glob as _glob

            matched = sorted(_glob.glob(inp, recursive=True))
            if matched:
                for m in matched:
                    if os.path.isfile(m):
                        collected.append(m)
            else:
                # Keep it so run_runpython_blocks can emit a warning
                collected.append(inp)
    return collected


def _cmd_run_doc_examples(argv: List[Any]):
    from .helpers._check_runpython import run_runpython_blocks

    parser = get_parser_run_doc_examples()
    args = parser.parse_args(argv[1:])

    extensions = [e.strip() for e in args.ext.split(",") if e.strip()]
    files = _collect_files(args.inputs, extensions)

    if not files:
        print("[run-doc-examples] No files found.")
        sys.exit(0)

    if args.verbose:
        print(f"[run-doc-examples] scanning {len(files)} file(s) ...")

    _, n_failed = run_runpython_blocks(
        files, verbose=args.verbose, raise_on_error=False, timeout=args.timeout
    )

    if n_failed:
        sys.exit(1)


def _gallery_auto_output_path(input_path: str) -> str:
    """
    Derive the sphinx-gallery ``auto_`` output RST path from a gallery source file.

    The sphinx-gallery convention maps:

    * ``<base>/examples/<category>/plot_foo.py``  →
      ``<base>/auto_examples_<category>/plot_foo.rst``

    :param input_path: absolute or relative path to the gallery ``.py`` file
    :return: absolute path to the corresponding ``.rst`` file in the
        ``auto_examples_<category>`` directory
    """
    abs_path = os.path.normpath(os.path.abspath(input_path))
    parts = abs_path.split(os.sep)
    # Walk backwards looking for a directory component named 'examples'
    # followed immediately by the category name.
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "examples" and i + 1 < len(parts) - 1:
            category = parts[i + 1]
            base_dir = os.sep.join(parts[:i]) or os.sep
            basename = os.path.splitext(parts[-1])[0] + ".rst"
            return os.path.join(base_dir, f"auto_examples_{category}", basename)
    # Fallback: write next to the source file
    return os.path.splitext(abs_path)[0] + ".rst"


def get_parser_render_gallery() -> ArgumentParser:
    parser = ArgumentParser(
        prog="render-gallery",
        description=textwrap.dedent("""
            Converts a sphinx-gallery Python example file (.py) to RST without
            executing any code and writes the result to the corresponding
            auto_examples_<category>/ folder.

            A sphinx-gallery example file consists of:
              - A module docstring (verbatim RST: title, description, labels, …)
              - Python code blocks separated by ``# %%`` section markers
              - Comment lines following ``# %%`` are treated as RST prose

            For an input file at ``docs/examples/<category>/plot_foo.py`` the
            output is written to ``docs/auto_examples_<category>/plot_foo.rst``.
            """),
        epilog=textwrap.dedent("""
            examples:

                # Convert a single gallery example
                python -m yobx render-gallery docs/examples/core/plot_dot_graph.py

                # Convert several examples at once
                python -m yobx render-gallery \\
                    docs/examples/core/plot_dot_graph.py \\
                    docs/examples/sklearn/plot_sklearn_pipeline.py
            """),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", help="sphinx-gallery Python file(s) to convert to RST."
    )
    parser.add_argument("-v", "--verbose", type=int, default=0, required=False, help="verbosity")
    return parser


def _cmd_render_gallery(argv: List[Any]):
    from .helpers._gallery_helper import gallery_to_rst

    parser = get_parser_render_gallery()
    args = parser.parse_args(argv[1:])

    for filepath in args.inputs:
        if not os.path.isfile(filepath):
            print(f"[render-gallery] ERROR: file not found: {filepath!r}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"[render-gallery] converting {filepath!r}")

        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()

        rst = gallery_to_rst(source)

        out_path = _gallery_auto_output_path(filepath)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if args.verbose:
            print(f"[render-gallery] writing {out_path!r}")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rst)


def get_parser_validate() -> ArgumentParser:
    _DEFAULT_PROMPT = "Continue: it rains, what should I do?"
    parser = ArgumentParser(
        prog="validate",
        description=textwrap.dedent("""
            Validates an ONNX export for a HuggingFace model.
            This command captures real inputs by running the model on a default
            text prompt with InputObserver, then exports to ONNX and checks
            discrepancies.
            """),
        epilog=textwrap.dedent("""
            Examples:

                python -m yobx validate -m arnir0/Tiny-LLM -v 1
                python -m yobx validate -m arnir0/Tiny-LLM -v 1 -o dump_validate
                python -m yobx validate -m arnir0/Tiny-LLM --no-patch --no-run

            With mode arguments:

                python -m yobx validate -m arnir0/Tiny-LLM \\
                       -e yobx --opt default --opset 22 --device cuda --dtype float32 \\
                       --patch -r -o dump_test -v 1
            """),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mid",
        type=str,
        required=True,
        help="Model id, usually <author>/<name> on HuggingFace Hub.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=_DEFAULT_PROMPT,
        help=f"Text prompt used to drive model.generate() during input capture. "
        f"Default: {_DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "-e", "--export", default="yobx", type=str, help="ONNX exporter to use (default: 'yobx')."
    )
    parser.add_argument(
        "--opt",
        default="default",
        type=str,
        required=False,
        help="Optimisation level applied after export (default: 'default').",
    )
    parser.add_argument(
        "-r",
        "--run",
        default=True,
        action=BooleanOptionalAction,
        help="Check discrepancies after export (default: True).",
    )
    parser.add_argument(
        "--patch",
        default=True,
        action=BooleanOptionalAction,
        help="Apply apply_patches_for_model and register_flattening_functions "
        "during input capture and export (default: True).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        default=False,
        action=BooleanOptionalAction,
        help="Catch exceptions and report them in the summary instead of re-raising.",
    )
    parser.add_argument(
        "--opset", default=22, type=int, help="ONNX opset version to target (default: 22)."
    )
    parser.add_argument(
        "--dtype",
        default=None,
        type=str,
        help="Cast the model and inputs to this dtype before exporting, e.g. 'float16'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device to run on, e.g. 'cpu' or 'cuda' (default: 'cpu').",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=10,
        type=int,
        help="Number of tokens generated by model.generate() during input capture "
        "(default: 10).",
    )
    parser.add_argument(
        "-o",
        "--dump-folder",
        default=None,
        type=str,
        help="Save ONNX artefacts under this folder.",
    )
    parser.add_argument(
        "-v", "--verbose", default=0, type=int, help="Verbosity level (default: 0)."
    )
    parser.add_argument(
        "--random-weights",
        default=False,
        action="store_true",
        help="Instantiate the model from config with random weights instead of "
        "downloading pretrained weights (useful for fast CI tests).",
    )
    parser.add_argument(
        "--config-override",
        default=None,
        action="append",
        metavar="KEY=VALUE",
        dest="config_override",
        help="Override a config attribute before creating the model, "
        "e.g. --config-override num_hidden_layers=2. "
        "Can be repeated for multiple overrides.",
    )
    return parser


def _cmd_validate(argv: List[Any]):
    import ast

    from .torch.validate import validate_model

    parser = get_parser_validate()
    args = parser.parse_args(argv[1:])

    config_overrides: Optional[Dict[str, Any]] = None
    if args.config_override:
        config_overrides = {}
        for item in args.config_override:
            k, _, v = item.partition("=")
            try:
                config_overrides[k.strip()] = ast.literal_eval(v.strip())
            except Exception:
                config_overrides[k.strip()] = v.strip()

    summary, _data = validate_model(
        model_id=args.mid,
        prompt=args.prompt,
        exporter=args.export,
        optimization=args.opt,
        verbose=args.verbose,
        dump_folder=args.dump_folder,
        opset=args.opset,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_run=args.run,
        patch=args.patch,
        quiet=args.quiet,
        config_overrides=config_overrides,
        random_weights=args.random_weights,
    )

    print("")
    print("-- summary --")
    for k, v in sorted(summary.items()):
        print(f":{k},{v};")


def get_parser_stats() -> ArgumentParser:
    parser = ArgumentParser(
        prog="stats",
        description=textwrap.dedent("""
            Computes statistics on an ONNX model:
            number of nodes per op_type and estimation of computational cost.
            """),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("input", type=str, help="onnx model to analyze")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        type=str,
        required=False,
        help="output file (.xlsx or .csv) or empty to print on standard output",
    )
    parser.add_argument("-v", "--verbose", type=int, default=0, required=False, help="verbosity")
    return parser


def _cmd_stats(argv: List[Any]):
    from .helpers.stats_helper import model_statistics

    parser = get_parser_stats()
    args = parser.parse_args(argv[1:])
    if args.verbose:
        print(f"-- loads {args.input!r}")
    onx = onnx.load(args.input, load_external_data=False)
    if args.verbose:
        print("-- computes statistics")
    stats = model_statistics(onx, verbose=args.verbose)

    if args.output:
        outname = process_outputname(args.output, args.input)
        assert outname.endswith(
            (".csv", ".xlsx")
        ), f"Unexpected extension for outname={outname!r}"
        import pandas

        df = pandas.DataFrame(stats).sort_values()  # type: ignore
        if outname.endswith(".csv"):
            df.to_csv(outname, index=False)
        else:
            df.to_excel(outname, index=False)
    else:
        _print_model_statistics(stats)


def _print_model_statistics(stats: Dict[str, Any]) -> None:
    """Prints two model-statistics reports on standard output."""
    # Report 1: node counts per op_type
    print(f"Number of nodes: {stats['n_nodes']}")
    print("")
    print("Report 1 — Node counts per op_type:")
    for op, count in sorted(stats["node_count_per_op_type"].items()):
        print(f"  {op}: {count}")

    print("")

    # Report 2: estimated FLOPs per op_type
    print("Report 2 — Estimated FLOPs per op_type:")
    for op, flops in sorted(stats["flops_per_op_type"].items()):
        flops_str = str(flops) if flops is not None else "n/a"
        print(f"  {op}: {flops_str}")
    print("")
    total = stats["total_estimated_flops"]
    if total is not None:
        print(f"Total estimated FLOPs: {total}")
    else:
        print("Total estimated FLOPs: n/a (dynamic shapes or unsupported op_types)")


#############
# main parser
#############


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="yobx",
        description="yobx main command line.\n",
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            Type 'python -m yobx <cmd> --help'
            to get help for a specific command.

            agg               - aggregates statistics from multiple files
            dot               - converts an onnx model into dot format
            find              - find node consuming or producing a result
            partition         - partition a model, each partition appears as local function
            print             - prints the model on standard output
            render-gallery    - convert a sphinx-gallery .py example to RST (no execution)
            run-doc-examples  - run all runpython:: and gdot:: examples in RST/Python files
            stats             - compute statistics on an onnx model
            validate          - validate a model (knowing its model id on HuggingFace Hub)
            """),
    )
    parser.add_argument(
        "cmd",
        choices=[
            "agg",
            "dot",
            "find",
            "partition",
            "print",
            "render-gallery",
            "run-doc-examples",
            "stats",
            "validate",
        ],
        help="Selects a command.",
    )
    return parser


def main(argv: Optional[List[Any]] = None):
    fcts = dict(
        agg=_cmd_agg,
        dot=_cmd_dot,
        find=_cmd_find,
        partition=_cmd_partition,
        print=_cmd_print,
        **{"render-gallery": _cmd_render_gallery},
        **{"run-doc-examples": _cmd_run_doc_examples},
        stats=_cmd_stats,
        validate=_cmd_validate,
    )

    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0 or (len(argv) <= 1 and argv[0] not in fcts) or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(
                agg=get_parser_agg,
                dot=get_parser_dot,
                find=get_parser_find,
                partition=get_parser_partition,
                print=get_parser_print,
                **{"render-gallery": get_parser_render_gallery},
                **{"run-doc-examples": get_parser_run_doc_examples},
                stats=get_parser_stats,
                validate=get_parser_validate,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()  # type: ignore[operator]
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    if cmd in fcts:
        fcts[cmd](argv)
    else:
        raise ValueError(f"Unknown command {cmd!r}, use --help to get the list of known command.")
