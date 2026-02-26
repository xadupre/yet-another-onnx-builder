import argparse
import os
import re
import sys
import textwrap
import onnx
from typing import Any, List, Optional
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
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        required=False,
        help="verbosity",
    )
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
        print(dot)
    if args.run:
        assert args.output, "Cannot run dot without an output file."
        outname = process_outputname(outname, args.input)
        cmds = ["dot", f"-T{args.run}", outname, "-o", f"{args.output}.{args.run}"]
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
        epilog="Enables Some quick validation.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to unlighten",
    )
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        required=False,
        help="Names to look at comma separated values, if 'SHADOW', "
        "search for shadowing names.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        type=int,
        required=False,
        help="verbosity",
    )
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
        res = list(enumerate_results(onx, name=set(args.names.split(",")), verbose=args.verbose))
        if not args.verbose:
            print("\n".join(map(str, res)))
    else:
        onnx_find(args.input, verbose=args.verbose, watch=set(args.names.split(",")))


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

                python -m onnx_diagnostic agg test_agg.xlsx raw/*.zip -v 1
                python -m onnx_diagnostic agg agg.xlsx raw/*.zip raw/*.csv -v 1 \\
                    --no-raw  --keep-last-date --filter-out "exporter:test-exporter"

            Another to create timeseries:

                python -m onnx_diagnostic agg history.xlsx raw/*.csv -v 1 --no-raw \\
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
        help="Keeps only the most recent experiment for the same of keys.",
    )
    parser.add_argument(
        "--keep-last-date",
        default=False,
        action=BooleanOptionalAction,
        help="Rewrite all dates to the last one to simplifies the analysis, "
        "this assume changing the date does not add ambiguity, if any, option "
        "--recent should be added.",
    )
    parser.add_argument(
        "--raw",
        default=True,
        action=BooleanOptionalAction,
        help="Keeps the raw data in a sheet.",
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
    parser.add_argument(
        "--csv",
        default="raw-short",
        help="Views to dump as csv files.",
    )
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
        from tqdm import tqdm

        loop = tqdm(csv)
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

                python -m onnx_diagnostic partition \\
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
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        type=int,
        help="verbosity",
    )
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
        choices=["dot", "pretty", "printer", "raw", "shape"],
        default="pretty",
        help=textwrap.dedent("""
            Prints out a model on the standard output.

            dot     - converts the graph into dot
            pretty  - an improved rendering
            printer - onnx.printer.to_text(...)
            raw     - just prints the model with print(...)
            shape   - prints every node node with input and output shapes
            text    - uses GraphRendering

            """.strip("\n")),
    )
    parser.add_argument("input", type=str, help="onnx model to load")
    return parser


def _cmd_print(argv: List[Any]):
    parser = get_parser_print()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.input)
    if args.fmt == "raw":
        print(onx)
    elif args.fmt == "pretty":
        from .helpers.onnx_helper import pretty_onnx

        print(pretty_onnx(onx))
    elif args.fmt == "printer":
        print(onnx.printer.to_text(onx))
    elif args.fmt == "shape":
        from experimental_experiment.xbuilder import GraphBuilder

        print(GraphBuilder(onx).pretty_text())
    elif args.fmt == "dot":
        from .helpers.dot_helper import to_dot

        print(to_dot(onx))
    else:
        raise ValueError(f"Unexpected value fmt={args.fmt!r}")


#############
# main parser
#############


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx_diagnostic",
        description="onnx_diagnostic main command line.\n",
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            Type 'python -m onnx_diagnostic <cmd> --help'
            to get help for a specific command.

            agg          - aggregates statistics from multiple files
            dot          - converts an onnx model into dot format
            find         - find node consuming or producing a result
            partition    - partition a model, each partition appears as local function
            print        - prints the model on standard output
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
