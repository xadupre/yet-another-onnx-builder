import difflib
import inspect
import pprint
import re
import textwrap
from typing import Any, Dict, Callable, Iterable, Iterator, List, Optional, Tuple


def clean_code_with_black(code: str) -> str:
    """Changes the code style with :epkg:`black` if available."""
    code = textwrap.dedent(code)
    try:
        import black  # type: ignore
    except ImportError:
        return code
    try:
        return black.format_str(code, mode=black.FileMode(line_length=98))
    except black.parsing.InvalidInput as e:
        raise RuntimeError(f"Unable to parse code\n\n---\n{code}\n---\n") from e


def make_diff_code(code1: str, code2: str, output: Optional[str] = None) -> str:
    """
    Creates a diff between two codes.

    :param code1: first code
    :param code2: second code
    :param output: if not empty, stores the output in this file
    :return: diff
    """
    text = "\n".join(
        difflib.unified_diff(
            code1.strip().splitlines(),
            code2.strip().splitlines(),
            fromfile="original",
            tofile="rewritten",
            lineterm="",
        )
    )
    if output:
        with open(output, "w") as f:
            f.write(text)
    return text


class PatchInfo:
    """
    Stores information about patches.

    :param function_to_patch: function to patch
    :param patch: function patched
    :param family: a category, anything to classify the patch
    :param do: applies the patch, this function returns the patched function
    :param undo: remove the patch
    :param _last_patched_function: this is used for patches applied when another
        is used
    """

    __slots__ = (
        "_do",
        "_last_patched_function",
        "_undo",
        "depends_on",
        "family",
        "function_to_patch",
        "patch",
    )

    @classmethod
    def _setattr(cls, to_patch, function_or_method: "str", new_value: Callable) -> Callable:
        previous = getattr(to_patch, function_or_method)
        setattr(to_patch, function_or_method, new_value)
        return previous

    @classmethod
    def make(
        cls,
        patch: Callable,
        module_or_class: Any,
        method_or_function_name: str,
        family: str,
        _last_patched_function: Optional[Callable] = None,
    ) -> "PatchInfo":
        """
        Creates a patch with the given information.

        :param patch: the patched method or function
        :param module_or_class: the module or the class to patch
        :param method_or_function_name: method or function name to patch
        :param family: category of the patch
        :param _last_patched_function: this is used for patches applied when another
            is used
        :return: the patch
        """
        return PatchInfo(
            patch=patch,
            family=family,
            do=lambda: PatchInfo._setattr(  # type: ignore
                module_or_class, method_or_function_name, patch
            ),
            undo=lambda original: PatchInfo._setattr(  # type: ignore
                module_or_class, method_or_function_name, original
            ),
            _last_patched_function=_last_patched_function,
        )

    def __init__(
        self,
        patch: Callable,
        do: Callable[[], None],
        undo: Callable[[Callable], None],
        family: str = "",
        _last_patched_function: Optional[Callable] = None,
    ):
        assert callable(patch), (
            f"function_to_patch is not a function but {type(patch)} - {patch!r}, "
            f"function_to_patch={patch!r}"
        )
        self.family = family
        self.function_to_patch = None
        self.patch = patch
        self.depends_on: List[PatchInfo] = []
        self._do = do
        self._undo = undo
        self._last_patched_function = _last_patched_function

    def do(self):
        """Applies the patch."""
        assert self.function_to_patch is None, f"The patch {self.patch} is already applied."
        self.function_to_patch = self._do()
        self._last_patched_function = self.function_to_patch
        assert self.function_to_patch is not None, f"The patch {self.patch} is not applied."
        assert not self.function_to_patch.__name__.startswith(
            "patch_"
        ), f"The function to patch {self.function_to_patch} is already a patch."

    def undo(self):
        """Removes the patch."""
        assert self.function_to_patch is not None, f"The patch {self.patch} is not applied."
        self._undo(self.function_to_patch)
        self.function_to_patch = None

    def add_dependency(self, patch_info: "PatchInfo"):
        self.depends_on.append(patch_info)

    @property
    def name(self) -> str:
        """Returns the name of the patch."""
        return self.patch.__name__

    def __repr__(self) -> str:
        "usual"
        return (
            (f"{self.__class__.__name__}({self.patch!r}, family={self.family!r})")
            if self.family
            else f"{self.__class__.__name__}({self.patch!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        "usual"
        return {k: getattr(self, k) for k in self.__slots__}

    def make_diff(self) -> str:
        """
        Returns a diff as a string.
        See example :ref:`l-plot-patch-model-diff`.

        ::

            -- original
            +++ rewritten
            @@ -1,6 +1,5 @@
            def _print_Symbol(self, expr: sympy.Symbol) -> str:
            -    if not isinstance(expr, sympy.Symbol):
            -        raise AssertionError(f"Expected sympy.Symbol, got {type(expr)}")
            -    if not self.symbol_to_source.get(expr):
            -        raise AssertionError(f"Unknown symbol {expr} created by constraints solver")
            -    return self.symbol_to_source[expr][0].name
            +    assert isinstance(expr, sympy.Symbol), str(type(expr))
            +    if self.symbol_to_source.get(expr):  # type: ignore
            +        return self.symbol_to_source[expr][0].name  # type: ignore
            +    return str(expr)
        """
        if isinstance(self.function_to_patch, str):
            return clean_code_with_black(inspect.getsource(self.patch))
        f = self.function_to_patch or self._last_patched_function
        assert f is not None, (
            f"The patch {self.name!r} was never applied "
            f"{self.function_to_patch=}, {self._last_patched_function=}."
        )
        src1 = clean_code_with_black(inspect.getsource(f))
        src2 = clean_code_with_black(inspect.getsource(self.patch))
        diff = make_diff_code(src1, src2)
        if not self.depends_on:
            return diff
        res = [diff]
        for d in self.depends_on:
            res.append("")
            res.append(d.make_diff())
        return "\n".join(res)

    @classmethod
    def function_name(cls, f: Callable) -> str:
        return f.__qualname__

    @property
    def refid(self) -> str:
        kind = self.family or ""
        patch_name = (
            self.function_name(self.patch)
            .replace(".", "-")
            .replace("/", "-")
            .replace(">", "")
            .replace("<", "")
        )
        return f"patch-{kind}-{patch_name}"

    def format_diff(self, format: str = "raw") -> str:
        """
        Formats a diff between two function as a string.

        :param format: ``'raw'`` or ``'rst'``
        :return: diff
        """
        diff = self.make_diff()
        kind = self.family or ""
        if kind:
            kind = f"{kind}: "
        f = self.function_to_patch or self._last_patched_function
        assert f, f"f is None for patch {self.name!r}"
        function_to_pach_name = f"{f!r}" if isinstance(f, str) else self.function_name(f)
        patch_name = self.function_name(self.patch)
        kind = kind.replace("_PATCHED_", "")
        title = f"{kind}{function_to_pach_name} -> {patch_name}"
        if format == "raw":
            return f"{title}\n{diff}"

        rows = [
            "",
            f".. _{self.refid}:",
            "",
            title,
            "=" * len(title),
            "",
            ".. code-block:: diff",
            "    :linenos:",
            "",
            textwrap.indent(diff, prefix="    "),
        ]
        return "\n".join(rows)


class PatchDetails:
    """
    This class is used to store patching information.
    This helps understanding which rewriting was applied to which
    method of functions. See page :ref:`l-plot-patch-model-diff`.
    """

    def __init__(self):
        self.patched = []
        self.find_cache = {}

    def __iter__(self) -> Iterator[PatchInfo]:
        """Iterates on all patches."""
        yield from self.patched

    def __len__(self) -> int:
        """Returns the number of applied patches."""
        return len(self.patched)

    def __getitem__(self, index: int) -> PatchInfo:
        """Returns a patch from the list at position `index`."""
        return self.patched[index]

    def find(self, name: str) -> Optional[PatchInfo]:
        """Finds a patch by name."""
        if name in self.find_cache:
            return self.find_cache[name]
        for p in self.patched:
            if p.name == name:
                self.find_cache[name] = p
                return p
        return None

    def append(self, patch: PatchInfo):
        """Adds a patch to the list of patches."""
        self.patched.append(patch)

    def extend(self, patches: Iterable[PatchInfo]):
        """Adds a patches to the list of patches."""
        self.patched.extend(patches)

    @property
    def n_patches(self) -> int:
        "Returns the number of stored patches."
        # Overwritten __len__ may have an impact on bool(patch_details: PatchDetails)
        return len(self.patched)

    def data(self) -> List[Dict[str, Any]]:
        """Returns the data for a dataframe."""
        return [p.to_dict() for p in self.patched]

    def patches_involved_in_graph(self, graph: Any) -> List[Tuple[PatchInfo, List[Any]]]:
        """
        Enumerates all patches impacting a graph.
        The function goes through the graph node (only the main graph) and
        looks into the metadata to determine if a listed patch was involved.

        :param graph: a graph object whose nodes can be iterated.
            The method is designed for :class:`torch.fx.Graph` but works with
            any object that satisfies the following minimal contract:

            * ``graph.nodes`` — iterable of node objects.
            * ``node.meta`` — a :class:`dict` attached to each node.
            * ``node.meta["stack_trace"]`` — a string containing the
              call-stack captured when the node was created.

            Any custom graph representation that provides these three
            attributes will work just as well as a native ``fx.Graph``.
        :return: list of nodes impacted by a patch
        """
        patches = []
        for patch in self.patched:
            f = patch.patch
            source = inspect.getsourcefile(f)
            lines, lineno = inspect.getsourcelines(f)
            interval = [lineno, lineno + len(lines)]
            patches.append((patch, f, source, interval))

        assert hasattr(graph, "nodes"), "graph has no attribute 'nodes'"
        cst = "yobx"
        node_stack = []
        for node in graph.nodes:
            assert hasattr(node, "meta"), "node has no attribute 'meta'"
            assert isinstance(node.meta, dict), "node.meta is not a dictionary"
            meta = node.meta
            if "stack_trace" not in meta:
                continue
            stack = meta["stack_trace"]
            if cst not in stack:
                # to reduce the cost of the next iteration
                continue
            node_stack.append((node, stack))

        patch_node = []
        patched_nodes = set()
        for patch, _f, source, interval in patches:
            exp = 'File "([^"]*?%s[^"]+?)", line (\\d+)' % cst
            reg = re.compile(exp)
            for node, stack in node_stack:
                occ = reg.findall(stack)
                if not occ:
                    continue
                for filename, line_number in occ:
                    if source and source.replace("\\", "/").strip("/") != filename.replace(
                        "\\", "/"
                    ).strip("/"):
                        continue
                    line = int(line_number)
                    if (
                        line >= interval[0]
                        and line <= interval[1]
                        and self.matching_pair(patch, node)
                    ):
                        patch_node.append((patch, node))
                        patched_nodes.add(id(node))

        # checks all patches were discovered
        for node, _ in node_stack:
            assert hasattr(node, "meta"), "node has no attribute 'meta'"
            assert id(node) in patched_nodes, (
                f"One node was patched but no patch was found:\n"
                f"node.meta={pprint.pformat(node.meta)}"
            )

        res = {}  # type: ignore[var-annotated]
        for patch, node in patch_node:
            if patch not in res:
                res[patch] = []
            res[patch].append(node)
        return list(res.items())

    def matching_pair(cls, patch: PatchInfo, node: "torch.fx.Node") -> bool:  # type: ignore # noqa: F821
        """
        Last validation for a pair. RotaryEmbedding has many rewriting
        and they all end up in the same code line.
        """
        f = patch.function_to_patch or patch._last_patched_function
        assert f is not None, f"The patch {patch.name!r} was never applied."
        cls_name = f.__qualname__.split(".")[0]
        if not cls_name.endswith("RotaryEmbedding"):
            return True
        return cls_name in str(node.meta)

    def make_report(
        cls,
        patches: List[Tuple[PatchInfo, List["torch.fx.Node"]]],  # type: ignore # noqa: F821
        format: str = "raw",
    ) -> str:
        """
        Creates a report based on the involved patches.

        :param patches: from method :meth:`patches_involved_in_graph`
        :param format: format of the report
        :return: report

        See example :ref:`l-plot-patch-model-diff`.

        ::

            -- original
            +++ rewritten
            @@ -1,6 +1,5 @@
            def _print_Symbol(self, expr: sympy.Symbol) -> str:
            -    if not isinstance(expr, sympy.Symbol):
            -        raise AssertionError(f"Expected sympy.Symbol, got {type(expr)}")
            -    if not self.symbol_to_source.get(expr):
            -        raise AssertionError(f"Unknown symbol {expr} created by constraints solver")
            -    return self.symbol_to_source[expr][0].name
            +    assert isinstance(expr, sympy.Symbol), str(type(expr))
            +    if self.symbol_to_source.get(expr):  # type: ignore
            +        return self.symbol_to_source[expr][0].name  # type: ignore
            +    return str(expr)
        """
        rows = []
        for patch, nodes in patches:
            rows.append(patch.format_diff(format=format))
            rows.append("")
            if format == "rst":
                rows.extend(["", "", "**impacted nodes**", "", "", ".. code-block::", ""])
            for node in nodes:
                rows.append(f"    {node.target}({', '.join(map(str,node.args))}) -> {node.name}")
            rows.append("")
        return "\n".join(rows)
