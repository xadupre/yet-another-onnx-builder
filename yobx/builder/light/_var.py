from typing import Any, Dict, List, Optional, Union
import numpy as np
from onnx import TensorProto
from ._graph import OnnxGraph
from ._op_var import OpsVar
from ._op_vars import OpsVars


class BaseVar:
    """
    Base class for :class:`Var` and :class:`Vars`.

    :param parent: the :class:`OnnxGraph` that owns this variable
    """

    def __init__(self, parent: OnnxGraph):
        if not isinstance(parent, OnnxGraph):
            raise TypeError(f"Expected OnnxGraph, got {type(parent)}.")
        self.parent = parent

    def make_node(
        self,
        op_type: str,
        *inputs: Any,
        domain: str = "",
        n_outputs: int = 1,
        output_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Union["Var", "Vars"]:
        """
        Creates a node using *inputs* and returns the resulting :class:`Var`
        (single output) or :class:`Vars` (multiple outputs).

        :param op_type: operator type
        :param inputs: input :class:`Var` instances or numpy arrays
        :param domain: operator domain
        :param n_outputs: number of output tensors
        :param output_names: explicit output names
        :param kwargs: operator attributes
        :return: :class:`Var` or :class:`Vars`
        """
        node = self.parent.make_node(
            op_type,
            *inputs,
            domain=domain,
            n_outputs=n_outputs,
            output_names=output_names,
            **kwargs,
        )
        names = list(node.output)
        if len(names) == 1:
            return Var(self.parent, names[0])
        return Vars(self.parent, *[Var(self.parent, n) for n in names])

    # ------------------------------------------------------------------
    # Graph-level helpers accessible from any variable
    # ------------------------------------------------------------------

    def vin(
        self,
        name: str,
        elem_type: int = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ) -> "Var":
        """
        Declares a new graph input.

        :param name: input name
        :param elem_type: ONNX element type integer
        :param shape: optional shape
        :return: :class:`Var`
        """
        return self.parent.vin(name, elem_type=elem_type, shape=shape)

    def cst(self, value: np.ndarray, name: Optional[str] = None) -> "Var":
        """
        Adds a constant initializer.

        :param value: constant numpy array
        :param name: optional name
        :return: :class:`Var`
        """
        tensor = self.parent.make_constant(value, name=name)
        return Var(self.parent, tensor.name, elem_type=tensor.data_type, shape=tuple(tensor.dims))

    def v(self, name: str) -> "Var":
        """
        Retrieves a variable by name from the parent graph.

        :param name: variable name
        :return: :class:`Var`
        """
        return self.parent.get_var(name)

    def bring(self, *vars: Any) -> Union["Var", "Vars"]:
        """
        Combines variables into a :class:`Vars` (or a single :class:`Var`).

        :param vars: variable names or :class:`Var` instances
        :return: :class:`Var` or :class:`Vars`
        """
        if len(vars) == 1:
            v = vars[0]
            if isinstance(v, str):
                return self.parent.get_var(v)
            return v
        return Vars(self.parent, *vars)

    def left_bring(self, *vars: Any) -> "Vars":
        """Creates :class:`Vars` with ``*vars`` first, then ``self``."""
        return Vars(self.parent, *vars, self)

    def right_bring(self, *vars: Any) -> "Vars":
        """Creates :class:`Vars` with ``self`` first, then ``*vars``."""
        return Vars(self.parent, self, *vars)

    def to_onnx(self):
        "Delegates to the parent graph's :meth:`OnnxGraph.to_onnx`."
        return self.parent.to_onnx()

    def vout(self, **kwargs: Dict[str, Any]) -> Union["Var", "Vars"]:
        "Declare outputs - must be overridden in subclasses."
        raise NotImplementedError(f"vout() not implemented in {type(self).__name__}.")


class Var(BaseVar, OpsVar):
    """
    Represents a single ONNX variable (input, constant, node output, or graph output).

    :param parent: the owning :class:`OnnxGraph`
    :param name: variable name
    :param elem_type: ONNX element type integer (default: ``TensorProto.FLOAT``)
    :param shape: optional shape tuple
    """

    def __init__(
        self,
        parent: OnnxGraph,
        name: str,
        elem_type: Optional[int] = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ):
        BaseVar.__init__(self, parent)
        self.name_ = name
        self.elem_type = elem_type
        self.shape = shape

    @property
    def name(self) -> str:
        "Returns the current (possibly renamed) name."
        return self.parent.true_name(self.name_)

    def __str__(self) -> str:
        s = self.name
        if self.elem_type is not None:
            s = f"{s}:{self.elem_type}"
        if self.shape is not None:
            s = f"{s}:{list(self.shape)}"
        return s

    def __repr__(self) -> str:
        return f"Var({self.name!r})"

    def vout(
        self,
        elem_type: int = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ) -> "Var":
        """
        Declares this variable as a graph output.

        :param elem_type: ONNX element type integer
        :param shape: optional shape
        :return: ``self``
        """
        self.parent.make_output(self.name, elem_type=elem_type, shape=shape)
        return self

    def rename(self, new_name: str) -> "Var":
        """
        Renames this variable.

        :param new_name: new name
        :return: ``self`` (the variable now answers to the new name)
        """
        self.parent.rename(self.name, new_name)
        return self

    def to(self, to: int) -> "Var":
        "Casts to another ONNX element type."
        return self.Cast(to=to)

    def astype(self, to: int) -> "Var":
        "Casts to another ONNX element type (alias for :meth:`to`)."
        return self.Cast(to=to)

    def reshape(self, new_shape: Any) -> "Var":
        "Reshapes the variable."
        if isinstance(new_shape, tuple):
            cst = self.cst(np.array(new_shape, dtype=np.int64))
            return self.bring(self, cst).Reshape()
        return self.bring(self, new_shape).Reshape()

    # ------------------------------------------------------------------
    # Python operator overloads
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> "Var":
        return self.bring(self, other).Add()

    def __sub__(self, other: Any) -> "Var":
        return self.bring(self, other).Sub()

    def __mul__(self, other: Any) -> "Var":
        return self.bring(self, other).Mul()

    def __truediv__(self, other: Any) -> "Var":
        return self.bring(self, other).Div()

    def __matmul__(self, other: Any) -> "Var":
        return self.bring(self, other).MatMul()

    def __neg__(self) -> "Var":
        return self.Neg()

    def __abs__(self) -> "Var":
        return self.Abs()

    def __eq__(self, other: Any) -> "Var":  # type: ignore[override]
        return self.bring(self, other).Equal()

    def __ne__(self, other: Any) -> "Var":  # type: ignore[override]
        return self.bring(self, other).Equal().Not()

    def __lt__(self, other: Any) -> "Var":
        return self.bring(self, other).Less()

    def __le__(self, other: Any) -> "Var":
        return self.bring(self, other).LessOrEqual()

    def __gt__(self, other: Any) -> "Var":
        return self.bring(self, other).Greater()

    def __ge__(self, other: Any) -> "Var":
        return self.bring(self, other).GreaterOrEqual()

    def __mod__(self, other: Any) -> "Var":
        return self.bring(self, other).Mod()

    def __pow__(self, other: Any) -> "Var":
        return self.bring(self, other).Pow()


class Vars(BaseVar, OpsVars):
    """
    Represents a tuple of :class:`Var` instances to feed into a multi-input operator.

    :param parent: the owning :class:`OnnxGraph`
    :param vars: :class:`Var` instances or variable names
    """

    def __init__(self, parent: OnnxGraph, *vars: Any):
        BaseVar.__init__(self, parent)
        self.vars_: List[Var] = []
        for v in vars:
            if isinstance(v, str):
                self.vars_.append(self.parent.get_var(v))
            elif isinstance(v, Var):
                self.vars_.append(v)
            else:
                # Allow numpy arrays and scalar-like values by turning them into
                # constant Vars via the parent graph when possible. This mirrors
                # OnnxGraph.make_node(), which already accepts np.ndarray inputs.
                if isinstance(v, (np.ndarray, np.generic)) or np.isscalar(v):
                    const_maker = getattr(self.parent, "make_const", None)
                    if const_maker is None:
                        const_maker = getattr(self.parent, "const", None)
                    if const_maker is None or not callable(const_maker):
                        raise TypeError(
                            f"Expected Var or str, and no suitable constant "
                            f"factory found on parent graph to handle value "
                            f"of type {type(v)}."
                        )
                    const_var = const_maker(v)
                    self.vars_.append(const_var)
                else:
                    raise TypeError(f"Expected Var or str, got {type(v)}.")

    def __len__(self) -> int:
        return len(self.vars_)

    def __repr__(self) -> str:
        return f"Vars({', '.join(repr(v) for v in self.vars_)})"

    def _check_nin(self, n_inputs: int) -> "Vars":
        "Asserts the number of variables matches *n_inputs*."
        if len(self) != n_inputs:
            raise RuntimeError(f"Expected {n_inputs} inputs but got {len(self)}.")
        return self

    def __getitem__(self, index: int) -> Var:
        "Returns the :class:`Var` at position *index*."
        return self.vars_[index]

    def rename(self, *new_names: str) -> "Vars":
        """
        Renames each variable.

        :param new_names: one new name per variable
        :return: ``self``
        """
        if len(new_names) != len(self):
            raise ValueError(f"Expected {len(self)} names but got {len(new_names)}.")
        for var, new_name in zip(self.vars_, new_names):
            var.rename(new_name)
        return self

    def vout(
        self,
        elem_type: int = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ) -> "Vars":
        """
        Declares all variables as graph outputs.

        :param elem_type: ONNX element type integer
        :param shape: optional shape (applied to all outputs)
        :return: ``self``
        """
        for var in self.vars_:
            self.parent.make_output(var.name, elem_type=elem_type, shape=shape)
        return self
