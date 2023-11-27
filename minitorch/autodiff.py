from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Set, Deque

from typing_extensions import Protocol
from collections import deque

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_p = tuple([x + epsilon if i == arg else x for i, x in enumerate(vals)])
    vals_m = tuple([x - epsilon if i == arg else x for i, x in enumerate(vals)])

    return (f(*vals_p) - f(*vals_m)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    if variable.is_constant():
        return []

    stack: Deque[Variable] = deque()
    visited: Set[int] = set()

    def topological_sort_helper(var: Variable, visited: Set[int], stack: Deque[Variable]) -> None:
        if var.unique_id not in visited:
            visited.add(var.unique_id)

            for parent in var.parents:
                if not parent.is_constant():
                    topological_sort_helper(parent, visited, stack)

            stack.append(var)

    topological_sort_helper(variable, visited, stack)
    return list(stack)[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    ders = {variable.unique_id: deriv}

    visit_order = topological_sort(variable)
    for var in visit_order:
        v_der = ders[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(v_der)
        else:
            for parent, p_der in var.chain_rule(v_der):
                ders[parent.unique_id] = ders.get(parent.unique_id, 0) + p_der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
