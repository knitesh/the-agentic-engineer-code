## agent/safe.py — the sandboxed arithmetic evaluator from Chapter 2, kept as a
## shared primitive so the growing tool suite (Ch4, Ch5) imports one implementation.
import ast
import operator as op

## Only these AST node types and operators are permitted. Anything else
## (names, calls, attribute access, etc.) is rejected — no arbitrary code runs.
_ALLOWED_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

def safe_eval(expression: str) -> float:
    """Evaluate a numeric arithmetic expression safely via AST parsing.
    Supports + - * / // % ** and parentheses. Rejects anything else."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)
