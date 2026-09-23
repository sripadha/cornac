"""The child-side program behind run_python: a notebook-cell runner.

Why this exists: the weak local model (Qwen 2.5 7B) keeps ending its snippet with a
bare expression — `total_revenue` rather than `print(total_revenue)` — the habit of
someone who lives in notebooks, where a cell shows the value of its last line. Under a
plain `python -c` that last line evaluates to a value nobody prints, the tool reports
"(no output)", and the model concludes its code did nothing and gives up. Echoing the
final expression's value, exactly as Jupyter/IPython do, meets the model where it is.
That is scaffolding — the harness closing a gap the model can't — which is this
project's thesis.

The rule mirrors IPython's "last_expr" mode: if the snippet's LAST top-level
statement is a bare expression (an `ast.Expr` node) and its value is not None, print
repr(value) after everything else has run. `print(x)` as the last line still prints
once — print returns None, so nothing extra is echoed. An assignment, def, class,
loop, import, or a call that returns None as the last statement echoes nothing.

How it stays honest about line numbers: the snippet's text is never rewritten. The
runner parses it with `ast`, pops the trailing Expr node if there is one, compiles the
remaining Module for exec and the popped expression for eval. Compiled AST nodes keep
their original `lineno`, so a syntax error on line 3 or a runtime error on line 2 is
reported at line 3 / line 2 — the lines the model wrote.

RUNNER_SOURCE is a string executed with `python -c`, not a module invoked with
`python -m`: the child runs with cwd = the workspace, which can be any directory, and
`-m` would need `cornac` to be importable from there (an editable install). With `-c`
the child needs nothing but the interpreter, and — like the plain `python -c` this
replaces — sys.path[0] is the cwd, so a snippet can still import modules that live in
the workspace. python_exec.py explains how the snippet itself reaches the child.
"""

from __future__ import annotations

# The name the snippet's frames carry in tracebacks. Distinct from "<string>", which is
# what the runner's own `-c` frames are called — that is how the two are told apart.
SNIPPET_FILENAME = "<run_python>"

RUNNER_SOURCE = f'''
import ast, linecache, os, sys, traceback, types

FILENAME = {SNIPPET_FILENAME!r}

# The snippet waits in a file whose path is our one argument. Popping it leaves
# sys.argv == ["-c"], exactly what the snippet saw under a plain `python -c`.
# "utf-8-sig" drops a leading byte-order mark, as `python file.py` would; left in, the
# tokenizer rejects it as "invalid non-printable character U+FEFF" on line 1.
path = sys.argv.pop(1)
with open(path, encoding="utf-8-sig") as f:
    source = f.read()
try:
    os.unlink(path)     # its job is done; nothing of ours should outlive the run
except OSError:
    pass                # python_exec.py deletes it anyway once the run is over

# Let tracebacks quote the offending source line, as they would for a real file.
linecache.cache[FILENAME] = (len(source), None, source.splitlines(True), FILENAME)

# A fresh module standing in as __main__: `if __name__ == "__main__":` blocks run, and
# functions defined in the snippet can be pickled (multiprocessing finds them by name).
main = types.ModuleType("__main__")
sys.modules["__main__"] = main

echoing = False         # set once the snippet is done and only our own echo remains
try:
    tree = ast.parse(source, FILENAME)          # a SyntaxError surfaces from here
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()                  # echoed below, after the rest has run
    exec(compile(tree, FILENAME, "exec"), main.__dict__)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), FILENAME, "eval"), main.__dict__)
        if value is not None:
            echoing = True
            print(repr(value))
except SystemExit:
    raise               # sys.exit(3) means "exit with 3", here as anywhere else
except BaseException as exc:
    # Drop the runner's own frames so the traceback starts at the snippet, the way a
    # plain `python -c` shows it. BaseException rather than Exception: a
    # KeyboardInterrupt escaping the snippet would otherwise reach the interpreter's
    # default handler, which prints every frame — ours included.
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != FILENAME:
        tb = tb.tb_next
    if tb is None and echoing:
        # The snippet ran and its value was computed; what failed is repr() or
        # print() of it, in our code — a __repr__ that returned a non-string, an int
        # too long for sys.get_int_max_str_digits(), a snippet that closed
        # sys.stdout. No snippet frame means no "line N" for the model, so name it.
        print(
            f"Error while echoing the value of the final expression (line {{last.lineno}}):",
            file=sys.stderr,
        )
    # A SyntaxError never reached the snippet either: no frame exists, and only the
    # error itself (file, line, caret) is printed — which is all the model needs.
    traceback.print_exception(type(exc), exc, tb)
    sys.exit(130 if isinstance(exc, KeyboardInterrupt) else 1)   # 130 = 128 + SIGINT
'''
