import itertools
import os.path

import networks
import ast

from typing import TypedDict, Sequence, Optional, Union
from pathlib import Path
from rich import print

from docs.scripts.utils import PROJECT_ROOT


class FunctionDoc(TypedDict):
    name: str
    docstring: Optional[str]
    args: Sequence[str]


class ClassDoc(TypedDict):
    name: str
    args: Optional[Sequence[str]]
    docstring: Optional[str]
    methods: Sequence[FunctionDoc]


class FileDoc(TypedDict):
    type: str
    name: str
    classes: list[ClassDoc]
    functions: list[FunctionDoc]
    docstring: Optional[str]


class ModuleDoc(TypedDict):
    type: str
    name: str
    docstring: Optional[str]
    content: list[Union["ModuleDoc", FileDoc]]


def get_scoped_name(name, scope):
    if scope is None:
        return name
    return f"{scope}.{name}"


def retrieve_class_docstring(class_def, file_qual) -> ClassDoc:
    if not class_def.bases and not class_def.keywords:
        args = None
    else:
        args = [get_name_or_value(name) for name in class_def.bases]
        kwds = [f"{keyword.arg}={keyword.value.id}" for keyword in class_def.keywords]
        args.extend(kwds)

    current_tree = {
        "name": get_scoped_name(class_def.name, file_qual),
        "args": args,
        "docstring": ast.get_docstring(class_def, True),
        "methods": [],
    }

    function_defs = (x for x in class_def.body if isinstance(x, ast.FunctionDef))

    for function_def in function_defs:
        current_tree["methods"].append(get_function_doc(function_def, None))

    return current_tree


def get_name_or_value(x):
    if isinstance(x, ast.Constant):
        return x.value
    elif isinstance(x, ast.Name):
        return x.id
    elif isinstance(x, ast.Attribute):
        return f"{get_name_or_value(x.value)}.{x.attr}"
    raise Exception(f"unknown type: {type(x)}")


def get_args(args: ast.arguments):
    """
    positional only, /, args=defaults, (*vararg, *), kwonlyargs=kw_defaults, kwarg

    :param args:
    :return:
    """

    components = []

    if args.posonlyargs:
        components.extend(arg.arg for arg in args.posonlyargs)
        components.append("/")

    args_list = [
        f"{arg.arg}{'' if default is None else f'={get_name_or_value(default)}'}"
        for arg, default in itertools.zip_longest(args.args[::-1], args.defaults[::-1])
    ][::-1]

    components.extend(args_list)

    have_vararg = args.vararg is not None
    have_kwonly = bool(args.kwonlyargs)

    yes_vararg_no_kwonly = have_vararg and not have_kwonly
    yes_vararg_yes_kwonly = have_vararg and have_kwonly
    no_vararg_yes_kwonly = not have_vararg and have_kwonly

    if yes_vararg_yes_kwonly:
        components.append(f"*{args.vararg.arg}")
        kwonly_list = [
            f"{kwarg.arg}{'' if default is None else f'={get_name_or_value(default)}'}"
            for kwarg, default in zip(args.kwonlyargs, args.kw_defaults)
        ]
        components.extend(kwonly_list)
    elif yes_vararg_no_kwonly:
        components.append(f"*{args.vararg.arg}")
    elif no_vararg_yes_kwonly:
        components.append(f"*")
        kwonly_list = [
            f"{kwarg.arg}{'' if default is None else f'={get_name_or_value(default)}'}"
            for kwarg, default in zip(args.kwonlyargs, args.kw_defaults)
        ]
        components.extend(kwonly_list)

    if args.kwarg is not None:
        components.append(f"**{args.kwarg.arg}")

    return components


def get_function_doc(function_def, file_qual) -> FunctionDoc:
    return {
        "name": get_scoped_name(function_def.name, file_qual),
        "docstring": ast.get_docstring(function_def, True),
        "args": get_args(function_def.args),
    }


def retrieve_file_docstring(file_qual) -> FileDoc:
    current_tree = {"type": "file", "name": file_qual, "classes": [], "functions": [], "docstring": None}

    location = PROJECT_ROOT / (file_qual.replace(".", "/") + ".py")

    with open(location, "r", encoding="utf-8") as f:
        ast_tree = ast.parse(f.read())

    current_tree["docstring"] = ast.get_docstring(ast_tree, True)

    function_defs = (x for x in ast_tree.body if isinstance(x, ast.FunctionDef))
    class_defs = (x for x in ast_tree.body if isinstance(x, ast.ClassDef))

    for function_def in function_defs:
        current_tree["functions"].append(get_function_doc(function_def, file_qual))

    for class_def in class_defs:
        current_tree["classes"].append(retrieve_class_docstring(class_def, file_qual))

    return current_tree


def retrieve_file_tree_recursion(folder: Path, scope=None) -> ModuleDoc:
    current_tree = {"type": "module", "docstring": None, "name": scope, "content": []}
    for child in folder.iterdir():
        name = child.parts[-1]
        if os.path.isfile(child):
            if name == "__init__.py":
                current_tree["docstring"] = retrieve_file_docstring(
                    get_scoped_name(name.removesuffix(".py"), scope)
                )["docstring"]
            elif name.endswith(".py"):
                qual = get_scoped_name(name.removesuffix(".py"), scope)
                current_tree["content"].append(retrieve_file_docstring(qual))
        elif os.path.isdir(child) and name != "__pycache__":
            qual = get_scoped_name(name, scope)
            current_tree["content"].append(retrieve_file_tree_recursion(child, qual))
    return current_tree


def retrieve_file_tree():
    base_file = Path(networks.__file__).parent
    tree = retrieve_file_tree_recursion(base_file, scope="networks")
    return tree


if __name__ == "__main__":
    print(retrieve_file_tree())
