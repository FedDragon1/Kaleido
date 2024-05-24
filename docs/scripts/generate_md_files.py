import os
import string

from utils import PUBLIC_ROOT
from doc_extractor import retrieve_file_tree, ClassDoc


def join_if_not_none(args, symbol=", "):
    if args is None:
        return ''
    return symbol.join(args)


def par(s):
    if not s:
        return s
    return f"({s})"


def code(s):
    return f"`{s}`"


def codeblock(s):
    return f"```{s}```"


def s(s):
    return '' if s is None else s


def first_line(s):
    return s.split('\n')[0].strip(string.punctuation)


def extract_name_and_first_line_docstring(methods):
    return [(code(method['name']), first_line(s(method['docstring']))) for method in methods]


def convert_to_md_table(header, twod_seq):

    lines = [
        f"|{'|'.join(header)}|",
        f"|{'|'.join(['---'] * len(header))}|",
        *(
            f"|{'|'.join(entry)}|" for entry in twod_seq
        )
    ]

    return '\n'.join(lines)


def get_dd_for_doctring(docstring):
    docstring = process_docstring(docstring).split('\n\n')
    return ''.join(f"<dd>{line}</dd>" for line in docstring)


def get_detail_method(method):
    return f"""<dl>
    <dt><pre>{method['name']}({join_if_not_none(method['args'])})</pre></dt>
    {get_dd_for_doctring("No documentation.") if method['docstring'] is None else get_dd_for_doctring(method['docstring'])}
</dl>"""


def get_detail_method_list(methods):
    ret = (get_detail_method(method) for method in methods)
    return '\n'.join(ret)


def process_docstring(docstring):
    lines = docstring.split('\n\n')
    reunioned = [' '.join(line.split('\n')) for line in lines]
    return '\n\n'.join(reunioned)


def generate_md_for_class(cls: ClassDoc):
    qual_name = cls["name"].split(".")[-1]
    docstring = process_docstring(cls['docstring'])

    args = par(join_if_not_none(cls['args']))
    methods = convert_to_md_table(
        ("method", "shortened documentation"),
        extract_name_and_first_line_docstring(
            cls['methods']
        )
    )

    methods_detail = get_detail_method_list(cls['methods'])

    md = f"""# {qual_name}
    
Qualified name: {cls["name"]}

`class {qual_name}{args}`

{docstring}

## Methods

{methods}

---

{methods_detail}

"""

    return md


def generate_md_for_module(module):
    ...


def generate_md_for_file(file):
    ...


def generate_folders(qualname):
    os.makedirs(PUBLIC_ROOT / qualname.replace(".", "/"))


def create_files(tree):
    ...


if __name__ == '__main__':
    cls = {
                    'name': 'networks.engines.sequential.Sequential',
                    'args': None,
                    'docstring': 'Sequential model that flows data from layer to layer',
                    'methods': [
                        {'name': '__init__', 'docstring': None, 'args': ['self', '*layers', 'batch_size=32']},
                        {'name': '__repr__', 'docstring': None, 'args': ['self']},
                        {'name': 'add', 'docstring': None, 'args': ['self', 'layer']},
                        {
                            'name': 'layers_sanity_check',
                            'docstring': 'Checks whether the provided layers is legit,\nand calculates the index of layer that actually does meaningful calculation\ninstead of preprocessing the data',
                            'args': ['self']
                        },
                        {'name': 'forward', 'docstring': None, 'args': ['self', 'x']},
                        {'name': 'fit', 'docstring': None, 'args': ['self', 'x_train', 'y_train', 'epochs=100']},
                        {'name': 'compile', 'docstring': None, 'args': ['self', '*', 'optimizer', 'loss', 'metrics=None', 'input_shape=None']},
                        {'name': 'predict', 'docstring': 'Predicts a batch of inputs\n\n:param x_pred: array of input\n:return: array of prediction', 'args': ['self', 'x_pred']},
                        {'name': 'layers', 'docstring': None, 'args': ['self']},
                        {'name': 'input_shape', 'docstring': None, 'args': ['self']},
                        {'name': 'output_shape', 'docstring': None, 'args': ['self']},
                        {'name': 'total_params', 'docstring': None, 'args': ['self']},
                        {'name': 'save', 'docstring': None, 'args': ['self', 'path']},
                        {'name': 'plot_metrics', 'docstring': None, 'args': ['self']},
                        {'name': 'make_metric_histories', 'docstring': None, 'args': ['self']},
                        {'name': 'summary', 'docstring': None, 'args': ['self']}
                    ]
                }
    print(generate_md_for_class(cls))
