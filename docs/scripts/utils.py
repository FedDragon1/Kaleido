from pathlib import Path


PROJECT_ROOT = Path("../../").resolve()
DOCS_ROOT = Path("..").resolve()
TEX_ROOT = Path("../tex").resolve()
RAW_ROOT = (TEX_ROOT / "raw").resolve()
PUBLIC_ROOT = (DOCS_ROOT / "public").resolve()


def raw_path_to_out(path):
    """
    Converts the path in raw directory to out.

    Due to the implementation detail there should be no intermediate
    folder named raw or out.

    :param path: original path in raw folder
    :return: path that swapped raw to out
    """
    return Path(*("out" if x == "raw" else x for x in path.parts))


def replace_escape(s):
    return s.replace("\\", "/")
