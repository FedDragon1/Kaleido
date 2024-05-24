import os

import subprocess
from pathlib import Path

from utils import RAW_ROOT, raw_path_to_out

from pdf2image import convert_from_path


def remove_aux_log(path: Path):
    for file in path.iterdir():
        if os.path.isdir(file):
            continue
        if not str(file).endswith(".webp"):
            os.remove(file.resolve())


def compile_to_pdf(path):
    save_path = raw_path_to_out(path)
    folder = save_path.parent

    if not os.path.exists(folder):
        os.makedirs(folder)

    args = fr"powershell.exe D:; pdflatex {path.resolve()} -output-directory='{folder.resolve()}';"

    p = subprocess.run(args)
    if p.returncode != 0:
        raise Exception("Error occurred in generating PDF.")

    pdf_name = save_path.parts[-1].removesuffix(".tex") + ".pdf"

    return folder / pdf_name


def generate_image(file_path):
    pdf = compile_to_pdf(file_path)
    images = convert_from_path(pdf)

    if len(images) > 1:
        raise Exception(f"Expected pdf length to be 1 page, found {len(images)}. ({pdf})")

    file_name = pdf.parts[-1].removesuffix(".pdf") + ".webp"
    path_to_img = pdf.parent

    images[0].save(path_to_img / file_name, 'WEBP')


def generate_tex_images(path):
    for children in path.iterdir():
        if os.path.isdir(children):
            generate_tex_images(children)
        else:
            generate_image(children)
    remove_aux_log(raw_path_to_out(path))


if __name__ == '__main__':
    generate_tex_images(RAW_ROOT)
