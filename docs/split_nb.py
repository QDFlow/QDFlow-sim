import os
import re
from nbmanips import Notebook

notebook_path = os.path.join("..", "tutorial", "qdflow_tutorial.ipynb")
output_dir = os.path.join("source", "_tutorial")

def file_name(header:str):
    return re.sub(r"\W", "_", header.split("\n")[0].lstrip("#").strip()).lower() + ".ipynb"

if __name__ == "__main__":
    notebook = Notebook.read_ipynb(notebook_path)
    sections = notebook.select("has_html_tag", "h1").split_on_selection()

    for section in sections:
        header_idx = section.select("has_html_tag", "h1").first()
        if header_idx is not None:
            cell_text = section.cells[header_idx]["source"]
            fname = file_name(cell_text)
            if "qdflow_tutorial" in fname or "table_of_contents" in fname:
                continue
            section.to_ipynb(os.path.join(output_dir, fname))