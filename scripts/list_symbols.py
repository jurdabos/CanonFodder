import ast
from pathlib import Path
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
py_files = PROJECT_ROOT.rglob("*.py")


def list_defs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{node.name}{'()' if isinstance(node, ast.FunctionDef) else ''}"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


report_lines = []
for file in sorted(py_files):
    if ".venv" in file.parts or ".idea" in file.parts:
        continue
    defs = list_defs(file)
    if defs:
        report_lines.append(f"{file.relative_to(PROJECT_ROOT)}")
        report_lines.append(textwrap.indent("\n".join(defs), "    "))
        report_lines.append("")

Path("function_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
print("Wrote function report")
