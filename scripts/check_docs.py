"""Check local links, Python snippet syntax, control characters and fenced blocks."""

import ast
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors = []
for path in [root / "README.md", *root.glob("docs/*.md"), *root.glob("experiments/*/README.md")]:
    text = path.read_text()
    for c in text:
        if ord(c) < 32 and c not in "\n\t\r":
            errors.append(f"{path.name}: control char {ord(c)}")
    if text.count("```") % 2:
        errors.append(f"{path.name}: unmatched fence")
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1).split("#")[0]
        if (
            target
            and not target.startswith(("http:", "https:", "mailto:", "/"))
            and not (path.parent / target).exists()
        ):
            errors.append(f"{path.name}: missing {target}")
    for i, code in enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL)):
        try:
            ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"{path.name}: snippet {i}: {exc}")
if errors:
    raise SystemExit("\n".join(errors))
print("Documentation links, fences, control characters and Python syntax passed")
