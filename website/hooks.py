"""Publish original Markdown as virtual site files; never edit the backup sources."""

import html
import re
from pathlib import Path
from urllib.parse import quote

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/sajal-madan-ds/world-models-research"
SOURCE_PAGES = {}


def on_files(files, config):
    SOURCE_PAGES.clear()
    sources = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
               *sorted((ROOT / "research").glob("*.md")),
               *sorted((ROOT / "experiments").glob("*/README.md"))]
    for source in sources:
        relative = source.relative_to(ROOT).as_posix()
        uri = "lab.md" if relative == "README.md" else relative
        SOURCE_PAGES[uri] = source
        files.append(File.generated(config, uri, content=source.read_text()))
    built = ROOT / "website/built"
    if not (built / "runtime.js").exists():
        raise RuntimeError("Website assets missing. Run: cd website && npm ci && npm run build")
    for source in sorted(built.rglob("*")):
        if source.is_file() and source.suffix != ".md":
            files.append(File.generated(config, "assets/" + source.relative_to(built).as_posix(),
                                        abs_src_path=source))
    for name in ["course.css", "logo.svg"]:
        files.append(File.generated(config, f"assets/{name}", abs_src_path=ROOT / "website/assets" / name))
    plots = {
        "ball-trajectories.png": "results/20261008T074315248466Z/02_seed0/trajectories.png",
        "latent-rollout.png": "results/20261008T074315248466Z/04_seed0/latent_rollout.png",
        "jepa-rollout.png": "results/20261008T083404498291Z/05_seed0/latent_rollout.png",
        "value-geometry.png": "results/20261008T074315248466Z/09_seed0/value_geometry.png",
    }
    for name, path in plots.items():
        files.append(File.generated(config, f"assets/plots/{name}", abs_src_path=ROOT / path))
    return files


def on_page_markdown(markdown, page, config, files):
    source = SOURCE_PAGES.get(page.file.src_uri)
    if source is None:
        return markdown
    page.edit_url = GITHUB + "/edit/main/" + quote(source.relative_to(ROOT).as_posix())
    # Keep Markdown in one authoritative location; convert math delimiters for the site only.
    parts = re.split(r"(```.*?```|`[^`\n]+`)", markdown, flags=re.DOTALL)
    for index in range(0, len(parts), 2):
        text = parts[index]
        text = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n$$\n" + m[1].strip() + "\n$$\n",
                      text, flags=re.DOTALL)
        text = re.sub(r"\\\((.*?)\\\)", lambda m: "$" + m[1] + "$", text)

        def link(match):
            target = match[1]
            if target.startswith(("https:", "http:", "mailto:", "#")):
                return match[0]
            local, _, anchor = target.partition("#")
            destination = (source.parent / local).resolve()
            try:
                relative = destination.relative_to(ROOT).as_posix()
            except ValueError:
                return match[0]
            if destination.is_file() and relative not in SOURCE_PAGES:
                url = GITHUB + "/blob/main/" + quote(relative)
                return "](" + url + ("#" + anchor if anchor else "") + ")"
            return match[0]

        parts[index] = re.sub(r"\]\(([^)]+)\)", link, text)
    markdown = "".join(parts)
    lesson = re.match(r"docs/(\d{2})_", page.file.src_uri)
    if lesson:
        minutes = max(1, round(len(markdown.split()) / 180))
        number = lesson[1]
        tools = (f'<section class="lesson-tools" data-lesson="{html.escape(number)}" '
                 'aria-label="Chapter study tools">'
                 f'<span>Chapter {number} / 12 · About {minutes} min to read</span>'
                 '<button type="button" aria-pressed="false">Mark chapter complete</button>'
                 '<span class="lesson-status" role="status" aria-live="polite"></span></section>')
        markdown = re.sub(r"^(# [^\n]+\n)", lambda m: m[1] + "\n" + tools + "\n", markdown, count=1)
    return markdown
