# Course website

The teaching website renders the original `docs/`, `experiments/`, and `research/`
Markdown directly. These remain the authoritative, downloadable backups. The
homepage and study aids live in `website/content/`. No paid services are needed.

## Build and preview

From the repository root (Python 3.12, uv, Node.js, and npm required):

```sh
uv sync --project website --python 3.12
npm ci --prefix website
npm run build --prefix website
uv run --project website mkdocs build --strict
uv run --project website mkdocs serve
```

Open the local address printed by MkDocs. Math, diagrams, and fonts are served
locally: the site does not require external rendering APIs or font CDNs.
Chapter completion is stored only in the reader's browser, not in GitHub or a
user account. Clearing browser storage clears progress.

## Browser checks

```sh
npx --prefix website playwright install chromium
npm test --prefix website
```

The tests build their own local HTTP server and check all chapters, rendered
equations and diagrams, chapter completion, search, and mobile overflow.

## Publish updates

Commit and push changes to `main`, rebuild the assets, then run from the root:

```sh
uv run --project website mkdocs gh-deploy --strict
```

GitHub Pages serves the generated `gh-pages` branch. Generated HTML is not mixed
with the Markdown source. Updates are intentionally explicit, not automatically
published on every push. Dependency lockfiles pin the tested environment.
