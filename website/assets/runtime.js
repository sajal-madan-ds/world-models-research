import renderMathInElement from 'katex/contrib/auto-render';

const storageKey = 'world-models-course:v1';
const chapters = Array.from({ length: 12 }, (_, i) => String(i + 1).padStart(2, '0'));
function readProgress() {
  try {
    const data = JSON.parse(localStorage.getItem(storageKey) || '{}');
    return { completed: Array.isArray(data.completed) ? data.completed.filter(x => chapters.includes(x)) : [], last: typeof data.last === 'string' ? data.last : null };
  } catch { return { completed: [], last: null }; }
}
function writeProgress(data) {
  try { localStorage.setItem(storageKey, JSON.stringify(data)); return true; }
  catch { return false; }
}
function updateProgress() {
  const data = readProgress();
  for (const progress of document.querySelectorAll('[data-course-progress]')) progress.value = new Set(data.completed).size;
  for (const label of document.querySelectorAll('[data-course-progress-label]')) label.textContent = `${new Set(data.completed).size} of 12 chapters completed`;
  const resume = document.querySelector('[data-resume]');
  if (resume && data.last) {
    const destination = new URL(data.last, location.origin);
    if (destination.origin === location.origin && /\/docs\/\d{2}_/.test(destination.pathname)) {
      resume.href = destination.href;
      resume.textContent = 'Continue studying';
    }
  }
}
async function initialize() {
  const article = document.querySelector('article');
  if (!article) return;
  renderMathInElement(article, {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }, { left: '$', right: '$', display: false }],
    throwOnError: false,
    strict: 'ignore',
  });
  const tools = document.querySelector('[data-lesson]');
  if (tools) {
    const chapter = tools.dataset.lesson;
    const button = tools.querySelector('button');
    const status = tools.querySelector('[role="status"]');
    const refresh = () => {
      const complete = readProgress().completed.includes(chapter);
      button.setAttribute('aria-pressed', String(complete));
      button.textContent = complete ? 'Completed · mark unread' : 'Mark chapter complete';
    };
    const data = readProgress();
    data.last = location.pathname;
    writeProgress(data);
    refresh();
    button.addEventListener('click', () => {
      const next = readProgress();
      next.completed = next.completed.includes(chapter) ? next.completed.filter(x => x !== chapter) : [...next.completed, chapter];
      const saved = writeProgress(next);
      status.textContent = saved ? 'Progress saved in this browser.' : 'Browser storage is unavailable; progress could not be saved.';
      refresh();
      updateProgress();
    });
  }
  updateProgress();
  const diagrams = Array.from(article.querySelectorAll('.course-diagram'));
  if (diagrams.length) {
    try {
      const { default: mermaid } = await import('mermaid');
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: document.body.dataset.mdColorScheme === 'slate' ? 'dark' : 'neutral' });
      for (const [index, diagram] of diagrams.entries()) {
        const source = diagram.textContent;
        try {
          const { svg } = await mermaid.render(`course-diagram-${index}`, source);
          diagram.innerHTML = svg;
          diagram.setAttribute('role', 'img');
          diagram.setAttribute('aria-label', 'Architecture or concept diagram; equivalent explanation is provided in the surrounding text.');
        } catch {
          diagram.classList.add('diagram-fallback');
          diagram.textContent = `Diagram source (could not render):\n${source}`;
        }
      }
    } catch { /* The readable source remains available if loading fails. */ }
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
else initialize();
window.addEventListener('storage', updateProgress);
