"""Retrieve the relevant policy sections (the R in RAG), with no API needed."""
import math
import re
from collections import Counter
from pathlib import Path
from pydantic import BaseModel
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'to', 'of', 'in', 'on', 'for', 'is', 'are', 'it', 'this', 'that', 'with', 'you', 'your', 'i', 'me', 'my', 'be', 'not', 'as', 'at', 'by', 'do', 'if'}

class Section(BaseModel):
    id: str
    title: str
    text: str

def tokenize(text: str) -> list[str]:
    """Lowercase, keep runs of letters/digits, drop STOPWORDS."""
    return [t for t in re.findall('[a-z0-9]+', text.lower()) if t not in STOPWORDS]

class PolicyIndex:

    def __init__(self, sections: list[Section]):
        """Store the sections and precompute what search() needs (term counts, IDF weights)."""
        self.sections = sections
        counts = [Counter(tokenize(s.title + ' ' + s.text)) for s in sections]
        n = len(sections)
        df = Counter(term for c in counts for term in c)
        self.idf = {term: math.log((n + 1) / (d + 1)) + 1 for term, d in df.items()}
        self.vectors = [self._weigh(c) for c in counts]

    def _weigh(self, counts: Counter) -> dict[str, float]:
        """TF-IDF weights for the terms we know; unknown terms are ignored."""
        return {t: tf * self.idf[t] for t, tf in counts.items() if t in self.idf}

    @classmethod
    def from_markdown(cls, path: str | Path) -> 'PolicyIndex':
        """Build an index from a policy file.

Each section starts with a line like "## P1 — Scams and spam":
id = the first word after "## " ("P1"), title = the text after the dash,
text = every following line up to the next "## " heading, stripped.
Lines before the first "## " heading are ignored."""
        sections: list[Section] = []
        current = None  # [id, title, [lines]] for the section being read
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.startswith('## '):
                if current:
                    sections.append(Section(id=current[0], title=current[1], text='\n'.join(current[2]).strip()))
                m = re.match('##\\s+(\\S+)\\s*[—–-]*\\s*(.*)$', line)
                current = [m.group(1), m.group(2).strip(), []]
            elif current is not None:
                current[2].append(line)
        if current:
            sections.append(Section(id=current[0], title=current[1], text='\n'.join(current[2]).strip()))
        return cls(sections)

    def search(self, query: str, k: int=2) -> list[Section]:
        """Return up to k sections most relevant to query, best first.

Score with TF-IDF cosine similarity (or BM25) over tokenize(title + text).
Sections scoring 0 are never returned; an empty or stopword-only query returns []."""
        q = self._weigh(Counter(tokenize(query)))
        q_norm = math.sqrt(sum(w * w for w in q.values()))
        if not q or q_norm == 0:
            return []
        scored = []
        for section, vec in zip(self.sections, self.vectors):
            dot = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if dot <= 0:
                continue
            v_norm = math.sqrt(sum(w * w for w in vec.values()))
            scored.append((dot / (q_norm * v_norm), section))
        scored.sort(key=lambda pair: pair[0], reverse=True)  # stable: ties keep file order
        return [section for _, section in scored[:k]]

def policy_context(sections: list[Section]) -> str:
    """Format sections for a prompt: "[<id>] <title>\\n<text>" each, joined by a blank line."""
    return '\n\n'.join(f'[{s.id}] {s.title}\n{s.text}' for s in sections)
