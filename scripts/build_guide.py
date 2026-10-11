"""Render the local Russian guide, preserving TeX and using local KaTeX assets."""
from pathlib import Path
import html
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]
STYLE = '''
body {max-width: 1000px; margin: auto; padding: 36px 28px 90px; color: #202b36;
font: 18px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#fafbf8}
h1,h2,h3{line-height:1.25;color:#163b49}h2{margin-top:2.5em;border-top:1px solid #d5dfdd;padding-top:1em}
a{color:#075a91}pre{background:#eaf0ee;padding:20px;overflow:auto;border-radius:8px;font-size:14px;line-height:1.5}
code{font-size:.88em}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #cbd6d2;padding:9px;text-align:left}
th{background:#e7eeeb}img{max-width:100%}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}
.toc{background:#eaf0ee;padding:18px 24px;border-radius:8px}nav{font-size:15px;display:flex;gap:20px;flex-wrap:wrap}
@media print{body{font-size:11pt;padding:0}pre{white-space:pre-wrap}nav{display:none}h2{break-after:avoid}table{font-size:9pt}}
'''


def convert(source, target):
    formulas = []
    # Code fences/inline code are protected too, so dollar signs inside code
    # never turn into formulas. All placeholders are plain Markdown-safe words.
    chunks = []
    pattern = r'(```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+(?<!\\)\$)'
    def stash(match):
        token = match.group(0)
        if token.startswith('`'):
            return token
        key = f'MATHPLACEHOLDER{len(formulas)}END'
        formulas.append(token)
        return key
    text = re.sub(pattern, stash, source.read_text())
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
    for i, formula in enumerate(formulas):
        body = body.replace(f'MATHPLACEHOLDER{i}END', html.escape(formula))
    page = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(source.stem)} — MGM I2I</title>
<link rel="stylesheet" href="assets/katex/dist/katex.min.css"><style>{STYLE}</style></head><body>
<nav><a href="guide.html">Руководство</a><a href="verification.html">Проверки и результаты</a>
<a href="code-map.html">Карта кода</a><a href="../README.md">README</a></nav>
{body}
<script src="assets/katex/dist/katex.min.js"></script>
<script src="assets/katex/dist/contrib/auto-render.min.js"></script>
<script>renderMathInElement(document.body,{{delimiters:[{{left:"$$",right:"$$",display:true}},
{{left:"$",right:"$",display:false}}],throwOnError:false}});</script></body></html>'''
    target.write_text(page)
    return formulas


if __name__ == '__main__':
    all_formulas = []
    for name, target in [('GUIDE.md', 'guide.html'), ('VERIFICATION.md', 'verification.html'), ('CODE_MAP.md', 'code-map.html')]:
        source = ROOT/'offline'/name
        if source.exists():
            formulas = convert(source, ROOT/'offline'/target)
            all_formulas.extend(formulas)
            print(target, len(formulas), 'formulas')
    import json
    (ROOT/'offline/math-expressions.json').write_text(json.dumps(all_formulas, ensure_ascii=False, indent=2))
    (ROOT/'START_HERE.html').write_text('''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta http-equiv="refresh" content="0;url=offline/guide.html"><title>MGM I2I</title>
<a href="offline/guide.html">Открыть офлайн-руководство</a></html>''')
