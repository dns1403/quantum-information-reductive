"""Verify document references and source fingerprints without compiling LaTeX."""
import hashlib
import csv
import json
import re
from pathlib import Path

def expand_inputs(path, seen=None):
    seen=set() if seen is None else set(seen)
    assert path not in seen, f"Recursive input: {path}"
    seen.add(path)
    source=Path(path).read_text()
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda m: expand_inputs(m.group(1),seen),source)

report = expand_inputs("report.tex")
bib = Path("references.bib").read_text()
keys = re.findall(r"@\w+\{([^,]+),", bib)
assert len(keys) == len(set(keys)), "Duplicate bibliography key"
cited = set()
for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", report):
    cited.update(group.split(","))
assert cited <= set(keys), f"Missing bibliography entries: {cited-set(keys)}"
labels = re.findall(r"\\label\{([^}]+)\}", report)
assert len(labels) == len(set(labels)), "Duplicate label"
refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", report))
assert refs <= set(labels), f"Missing labels: {refs-set(labels)}"
stack = []
for mode, env in re.findall(r"\\(begin|end)\{([^}]+)\}", report):
    if mode == "begin":
        stack.append(env)
    else:
        assert stack and stack.pop() == env, (mode, env, stack)
assert not stack
note=expand_inputs('notes/hecke_positivity.tex')
note_labels=re.findall(r"\\label\{([^}]+)\}",note)
assert len(note_labels)==len(set(note_labels)), 'Duplicate standalone-note label'
note_refs=set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}",note))
assert note_refs<=set(note_labels),f'Standalone note has external references: {note_refs-set(note_labels)}'
note_cites=set()
for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}",note):
    note_cites.update(group.split(','))
assert note_cites<=set(keys)
with Path('scripts/results/claim_ledger.csv').open(newline='') as f:
    claims=list(csv.DictReader(f))
assert len({row['id'] for row in claims})==len(claims)
assert all(row['id']+' &' in report for row in claims), 'Ledger/report mismatch'
for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", report):
    assert Path(image).is_file(), image
assert "\\hline" not in report
experiments=re.findall(r"\\subsection\{(E\d+):",report)
for experiment in experiments:
    start = report.index("\\subsection{"+experiment)
    # Mathematical environments can intervene before the remaining fields.
    end = report.find("\\subsection{", start+12)
    text = report[start:end if end != -1 else None]
    for field in ["Goal", "Hypothesis", "Method", "Implementation", "Results", "Analysis", "Next steps"]:
        assert "\\paragraph{"+field+"}" in text, (experiment, field)
    assert "\\toprule" in text and "\\bottomrule" in text
expected = "fd5142f3ef6cd487c66bb122b8defbb95fa8dcf0e3f2ef394338697d0584ad55"
local = Path("smooth.pdf")
notes_status = "local source not present; expected hash recorded"
if local.exists():
    assert hashlib.sha256(local.read_bytes()).hexdigest() == expected
    notes_status = "supplied PDF hash matches audited source"
online = Path("artifacts/sources/smooth_online.pdf")
if online.exists():
    assert hashlib.sha256(online.read_bytes()).hexdigest() == expected
    notes_status += "; fetched author PDF byte-identical"
result = {"status": "PASS", "bibliography_entries": len(keys), "cited_entries": len(cited),
          "labels": len(labels), "references_resolved": len(refs),
          "environments_balanced": True, "experiment_fields_complete": True,
          "experiments":experiments,"shared_note_input_resolved":True,
          "standalone_note_references_resolved":True,"claim_ledger_rows":len(claims),
          "notes_sha256": expected, "notes_check": notes_status,
          "latex_compiled": False}
Path("scripts/results/document_audit.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
