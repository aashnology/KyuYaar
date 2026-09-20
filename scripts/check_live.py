"""
One-command check of the live model path.

Set GEMINI_API_KEY (or ANTHROPIC_API_KEY) in your shell first, then run:
    python scripts/check_live.py

It runs one real investigation and reports whether the model actually drove
it, without printing any credentials. Exit code 0 means the live path worked;
1 means the run fell back to offline mode or failed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data
from orchestrator import investigate, make_client
from toolkit import Toolkit

client = make_client()
if client is None:
    print("No API key found. Set GEMINI_API_KEY (PowerShell: $env:GEMINI_API_KEY=\"...\") and retry.")
    sys.exit(1)

print(f"Provider: {client.provider}   model: {client.model}")
orders, marketing = load_data()
events = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing), client=client))
inv = events[-1].data["investigation"]

llm_steps = sum(1 for s in inv.steps if s["narration_source"] == "llm")
filled = sum(1 for n in inv.notes if "did not run" in n)
print(f"Tools run: {len(inv.steps)}   chosen by the model: {len(inv.steps) - filled}   filled in afterwards: {filled}")
print(f"Readouts written by the model: {llm_steps}   Summary written by: {inv.summary_source}")
print(f"Guardrail blocks: {len(inv.guardrail_blocks)}")
for note in inv.notes:
    print("Note:", note)
for block in inv.guardrail_blocks:
    print("Blocked text had unsupported figures:", block["unsupported"])

supported = sorted(e.id for e in inv.evidence if e.evidence_type == "statistical" and e.strength != "weak")
print("Supported causes:", supported)

# The live path worked if the model drove the run without an API failure. A
# blocked passage is not a failure: it means the guardrail did its job.
failed = any("model call failed" in n for n in inv.notes)
drove_it = (len(inv.steps) - filled) > 0 and (llm_steps > 0 or inv.summary_source == "llm" or inv.guardrail_blocks)
ok = drove_it and not failed
print("LIVE PATH OK" if ok else "LIVE PATH DID NOT COMPLETE - see notes above")
sys.exit(0 if ok else 1)
