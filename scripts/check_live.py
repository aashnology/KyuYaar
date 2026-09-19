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
model_chose = [s["tool"] for s in inv.steps][:3]
print(f"Steps run: {len(inv.steps)}   readouts written by the model: {llm_steps}")
print(f"Summary written by: {inv.summary_source}")
print(f"Guardrail blocks: {len(inv.guardrail_blocks)}")
for note in inv.notes:
    print("Note:", note)
for block in inv.guardrail_blocks:
    print("Blocked text had unsupported figures:", block["unsupported"])

supported = sorted(e.id for e in inv.evidence if e.evidence_type == "statistical" and e.strength != "weak")
print("Supported causes:", supported)

ok = llm_steps > 0 and not inv.notes
print("LIVE PATH OK" if ok else "LIVE PATH DID NOT COMPLETE - see notes above")
sys.exit(0 if ok else 1)
