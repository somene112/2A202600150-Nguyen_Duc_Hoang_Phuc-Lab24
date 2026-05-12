import yaml, sys
with open(".github/workflows/eval-gate.yml") as f:
    doc = yaml.safe_load(f)

assert doc["on"]["pull_request"]["branches"] == ["main"], "trigger wrong"
assert "eval-gate" in doc["jobs"], "job missing"
steps = doc["jobs"]["eval-gate"]["steps"]
names = [s.get("name", "") for s in steps]
print("Steps:", names)

run_step = next(s for s in steps if "Run RAGAS" in s.get("name", ""))
assert "OPENAI_API_KEY" in run_step["env"], "OPENAI_API_KEY env missing"
assert "${{ secrets.OPENAI_API_KEY }}" in str(run_step["env"]["OPENAI_API_KEY"]), "secret ref wrong"

artifact_step = next(s for s in steps if "Upload" in s.get("name", ""))
assert artifact_step.get("if") == "always()", f"if:always() missing, got {artifact_step.get('if')}"
assert "ragas_results.csv" in artifact_step["with"]["path"], "artifact path wrong"

run_cmd = run_step.get("run", "")
assert "--threshold faithfulness=0.85" in run_cmd, "faithfulness threshold missing"
assert "--threshold answer_relevancy=0.80" in run_cmd, "answer_relevancy threshold missing"
assert "run_eval.py" in run_cmd, "run_eval.py not in run command"

print("[OK] YAML valid — all required fields present")
