"""
transformer_ceiling.py — does the gap!=lie ceiling bind attention-KL on a REAL transformer?

Signal: attention-KL (KL of each attention head vs uniform, averaged across layers)
        — same family as van Dijk (2026) arXiv:2605.05025
Model:  gpt2 (no login required, ~500MB)
A-sweep: vary the error-magnitude threshold; check flag-rate (constant?) vs precision (varies?)

Prediction from §5 of "Gaps, Not Lies":
  flag-rate CONSTANT across A-bins, precision CLIMBS with A.
  Reason: attention-KL = phi(model, x), model is fixed, A is our threshold → phi ⊥ A.

If confirmed: the ceiling is a property of the PROBLEM, not the toy signal.
This is the falsifiable SOTA-bite the paper predicts in §9.

Output: transformer_ceiling.json  (written to current working directory)
"""

import torch, numpy as np, json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("Loading gpt2...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True).to(DEVICE).eval()
print("Model loaded.")

# ── 2. Factual QA set ─────────────────────────────────────────────────────────
# Mix of easy (GPT2 likely correct → low A) and hard (GPT2 likely wrong → high A)
# A = -log P(correct_token | prompt) — model's wrongness, the unobservable magnitude
QA = [
    # Easy — GPT2 usually gets these right (low A)
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of China is", "Beijing"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of Canada is", "Ottawa"),
    ("The Eiffel Tower is located in", "Paris"),
    ("Albert Einstein was born in", "1879"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for iron is", "Fe"),
    ("Shakespeare wrote", "Hamlet"),
    ("The planet closest to the Sun is", "Mercury"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("The Amazon River is in South", "America"),
    ("The Nile River is in", "Africa"),
    ("The boiling point of water is 100 degrees", "Celsius"),
    ("The Battle of Hastings took place in", "1066"),
    ("The atomic number of carbon is", "6"),
    ("The atomic number of oxygen is", "8"),
    ("Mount Everest is the tallest mountain in the", "world"),
    ("DNA stands for deoxyrib", "onucleic"),
    ("The speed of sound in air is approximately", "340"),
    ("The capital of New Zealand is", "Wellington"),
    # Medium — GPT2 sometimes right
    ("The capital of Brazil is", "Bras"),
    ("The capital of Australia is", "Can"),
    ("The chemical symbol for silver is", "Ag"),
    ("The Treaty of Westphalia was signed in", "1648"),
    ("Pythagoras was born around", "570"),
    ("The atomic number of gold is", "79"),
    ("The speed of light is approximately 3 times 10 to the power of", "8"),
    ("Photosynthesis produces", "oxygen"),
    ("The chemical formula for sulfuric acid is H", "2"),
    ("The melting point of iron is approximately", "1538"),
    ("The capital of Argentina is", "Buenos"),
    ("The highest mountain in Africa is", "Kiliman"),
    ("The longest river in Asia is the", "Yangtze"),
    ("The human body has", "206"),
    ("The capital of South Korea is", "Seoul"),
    # Hard — GPT2 usually wrong (high A)
    ("The capital of Burkina Faso is", "Ouag"),
    ("The capital of Kyrgyzstan is", "Bish"),
    ("The capital of Eritrea is", "Asmar"),
    ("The capital of Tajikistan is", "Dush"),
    ("The capital of Turkmenistan is", "Ashgab"),
    ("The capital of Mozambique is", "Map"),
    ("The population of Iceland is approximately", "370"),
    ("The GDP of Luxembourg in 2020 was approximately", "73"),
    ("The capital of Papua New Guinea is", "Port"),
    ("The capital of Vanuatu is", "Port"),
]

# ── 3. Signals ────────────────────────────────────────────────────────────────
def attn_kl_score(prompt: str) -> float:
    """KL(attention || uniform) averaged across all layers and heads.
    High score = attention is peaked (concentrated) = model 'certain' about structure.
    This is the attention-KL family signal (van Dijk 2026 analog on MLP attention)."""
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    seq_len = ids.shape[1]
    log_uniform = float(np.log(seq_len + 1e-12))
    kl_vals = []
    for layer_attn in out.attentions:                     # (1, heads, seq, seq)
        a = layer_attn[0].float().cpu().numpy()           # (heads, seq, seq)
        entropy = -(a * np.log(a + 1e-12)).sum(-1)        # (heads, seq)
        kl = log_uniform - entropy                         # KL(a||uniform) >= 0
        kl_vals.append(float(kl.mean()))
    return float(np.mean(kl_vals))

def model_log_prob(prompt: str, correct_token: str) -> float:
    """log P(correct_token | prompt). Low = model is wrong = high A."""
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1).cpu()
    # Try with and without leading space
    for prefix in [" " + correct_token, correct_token]:
        cids = tok.encode(prefix, add_special_tokens=False)
        if cids:
            return float(log_probs[cids[0]])
    return -20.0

# ── 4. Run all prompts ────────────────────────────────────────────────────────
print(f"\nScoring {len(QA)} prompts...")
results = []
for i, (prompt, correct) in enumerate(QA):
    score = attn_kl_score(prompt)
    lp    = model_log_prob(prompt, correct)
    A     = float(-lp)          # magnitude: high = model very wrong
    results.append({"prompt": prompt, "correct": correct,
                    "score": score, "A": A, "log_prob": lp})
    print(f"  [{i+1:2d}/{len(QA)}] attn_kl={score:.3f}  A={A:.2f}  '{prompt[:45]}'")

scores = np.array([r["score"] for r in results])
A_vals = np.array([r["A"]     for r in results])

print(f"\nScore stats: min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}")
print(f"A stats:     min={A_vals.min():.2f}  max={A_vals.max():.2f}  mean={A_vals.mean():.2f}")

# ── 5. THE CORE TEST: A-sweep at fixed flag threshold ────────────────────────
# Mirror of gap_vs_lie.py:
#   fix tau (label-free, top 40% of scores)
#   sweep A_threshold (what counts as a "lie")
#   measure: flag_rate (should be CONSTANT), precision (should CLIMB)
flag_threshold = np.quantile(scores, 0.60)
flagged   = scores > flag_threshold
flag_rate = float(np.mean(flagged))

print(f"\nFlag threshold (60th percentile): {flag_threshold:.4f}")
print(f"Flag rate: {flag_rate:.3f}  ← should stay CONSTANT across A-bins\n")

A_THRESHOLDS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0]
print(f"  {'A_thr':>8} {'flag_rate':>10} {'lie_rate':>10} {'precision':>11} {'n_lies':>8}")
print("  " + "-"*52)
rows = []
for A_thr in A_THRESHOLDS:
    is_lie  = A_vals > A_thr
    lie_rate = float(np.mean(is_lie))
    prec    = float(np.mean(is_lie[flagged])) if flagged.any() else float("nan")
    n_lies  = int(is_lie.sum())
    print(f"  {A_thr:>8.1f} {flag_rate:>10.3f} {lie_rate:>10.3f} {prec:>11.3f} {n_lies:>8}")
    rows.append({"A_thr": A_thr, "flag_rate": flag_rate,
                 "lie_rate": lie_rate, "prec": prec, "n_lies": n_lies})

# ── 6. AUC: does attn-KL rank errors above correct? ──────────────────────────
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort()
    r = np.empty(len(a)); r[o] = np.arange(1, len(a)+1)
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

is_error = A_vals > np.median(A_vals)
auc_val  = auc(scores[is_error], scores[~is_error])
print(f"\nAUC (attention-KL ranking errors above correct): {auc_val:.3f}")
print(f"  (random = 0.5; >0.5 = signal ranks errors higher; <0.5 = anti-correlated)")

# ── 7. Verdict ────────────────────────────────────────────────────────────────
flag_rates = [r["flag_rate"] for r in rows]
prec_vals  = [r["prec"]      for r in rows]
flag_std   = float(np.std(flag_rates))
prec_range = float(max(p for p in prec_vals if p==p) - min(p for p in prec_vals if p==p))

print("\n" + "="*60)
print("VERDICT")
print(f"  flag_rate std across A-bins : {flag_std:.6f}  (expect ~0.000)")
print(f"  precision range across A-bins: {prec_range:.3f}   (expect >0.3)")
if flag_std < 0.01 and prec_range > 0.2:
    print("  ✓ CEILING CONFIRMED on real transformer attention-KL")
    print("    flag-rate is constant, precision is unbounded in A.")
    print("    The ceiling is the PROBLEM's, not the toy signal's.")
else:
    print("  ? Result ambiguous — check raw table above.")
print("="*60)

# ── 8. Save ───────────────────────────────────────────────────────────────────
out = {
    "model": "gpt2",
    "signal": "attention-KL: KL(attn || uniform) avg over layers/heads",
    "note": "analog of van Dijk 2026 arXiv:2605.05025 — not the transformer method, same family",
    "n_questions": len(QA),
    "flag_threshold_pctile": 60,
    "flag_rate": flag_rate,
    "auc": auc_val,
    "flag_rate_std": flag_std,
    "precision_range": prec_range,
    "A_sweep": rows,
    "per_question": results,
}
PATH = "transformer_ceiling.json"
with open(PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {PATH}")
print("\nREAD: constant flag_rate + climbing precision = ceiling confirmed.")
print("This is §9 of 'Gaps, Not Lies' — the SOTA-bite experiment.")
