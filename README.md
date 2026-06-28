# Faithfulness, Gaps, and Lies: A Unified Theory of Neural Model Honesty

**Pulkit Srivastava · Independent Researcher · India · June 2026**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20993925.svg)](https://doi.org/10.5281/zenodo.20993925)

**[Read the paper](https://pulkit6732.github.io/rfn-paper/)** · 
**[Zenodo deposit](https://zenodo.org/records/20993925)** · 
**[Prior work (RFN v1)](https://zenodo.org/records/20782418)**

---

## One-sentence result

Single-model reference-free hallucination detection is provably 
uncalibratable — the best internal signal detects *gaps*, not *lies* — 
and the lie/no-lie distinction inside a gap is information only 
external grounding can supply.

## Four results

1. **Conservation Law** — L(f) = E[|log|det J(x)||] = 0 iff faithful. 
   Verified to <7×10⁻¹¹ nats. Two-directional extension catches 
   fabrication (1.0986 nats). Quantum no-hiding gap = 1.01×10⁻¹³.

2. **Gauge Correction** — |det J|=1 is necessary but NOT sufficient. 
   Gauge-twin achieves 16.89× attribution swing (1689%) while passing 
   every determinant test. L_attr catches it at 5.163 nats.

3. **NTK Identity** — training-support concentration = NTK-GP posterior 
   variance. Verified to 3.27×10⁻¹³ over n=300 networks. AUC 0.859±0.084.

4. **Detection Ceiling** — ∂v/∂A = 0. Flag-rate locked constant while 
   precision spans 0.000→0.928. Confirmed across four architectures 
   including GPT-2 (117M parameters).

## Reproduce everything

```bash
# Clone and run any experiment — no arguments needed
git clone https://github.com/pulkit6732/rfn-paper
cd rfn-paper
pip install numpy scipy scikit-learn torch transformers
python transformer_ceiling.py   # → transformer_ceiling.json
```

Every claim maps to one JSON file. See the reproducibility ledger 
in the paper (Appendix A).

## Files

| File | Purpose |
|------|---------|
| `index.html` | The paper (rendered at pulkit6732.github.io/rfn-paper) |
| `transformer_ceiling.py` | GPT-2 ceiling experiment |
| `thermo_results.json` | Conservation law sweep |
| `two_directional_results.json` | Fabrication lie = 1.0986 nats |
| `nohiding_results.json` | Quantum gap = 1.01×10⁻¹³ |
| `gauge_results.json` | Gauge-twin 1689%, repair 9.2×10⁻¹⁷ |
| `summary.json` | NTK identity 3.27×10⁻¹³, AUC 0.859 |
| `sota_bite.json` | Two-signal universality table |
| `gap_vs_lie.json` | Precision 0.021→0.928, flag=0.429 |
| `real_test.json` | UCI wine 11-D, AUC 0.807 |
| `transformer_ceiling.json` | GPT-2 flag_std=0.000 |

## Citation

```bibtex
@misc{srivastava2026faithfulness,
  author    = {Srivastava, Pulkit},
  title     = {Faithfulness, Gaps, and Lies: A Unified Theory 
               of Neural Model Honesty},
  year      = {2026},
  doi       = {10.5281/zenodo.20993925},
  url       = {https://zenodo.org/records/20993925},
  note      = {Preprint}
}
```

## License

Paper: CC BY 4.0 · Code: MIT
