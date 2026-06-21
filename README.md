# Reversible Faithful Networks (RFN)

### Invertibility as a Tamper-Detection Certificate for Additive Attribution
*with a Two-Axis Completeness Theorem for Regression Hallucination Detection*

**Author:** Pulkit Srivastava
Independent Research · India
[GitHub](https://github.com/pulkit6732/aetherproof) · [LinkedIn](https://linkedin.com/in/pulkit-srivastava-2033942a6)

**Read the paper (live, rendered):** https://pulkit6732.github.io/rfn-paper/

Source file: [reversible_faithful_networks.html](./reversible_faithful_networks.html)

> **Status:** Preprint. Pending submission to arXiv (cs.LG).

**Code:** [github.com/pulkit6732/faithful-net](https://github.com/pulkit6732/faithful-net)
---

## Abstract

Post-hoc explanations for neural networks suffer from a fundamental *faithfulness gap*: the stated reasons for a model's decision need not reflect its actual computation, and no mechanism exists to catch a forged attribution. We introduce the **Reversible Faithful Network (RFN)**, an architecture that closes both problems by construction. Each per-feature contribution in an RFN is simultaneously (a) an exact additive component of the output logit and (b) an invertible encoding of the corresponding input feature via a strictly monotonic shape function. A forged or edited attribution is therefore mechanically detectable: running the explanation backward through the inverse map fails to reconstruct the original input.

We prove four properties experimentally — exact faithfulness (gap 1.78 × 10⁻¹⁵), lossless input reconstruction from explanation alone (error 1.33 × 10⁻¹⁵), lie-detection via reverse pass (mismatch 1.544 for a 50% feature overstatement), and competitive accuracy (92.4% vs 92.0% for an opaque MLP using 43 vs 75 parameters).

We further characterize a failure mode of single-axis hallucination detection — the *calibrated lie*, in which a bijective distortion of truth preserves mutual information yet is wholly wrong — and prove that a two-axis detector combining information (H) and grounding (G) is complete for value-honesty.

The core novelty is not invertible networks, additive models, or hallucination metrics individually, but their specific fusion: **reversibility as an integrity certificate on additive faithfulness**.

## Contributions

1. **Architecture (RFN)** — per-feature contributions that are simultaneously exact additive components of the output and invertible encodings of the input.
2. **Theorems** — formal proofs of exact faithfulness, exact reversibility, and lie-detectability by construction.
3. **Two-axis completeness** — a formal characterization of the calibrated-lie failure mode, with a proof that information + grounding detection is complete for value-honesty.

## Code & Data

All experiments are reproducible, pure-numpy, single-machine. Code lives in this repo / linked from [AetherProof](https://github.com/pulkit6732/aetherproof).

## Citation

```bibtex
@misc{srivastava2026rfn,
  author       = {Srivastava, Pulkit},
  title        = {Reversible Faithful Networks: Invertibility as a Tamper-Detection
                   Certificate for Additive Attribution},
  year         = {2026},
  howpublished = {\url{https://github.com/pulkit6732/rfn-paper}},
  note         = {Preprint, pending arXiv submission}
}
```

## License

Paper text: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code (where applicable): MIT

---

*This is a preprint and has not been peer-reviewed. Novelty claims are stated as corpus-verified-not-found (701,585 arXiv papers + live web, June 2026), not "provably first." Contributions and challenges welcome — open an issue.*
