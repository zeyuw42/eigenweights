# Section 5 computations

Exact SageMath computations for Section 5 of *Arithmetic Volume of Shtukas
and Langlands Duality* by Zeyu Wang and Wenqing Wei. Equation and table
references in the code refer to this paper:

- [arXiv:2604.03763](https://arxiv.org/abs/2604.03763)
- [Backup PDF on Zeyu Wang's homepage](https://zeyuw42.github.io/assets/volumeofshtukas.pdf)

## Two versions

- **AI-generated verifier:** `verify_section5.py`, with
  `section5_formulas.py`, `section5_representations.py`, and
  `section5_tables.py`. It compares the Section 5 formulas with direct
  representation calculations from the paper's definitions.
- **Human-written computations, collected and refactored by AI:**
  `old/compute_section5.py`. This standalone script assembles Zeyu Wang's
  Sage computations into a calculator and table checker. The human-written
  version is collected in `old/`; the AI-generated verifier is in the root folder.

Both versions use the paper's formulas. Sage supplies exact arithmetic,
permutations, multinomial coefficients, root systems, Lie algebras,
exterior powers, and eigenvalues. Stored table entries are used only as
expected answers in comparisons.

## Requirements and quick start

Install [SageMath](https://www.sagemath.org/) and make `sage` available in
your terminal. The scripts have been tested with **SageMath 10.7** and
require no additional Python packages. Run these commands from this folder:

```sh
sage -python verify_section5.py
sage -python old/compute_section5.py
```

The verifier reports **443 passing comparisons and 0 failures** with its
default settings. The standalone calculator reports **126 passing
comparisons and 0 failures**. Both commands check all seven numbered tables
and exit with a nonzero status if a comparison fails. Exceptional
representations, particularly E8, may take a few minutes.

To save every comparison:

```sh
sage -python verify_section5.py --report results/verification.txt
sage -python old/compute_section5.py --report results/tables.txt
```

If you download the paper, you can optionally add `--paper /path/to/paper.pdf`
to record its SHA-256 hash in the verification report. The scripts run
without a local PDF.

## Individual calculations

The calculator's `--type` specifies **G-check**, as in Tables 1–7.
`--weight k` selects the paper's fundamental weight `varpi_k`.

```sh
sage -python old/compute_section5.py --type D --rank 4 --weight 2
sage -python old/compute_section5.py --type D --rank 6 --weight 5
sage -python old/compute_section5.py --type E --rank 8 --representation adjoint
```

For classical types A, B, C, D, specify a fundamental weight. For G2, F4,
E6, and E7, use `--representation minimal` or `--representation adjoint`;
E8 supports `adjoint`. Output includes `d_lambda`, `deg_lambda`, `s`, `t`,
the eigenweights grouped by invariant-polynomial degree, and `b_lambda`.
For even type D, the repeated-degree block is evaluated as the full
2-by-2 matrix in (5.36); its eigenvalues are an unordered pair and may be
algebraic numbers.

## What is checked

The AI-generated verifier checks:

- The classical formulas (5.10)–(5.57) against (5.1), (5.2), and (4.5),
  for every fundamental weight through rank 4. The checks include the
  type D bases and both doubled half-spin embeddings in Section 5.1.7.
- The reduction identities (5.3)–(5.6) and (5.58), using tensor products
  of A2 fundamental representations, including composite weights.
- Proposition 5.2, cyclic examples of Proposition 5.3, and the
  minuscule formula and unnumbered table in Section 5.2.2.
- Tables 1–4 from both the classical formulas and representation
  calculations; Tables 5–7 from (4.5), Corollary 4.3, and Theorem 4.4.

The standalone calculator reproduces Tables 1–7. It uses the classical
formulas for eigenweights, (4.5) for exceptional representations, and
Theorem 4.4 or Proposition 5.13 for `b_lambda`.

These are exact checks in a finite range, not a proof for every rank or
weight. To extend the classical verification, or run only part of the
table calculation:

```sh
sage -python verify_section5.py --max-rank 5 --classical-only
sage -python old/compute_section5.py --tables classical
sage -python old/compute_section5.py --tables exceptional
```

Higher ranks require larger exterior powers and permutation sums. All
command-line options are listed by `--help`.

For reading the code, start with `section5_formulas.py`: each expression
is annotated with its equation number. `section5_representations.py`
implements the defining operator sums and prescribed representations;
`section5_tables.py` contains the expected values. In these internal
classical routines, `kind` denotes **G**, as in Sections 5.1.4–5.1.7,
so types B and C are exchanged relative to the calculator's `--type`.
