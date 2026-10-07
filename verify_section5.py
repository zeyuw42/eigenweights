#!/usr/bin/env -S sage -python
"""AI-generated verifier for Wang--Wei, Section 5, using exact Sage arithmetic.

Run: sage -python verify_section5.py --report verification_report.txt
Use --help for the reproducibility and finite-verification options.
All expected table entries are isolated in section5_tables.py.
"""

import argparse
import hashlib
import itertools
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from sage.all import QQ, CartanType, RootSystem, LieAlgebra, matrix, vector, identity_matrix
from sage.version import version as sage_version
from section5_formulas import (
    classical,
    first_eigenweight,
    add_weights,
    add_three_b,
    minuscule_b,
    type_d_auxiliary,
)
from section5_representations import (
    classical_standard,
    classical_exterior,
    classical_exterior_H2,
    classical_spin,
    exceptional_setup,
    exceptional_module,
    tensor_modules,
    exterior_action,
)
from section5_tables import (
    CLASSICAL_TABLES,
    EXCEPTIONAL_DEGREES,
    EXCEPTIONAL_EIGENWEIGHTS,
    EXCEPTIONAL_B,
)


class Report:
    """Never hide a discrepancy; continue the independent checks and exit nonzero."""

    def __init__(self, verbose=False):
        self.lines = []
        self.passed = 0
        self.failed = 0
        self.verbose = verbose

    def note(self, text):
        self.lines.append(text)
        print(text, flush=True)

    def check(self, reference, actual, expected):
        ok = actual == expected
        self.passed += int(ok)
        self.failed += int(not ok)
        text = (
            f"{'PASS' if ok else 'FAIL'} | {reference} | computed: {actual} | expected: {expected}"
        )
        self.lines.append(text)
        if self.verbose or not ok:
            print(text, flush=True)
        return ok


def polynomial_degrees(kind, n):
    """The d_j specified in Sections 5.1.4--5.1.7."""
    if kind == "A":
        return tuple(range(2, n + 2))
    return (
        tuple(2 * j for j in range(1, n + 1))
        if kind != "D"
        else tuple(2 * j for j in range(1, n)) + (n,)
    )


def canonical_spectrum(values, degrees):
    """(5.36): eigenvalues with the same d_j are unordered within that block."""
    return tuple(
        (d, tuple(sorted(values[i] for i, di in enumerate(degrees) if di == d)))
        for d in sorted(set(degrees))
    )


def check_invariants(report, kind, n, k, computed, printed):
    """Compare the literal Section 5 formulas with (5.1), (5.2), (4.5)."""
    name = f"G={kind}{n}, varpi_{k}"
    if kind == "A":
        refs = ("5.10", "5.11", "5.12", "5.13", "5.14")
    elif kind == "B":
        refs = ("5.17", "5.18", "5.19", "5.20", "5.21")
    elif kind == "C":
        refs = (
            ("5.24", "5.25", "5.26", "5.27", "5.28")
            if k < n
            else ("5.29", "5.30", "5.31", "5.32", "5.33")
        )
    else:
        refs = (
            ("5.37", "5.38", "5.39", "5.40", "5.41")
            if k <= n - 2
            else ("5.45", "5.46", "5.47", "5.48", "5.49")
        )
    report.check(f"({refs[0]}) | {name}", computed.d, printed.d)
    report.check(f"({refs[1]}) | {name}", computed.degree, printed.degree)
    bound = n - 1 if kind == "D" else n
    report.check(f"({refs[2]}) vs (5.1) | {name}", computed.s[:bound], printed.s[:bound])
    report.check(f"({refs[3]}) vs (5.2) | {name}", computed.t[:bound], printed.t[:bound])
    report.check(
        f"({refs[4]}) vs (4.5) | {name}",
        tuple(computed.kappa.diagonal()[:bound]),
        tuple(printed.kappa.diagonal()[:bound]),
    )
    if kind == "D":
        lastrefs = ("5.42", "5.43", "5.44") if k <= n - 2 else ("5.50", "5.51", "5.52")
        report.check(f"({lastrefs[0]}) vs (5.1), e_n | {name}", computed.s[-1], printed.s[-1])
        report.check(f"({lastrefs[1]}) vs (5.2), f_n | {name}", computed.t[-1], printed.t[-1])
        report.check(
            f"({lastrefs[2]}) vs (4.5), e_n,f_n | {name}",
            computed.kappa[-1, -1],
            printed.kappa[-1, -1],
        )
        if n % 2 == 0 and k >= n - 1:
            report.check(
                f"(5.54) vs (4.5), f_n | {name}",
                computed.kappa[n // 2 - 1, n - 1],
                printed.kappa[n // 2 - 1, n - 1],
            )
            report.check(
                f"(5.55) vs (4.5), e_n | {name}",
                computed.kappa[n - 1, n // 2 - 1],
                printed.kappa[n - 1, n // 2 - 1],
            )


def minuscule_norm(ct, k):
    """Use (4.4) on the Cartan and Sage's matrix inverse to get the dual form.

    This checks the kappa_min(lambda,lambda) column of the unnumbered table
    in Section 5.2.2.
    """
    L = LieAlgebra(QQ, cartan_type=ct)
    h = [L.e(i).bracket(L.f(i)) for i in range(1, ct.rank() + 1)]
    sym = ct.symmetrizer()
    lacing = max(sym.values()) / min(sym.values())
    Gram = matrix(
        QQ, [[L.killing_form(x, y) / (2 * ct.dual_coxeter_number() * lacing) for x in h] for y in h]
    )  # (4.4)
    coordinates = vector(QQ, ct.rank(), {k - 1: 1})
    return coordinates.dot_product(Gram.inverse() * coordinates)


def check_classical(report, max_rank):
    report.note(
        "CLASSICAL FORMULAS: direct (5.1), (5.2), (4.5) against Propositions 5.4, 5.6, 5.7, 5.9."
    )
    computed = {}
    for kind in ["A", "B", "C", "D"]:
        minimum = {"A": 1, "B": 2, "C": 2, "D": 3}[kind]
        for n in range(minimum, max_rank + 1):
            report.note(
                f'Computing G={kind}{n} (G-check type { {"A":"A","B":"C","C":"B","D":"D"}[kind] }).'
            )
            degrees = polynomial_degrees(kind, n)
            if kind == "D":
                E, Y, X, _, h = classical_standard(kind, n)
                H = matrix.diagonal(QQ, h)
                # Section 5.1.7 specifies these degrees and centralizer spaces.
                report.check(
                    f"Section 5.1.7: [2rho,e_n]=2(n-1)e_n, D{n}",
                    H * Y[-1] - Y[-1] * H == 2 * (n - 1) * Y[-1],
                    True,
                )
                report.check(f"Section 5.1.7: [e,e_n]=0, D{n}", E * Y[-1] - Y[-1] * E == 0, True)
                report.check(
                    f"Section 5.1.7: [2rho,f_n]=-2(n-1)f_n, D{n}",
                    H * X[-1] - X[-1] * H == -2 * (n - 1) * X[-1],
                    True,
                )
                report.check(
                    f"Section 5.1.7: [f,f_n]=0, D{n}", X[0] * X[-1] - X[-1] * X[0] == 0, True
                )
            for k in range(1, n + 1):
                printed = classical(kind, n, k)
                spin = (kind == "C" and k == n) or (kind == "D" and k >= n - 1)
                module, actual = (
                    classical_spin(kind, n, k) if spin else classical_exterior(kind, n, k)
                )
                computed[kind, n, k] = (module, actual)
                check_invariants(report, kind, n, k, actual, printed)
                dual_type = CartanType([{"A": "A", "B": "C", "C": "B", "D": "D"}[kind], n])
                sym = dual_type.symmetrizer()
                lacing = max(sym.values()) / min(sym.values())
                # The simple j=1 formula is checked independently, never used to generate a table.
                report.check(
                    f"(5.7) vs (4.5) | G={kind}{n},varpi_{k}",
                    actual.kappa[0, 0],
                    first_eigenweight(
                        actual.d, actual.degree, degrees, dual_type.dual_coxeter_number(), lacing
                    ),
                )
                if kind in ["A", "B"] and k == 1:
                    report.check(
                        f"Proposition 5.3 in Section 5.1.3 | G={kind}{n}",
                        module.E.right_kernel().dimension(),
                        1,
                    )
                    report.check(
                        f"Proposition 5.3 in Section 5.1.3 | G={kind}{n}",
                        actual.eigenweights(),
                        (actual.kappa[0, 0],) * n,
                    )
                # All rows of the minuscule table in Section 5.2.2, at the chosen ranks.
                minuscule = (
                    (kind == "A")
                    or (kind == "B" and k == 1)
                    or (kind == "C" and k == n)
                    or (kind == "D" and (k == 1 or k >= n - 1))
                )
                if minuscule:
                    bmodule, binvariants = module, actual
                    if not spin:
                        bmodule.H2 = classical_exterior_H2(kind, n, k)  # Theorem 4.4
                    norm_expected = (
                        QQ(k * (n + 1 - k)) / (n + 1)
                        if kind == "A"
                        else (
                            QQ.one()
                            if kind == "B" or (kind == "D" and k == 1)
                            else (QQ(n) / 2 if kind == "C" else QQ(n) / 4)
                        )
                    )  # unnumbered minuscule table
                    sage_k = k if kind != "D" or not spin else 2 * n - 1 - k
                    report.check(
                        f"Minuscule table, Section 5.2.2, norm | G={kind}{n},varpi_{k}",
                        minuscule_norm(dual_type, sage_k),
                        norm_expected,
                    )
                    report.check(
                        f"Proposition 5.13 vs Theorem 4.4 | G={kind}{n},varpi_{k}",
                        bmodule.b(),
                        minuscule_b(norm_expected, binvariants.d, binvariants.degree),
                    )
            if kind == "C":
                # The spin-square embedding in Section 5.1.6 independently checks
                # (5.30)--(5.33) via Proposition 5.1, (5.3)--(5.6).
                _, doubled = classical_exterior(kind, n, n)
                spin = classical(kind, n, n)
                predicted = add_weights(spin, spin, degrees)
                report.check(
                    f"(5.30)--(5.33) and spin-square embedding, G=C{n}: d,deg",
                    (predicted.d, predicted.degree),
                    (doubled.d, doubled.degree),
                )
                report.check(
                    f"(5.31), (5.4), spin-square embedding, G=C{n}", predicted.s, doubled.s
                )
                report.check(
                    f"(5.32), (5.5), spin-square embedding, G=C{n}", predicted.t, doubled.t
                )
                report.check(
                    f"(5.33), (5.3), spin-square embedding, G=C{n}", predicted.kappa, doubled.kappa
                )
            if kind == "D":
                _, sumspin = classical_exterior(kind, n, n - 1)
                doubles = []
                for branch, k in enumerate([n - 1, n]):
                    _, double = classical_exterior(kind, n, n, spin_branch=branch)
                    doubles.append(double)
                    spin = classical(kind, n, k)
                    predicted = add_weights(spin, spin, degrees)  # (5.3)--(5.6)
                    report.check(
                        f"Section 5.1.7, doubled half-spin embedding, 2varpi_{k}, D{n}",
                        double,
                        predicted,
                    )
                double = doubles[0]
                eps_sum, mixed_1, mixed_2 = type_d_auxiliary(n)
                report.check(f"(5.53) vs (4.5), G=D{n}", sumspin.kappa[-1, -1], eps_sum)
                if n % 2 == 0:
                    report.check(
                        f"(5.56) vs (4.5), f_n, G=D{n}", double.kappa[n // 2 - 1, n - 1], mixed_1
                    )
                    report.check(
                        f"(5.57) vs (4.5), e_n, G=D{n}", double.kappa[n - 1, n // 2 - 1], mixed_2
                    )

    report.note(
        "TABLES 1--4: compare both the literal formulas and independent representation calculations."
    )
    for table, rows in CLASSICAL_TABLES.items():
        for dual_kind, n, k, degree, d, eps in rows:
            if n > max_rank:
                continue
            kind = {"A": "A", "B": "C", "C": "B", "D": "D"}[dual_kind]
            printed = classical(kind, n, k)
            expected = (d, degree, canonical_spectrum(eps, polynomial_degrees(kind, n)))
            label = f"Table {table}, G-check={dual_kind}{n},varpi_{k}"
            report.check(
                label + " | literal Section 5 formulas",
                (
                    printed.d,
                    printed.degree,
                    canonical_spectrum(printed.eigenweights(), polynomial_degrees(kind, n)),
                ),
                expected,
            )
            _, actual = computed[kind, n, k]
            degrees = polynomial_degrees(kind, n)
            report.check(
                label + " | original (4.5), graded centralizers",
                (actual.d, actual.degree, canonical_spectrum(actual.eigenweights(), degrees)),
                expected,
            )
            report.note(f"Checked {label}: deg={degree}, d={d}, eigenweights={eps}.")
    return computed


def combine_representation(a, b):
    """Tensor action of the SAME Lie algebra; H_2 is recalculated from its root sum."""
    ma, Ya, Xa, Ha = a
    mb, Yb, Xb, Hb = b
    Ia = identity_matrix(QQ, ma.E.nrows(), sparse=True)
    Ib = identity_matrix(QQ, mb.E.nrows(), sparse=True)

    def combine(A, B):
        return A.tensor_product(Ib) + Ia.tensor_product(B)

    m = tensor_modules(ma, mb)
    Y = [combine(A, B) for A, B in zip(Ya, Yb)]
    X = [combine(A, B) for A, B in zip(Xa, Xb)]
    H = [combine(A, B) for A, B in zip(Ha, Hb)]
    # Theorem 4.4 for G=A2: h_G^vee=3; keep the printed root sum.
    m.H2 = sum(A * A for A in H) / (4 * 3)
    return m, Y, X, H


def check_reductions(report):
    """Propositions 5.1 and 5.12 on tensor realizations of Cartan components."""
    report.note("REDUCTION IDENTITIES: (5.3)--(5.6) and (5.58), with direct tensor actions for A2.")
    E, Y, X, degrees, h = classical_standard("A", 2)
    fundamental = {}
    simple_h = [matrix.diagonal(QQ, [1, -1, 0]), matrix.diagonal(QQ, [0, 1, -1])]
    for k in [1, 2]:
        m, _ = classical_exterior("A", 2, k)
        H = []
        for root in RootSystem(["A", 2]).coroot_lattice().roots():
            H.append(
                exterior_action(
                    sum(c * simple_h[i - 1] for i, c in root.monomial_coefficients().items()), k
                )
            )
        m.H2 = sum(A * A for A in H) / (4 * 3)  # Theorem 4.4
        fundamental[k] = (
            m,
            [exterior_action(A, k) for A in Y],
            [exterior_action(A, k) for A in X],
            H,
        )
    pairs = {}

    def inv(rep):
        return rep[0].invariants(rep[1], rep[2], degrees)

    def bdata(rep):
        m = rep[0]
        return m.d, m.degree, m.b()  # degree definition after (5.2), Theorem 4.4

    for i, j in itertools.combinations_with_replacement([1, 2], 2):
        actual = combine_representation(fundamental[i], fundamental[j])
        pairs[i, j] = actual
        expected = add_weights(inv(fundamental[i]), inv(fundamental[j]), degrees)
        result = inv(actual)
        label = f"lambda_1=varpi_{i},lambda_2=varpi_{j}, G=A2"
        report.check("(5.3) vs (4.5) | " + label, result.kappa, expected.kappa)
        report.check("(5.4) vs (5.1) | " + label, result.s, expected.s)
        report.check("(5.5) vs (5.2) | " + label, result.t, expected.t)
        report.check("(5.6) vs degree definition | " + label, result.degree, expected.degree)
    for i, j, k in itertools.combinations_with_replacement([1, 2], 3):
        a, b, c = fundamental[i], fundamental[j], fundamental[k]
        ab, ac, bc = (
            pairs[tuple(sorted([i, j]))],
            pairs[tuple(sorted([i, k]))],
            pairs[tuple(sorted([j, k]))],
        )
        triple = combine_representation(ab, c)
        expected = add_three_b(bdata(a), bdata(b), bdata(c), bdata(ab), bdata(ac), bdata(bc))
        report.check(f"(5.58) vs Theorem 4.4 | A2, weights ({i},{j},{k})", triple[0].b(), expected)
        # Include non-fundamental lambda_1 in Proposition 5.1.
        direct = inv(triple)
        predicted = add_weights(inv(ab), inv(c), degrees)
        report.check(f"(5.3)--(5.6), composite weight, A2 ({i},{j},{k})", direct, predicted)


def check_exceptional(report):
    """Tables 5--7 from the original Section 4 definitions only."""
    report.note("EXCEPTIONAL TABLES 5--7: (4.5), (4.4), Theorem 4.4, on Sage representations.")
    setups = {}
    for ct, name, degree, d in EXCEPTIONAL_DEGREES:
        if ct not in setups:
            report.note(f"Computing G-check={ct[0]}{ct[1]} and its graded centralizers.")
            setups[ct] = exceptional_setup(ct)
        setup = setups[ct]
        m, image, result = exceptional_module(setup, name)
        label = f"G-check={ct[0]}{ct[1]}, {name}"
        report.check("Table 5 | " + label, (result.degree, result.d), (degree, d))
        report.check(
            "Table 6 vs (4.5), Corollary 4.3 | " + label,
            result.eigenweights(),
            EXCEPTIONAL_EIGENWEIGHTS[ct, name],
        )
        report.check("Table 7 vs Theorem 4.4 | " + label, m.b(), EXCEPTIONAL_B[ct, name])
        degrees = setup[-1]
        sym = CartanType(ct).symmetrizer()
        lacing = max(sym.values()) / min(sym.values())
        report.check(
            "(5.7) vs (4.5) | " + label,
            result.kappa[0, 0],
            first_eigenweight(
                result.d, result.degree, degrees, CartanType(ct).dual_coxeter_number(), lacing
            ),
        )
        if ct == ("G", 2) and name == "quasi-minuscule":
            report.check("Proposition 5.3 | " + label, m.E.right_kernel().dimension(), 1)
            report.check(
                "Proposition 5.3 | " + label, result.eigenweights(), (result.kappa[0, 0],) * 2
            )
        if name == "minuscule":
            norm = QQ(4) / 3 if ct == ("E", 6) else QQ(3) / 2  # minuscule table, Section 5.2.2
            k = 1 if ct == ("E", 6) else 7
            report.check(
                "Minuscule table, Section 5.2.2, norm | " + label,
                minuscule_norm(CartanType(ct), k),
                norm,
            )
            report.check(
                "Proposition 5.13 vs Theorem 4.4 | " + label,
                m.b(),
                minuscule_b(norm, result.d, result.degree),
            )
        report.note(
            f"Checked Tables 5--7, {label}: deg={result.degree}, d={result.d}, eigenweights={result.eigenweights()}, b={m.b()}."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=4,
        help="largest classical rank checked (default: 4); finite checks are not a proof for all ranks",
    )
    parser.add_argument(
        "--classical-only", action="store_true", help="skip exceptional Tables 5--7"
    )
    parser.add_argument(
        "--paper", type=Path, help="record the SHA-256 of the manuscript being checked"
    )
    parser.add_argument("--report", type=Path, help="save all exact comparisons as plain text")
    parser.add_argument(
        "--verbose", action="store_true", help="also print every successful comparison"
    )
    args = parser.parse_args()
    if args.max_rank < 4:
        parser.error("--max-rank must be at least 4 to cover Tables 1--4")
    report = Report(args.verbose)
    report.note(f"Section 5 verification | SageMath {sage_version} | exact arithmetic over QQ")
    report.note(
        "Scope: all entries of Tables "
        + ("1--4" if args.classical_only else "1--7")
        + "; classical formula checks through rank "
        + str(args.max_rank)
        + "; A2 tensor checks of Propositions 5.1 and 5.12."
    )
    if args.paper:
        report.note("Manuscript SHA-256: " + hashlib.sha256(args.paper.read_bytes()).hexdigest())
    report.note(
        "Numbered references refer to main.pdf; no expected table value enters a calculation."
    )
    try:
        check_classical(report, args.max_rank)
        check_reductions(report)
        if not args.classical_only:
            check_exceptional(report)
    except Exception as error:
        report.failed += 1
        report.note(f"ERROR: {type(error).__name__}: {error}")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(report.lines) + "\n")
        raise
    report.note(f"RESULT: {report.passed} comparisons passed; {report.failed} comparisons failed.")
    if report.failed:
        report.note("Verification failed; see the comparisons marked FAIL.")
    else:
        report.note(
            "Every comparison in the stated finite scope passed. This is computational evidence, not a proof for all ranks or weights."
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(report.lines) + "\n")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
