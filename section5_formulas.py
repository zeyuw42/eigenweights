"""Literal transcription of Section 5 of Wang--Wei, main.pdf.

The argument ``kind`` is the type of G, as in Sections 5.1.4--5.1.7.
Thus kind B describes the dual group G-check of type C, and vice versa.
All arithmetic, permutations, coefficients and eigenvalues use SageMath.
"""

from dataclasses import dataclass
from functools import cache
from sage.all import QQ, ZZ, SymmetricGroup, binomial, factorial, multinomial, prod, matrix


@dataclass(frozen=True)
class Invariants:
    d: object
    degree: object
    s: tuple
    t: tuple
    kappa: object  # rows: e_i; columns: f_j, including the block (5.36)

    def eigenweights(self):
        """Corollary 4.3 and (5.36); retain degree order, sort repeated blocks."""
        n = len(self.s)
        answer = list(self.kappa.diagonal())
        for i in range(n):
            for j in range(i + 1, n):
                if self.kappa[i, j] or self.kappa[j, i]:
                    block = self.kappa.matrix_from_rows_and_columns([i, j], [i, j])
                    answer[i], answer[j] = sorted(block.eigenvalues())
        return tuple(answer)


def coefficient(total, entries):
    """Multinomial convention immediately after (5.14), including a=b.

    The top argument must equal the sum of the bottom arguments.
    """
    if any(x < 0 for x in entries):
        return ZZ.zero()
    if sum(entries) != total:
        raise ValueError(f"multinomial top {total} != sum({entries}) = {sum(entries)}")
    return multinomial(entries)


def signed_sum(k, total, entries):
    """The permutation sums in (5.12)--(5.14), (5.44), (5.51), (5.53), (5.56)."""
    return sum(sigma.sign() * coefficient(total, entries(sigma)) for sigma in SymmetricGroup(k))


@cache
def type_a(n, k):
    """Proposition 5.4: equations (5.10)--(5.14), exactly as printed."""
    kp = n + 1 - k
    d = ZZ(k * kp)  # (5.10)
    degree = factorial(d) / prod(
        i + h - 1 for i in range(1, k + 1) for h in range(1, kp + 1)
    )  # (5.11)
    s, t, eps = [], [], []
    for j in range(1, n + 1):

        def entries(sigma, a=None, b=None):
            bottom = [kp + sigma(i) - i for i in range(1, k + 1)]
            if a is not None:
                bottom[a - 1] -= j
            if b is not None:
                bottom[b - 1] += j + 1
            return bottom  # also implements the a=b convention after (5.14)

        s.append(
            sum(signed_sum(k, d - j, lambda p: entries(p, a=a)) for a in range(1, k + 1))
        )  # (5.12)
        t.append(
            sum(signed_sum(k, d + j + 1, lambda p: entries(p, b=b)) for b in range(1, k + 1))
        )  # (5.13)
        eps.append(
            sum(
                signed_sum(k, d + 1, lambda p: entries(p, a=a, b=b))
                for a in range(1, k + 1)
                for b in range(1, k + 1)
            )
        )  # (5.14)
    return Invariants(d, degree, tuple(s), tuple(t), matrix.diagonal(QQ, eps))


@cache
def classical(kind, n, k):
    """Propositions 5.4, 5.6, 5.7, 5.9; kind is the type of G, not G-check."""
    if kind == "A":
        return type_a(n, k)
    if kind == "B":
        a = type_a(2 * n - 1, k)
        return Invariants(
            ZZ(k * (2 * n - k)),
            2**k * a.degree,  # (5.17), (5.18)
            tuple(2**k * a.s[2 * j - 2] for j in range(1, n + 1)),  # (5.19)
            tuple(2 ** (k + 1) * a.t[2 * j - 2] for j in range(1, n + 1)),  # (5.20)
            matrix.diagonal(
                QQ, [2 ** (k + 1) * a.kappa[2 * j - 2, 2 * j - 2] for j in range(1, n + 1)]
            ),
        )  # (5.21)
    if kind == "C" and k < n:
        a = type_a(2 * n, k)
        return Invariants(
            ZZ(k * (2 * n + 1 - k)),
            2 ** (k * (2 * n - k)) * a.degree,  # (5.24), (5.25)
            tuple(2 ** (k * (2 * n - k)) * a.s[2 * j - 2] for j in range(1, n + 1)),  # (5.26)
            tuple(2 ** (k * (2 * n - k) + 2) * a.t[2 * j - 2] for j in range(1, n + 1)),  # (5.27)
            matrix.diagonal(
                QQ,
                [
                    2 ** (k * (2 * n - k) + 2) * a.kappa[2 * j - 2, 2 * j - 2]
                    for j in range(1, n + 1)
                ],
            ),
        )  # (5.28)
    if kind == "C":
        a = type_a(2 * n, n)
        d = ZZ(n * (n + 1) // 2)  # (5.29)
        degree = QQ(2 ** (n * n) * a.degree / binomial(2 * d, d)).sqrt()  # (5.30), positive root
        s = tuple(
            (
                QQ(2 ** (n * n - 1) * a.s[2 * j - 2]) / (binomial(2 * d - 2 * j + 1, d) * degree)
                if 2 * d - 2 * j + 1 >= d
                else QQ.zero()
            )
            for j in range(1, n + 1)
        )  # (5.31)
        t = tuple(
            QQ(2 ** (n * n + 1) * a.t[2 * j - 2]) / (binomial(2 * d + 2 * j, d) * degree)
            for j in range(1, n + 1)
        )  # (5.32)
        eps = [
            (
                2 ** (n * n + 1) * a.kappa[2 * j - 2, 2 * j - 2]
                - binomial(2 * d + 1, d - 2 * j + 1) * s[j - 1] * t[j - 1]
            )
            / (binomial(2 * d + 1, d) * degree)
            for j in range(1, n + 1)
        ]  # (5.33)
        return Invariants(d, degree, s, t, matrix.diagonal(QQ, eps))
    if kind != "D":
        raise ValueError(kind)
    if k <= n - 2:
        a = type_a(2 * n - 2, k)
        d = ZZ(k * (2 * n - 1 - k))  # (5.37)
        degree = 2**k * a.degree  # (5.38)
        s = tuple(2**k * a.s[2 * j - 2] for j in range(1, n)) + (ZZ.zero(),)  # (5.39), (5.42)
        t = tuple(2 ** (k + 1) * a.t[2 * j - 2] for j in range(1, n)) + (
            ZZ.zero(),
        )  # (5.40), (5.43)
        last = 2**k * signed_sum(
            k, d + 1, lambda p: [2 * n - k - 1 + p(i) - i + (i == 1) for i in range(1, k + 1)]
        )  # (5.44)
        eps = [2 ** (k + 1) * a.kappa[2 * j - 2, 2 * j - 2] for j in range(1, n)] + [last]  # (5.41)
        return Invariants(d, degree, s, t, matrix.diagonal(QQ, eps))
    a = type_a(2 * n - 2, n - 1)
    d = ZZ(n * (n - 1) // 2)  # (5.45)
    degree = QQ(2 ** (n - 1) * a.degree / binomial(2 * d, d)).sqrt()  # (5.46)
    s = [
        (
            QQ(2 ** (n - 2) * a.s[2 * j - 2]) / (binomial(2 * d - 2 * j + 1, d) * degree)
            if 2 * d - 2 * j + 1 >= d
            else QQ.zero()
        )
        for j in range(1, n)
    ]  # (5.47)
    t = [
        QQ(2 ** (n - 1) * a.t[2 * j - 2]) / (binomial(2 * d + 2 * j, d) * degree)
        for j in range(1, n)
    ]  # (5.48)
    diagonal = [
        (
            2 ** (n - 1) * a.kappa[2 * j - 2, 2 * j - 2]
            - binomial(2 * d + 1, d - 2 * j + 1) * s[j - 1] * t[j - 1]
        )
        / (binomial(2 * d + 1, d) * degree)
        for j in range(1, n)
    ]  # (5.49)
    sign = 1 if k == n - 1 else -1
    s.append(
        sign
        * (-1) ** (n - 1)
        * 2 ** (n - 2)
        * type_a(2 * n - 3, n - 1).degree
        / (binomial(2 * d - n + 1, d) * degree)
    )  # (5.50)
    t.append(
        sign
        * 2 ** (n - 2)
        * signed_sum(n, n * n, lambda p: [n - 1 + p(i) - i for i in range(1, n)] + [n - 1 + p(n)])
        / (binomial(2 * d + n, d) * degree)
    )  # (5.51)
    eps_sum, mixed_1, mixed_2 = type_d_auxiliary(n)  # (5.53), (5.56), (5.57)
    diagonal.append(
        (eps_sum / 2 + binomial(2 * d + 1, d - n + 1) * s[-1] * t[-1])
        / (binomial(2 * d + 1, d) * degree)
    )  # (5.52)
    K = matrix.diagonal(QQ, diagonal)
    if n % 2 == 0:
        K[n // 2 - 1, n - 1] = (
            sign * mixed_1 / 2 - binomial(2 * d + 1, d + 1 - n) * t[-1] * s[n // 2 - 1]
        ) / (
            binomial(2 * d + 1, d) * degree
        )  # (5.54)
        K[n - 1, n // 2 - 1] = (
            sign * mixed_2 / 2 - binomial(2 * d + 1, d + 1 - n) * t[n // 2 - 1] * s[-1]
        ) / (
            binomial(2 * d + 1, d) * degree
        )  # (5.55)
    return Invariants(d, degree, tuple(s), tuple(t), K)


def add_weights(a, b, degrees):
    """Proposition 5.1, (5.3)--(5.6), evaluated component by component.

    ``degrees`` contains the invariant-polynomial degrees d_j, not d_lambda.
    The formula (5.3) also applies to every entry of a repeated-degree block.
    """
    da, db = a.d, b.d
    D = da + db
    degree = binomial(D, da) * a.degree * b.degree  # (5.6)
    s = tuple(
        binomial(D - dj + 1, da - dj + 1) * a.s[j] * b.degree
        + binomial(D - dj + 1, db - dj + 1) * b.s[j] * a.degree
        for j, dj in enumerate(degrees)
    )  # (5.4)
    # Literal (5.5): the top is D+d_j; this is checked against (5.2).
    t = tuple(
        binomial(D + dj, da + dj) * a.t[j] * b.degree
        + binomial(D + dj, db + dj) * b.t[j] * a.degree
        for j, dj in enumerate(degrees)
    )  # (5.5)
    K = matrix(QQ, len(degrees))
    for i, di in enumerate(degrees):
        for j, dj in enumerate(degrees):
            if di != dj:
                continue
            K[i, j] = (
                binomial(D + 1, da + 1) * a.kappa[i, j] * b.degree
                + binomial(D + 1, db + 1) * b.kappa[i, j] * a.degree
                + binomial(D + 1, da - dj + 1) * b.t[j] * a.s[i]
                + binomial(D + 1, db - dj + 1) * a.t[j] * b.s[i]
            )  # (5.3)
    return Invariants(D, degree, s, t, K)


def first_eigenweight(d, degree, degrees, h_dual, lacing_G):
    """Proposition 5.2, (5.7); h_dual belongs to G-check, lacing to G."""
    return QQ(2 * h_dual * lacing_G * d * (d + 1) * (d + 2) * degree) / sum(
        (2 * dj - 2) * (2 * dj - 1) * 2 * dj for dj in degrees
    )


def add_three_b(a, b, c, ab, ac, bc):
    """Proposition 5.12, (5.58). Inputs are (d_lambda,deg_lambda,b_lambda)."""
    D = a[0] + b[0] + c[0]
    return (
        binomial(D + 1, a[0]) * a[1] * bc[2]
        + binomial(D + 1, b[0]) * b[1] * ac[2]
        + binomial(D + 1, c[0]) * c[1] * ab[2]
        - binomial(D + 1, a[0] + 1) * a[2] * bc[1]
        - binomial(D + 1, b[0] + 1) * b[2] * ac[1]
        - binomial(D + 1, c[0] + 1) * c[2] * ab[1]
    )


def minuscule_b(norm, d, degree):
    """Proposition 5.13, the displayed formula in Section 5.2.2."""
    return QQ(norm) * (d + 1) * degree / 2


def type_d_auxiliary(n):
    """The three auxiliary expressions (5.53), (5.56), (5.57)."""
    d = ZZ(n * (n - 1) // 2)  # (5.45)
    eps_sum = 2 ** (n - 1) * signed_sum(
        n - 1, 2 * d + 1, lambda p: [n + p(i) - i + (i == 1) for i in range(1, n)]
    )  # (5.53)
    mixed_1 = 2 ** (n - 1) * sum(
        signed_sum(
            n,
            2 * d + 1,
            lambda p: [n - 1 + p(i) - i - (n - 1 if i == a else 0) for i in range(1, n)]
            + [n - 1 + p(n)],
        )
        for a in range(1, n)
    )  # (5.56)
    mixed_2 = -(2**n) * type_a(2 * n - 3, n - 1).t[n - 2]  # (5.57)
    return eps_sum, mixed_1, mixed_2
