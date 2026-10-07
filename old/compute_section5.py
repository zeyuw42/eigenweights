#!/usr/bin/env -S sage -python
"""Human-written Sage computations by Zeyu Wang, collected and refactored by AI.

Compute the formulas and tables in Wang--Wei, Section 5.

Standalone: sage -python old/compute_section5.py
One weight: sage -python old/compute_section5.py --type D --rank 6 --weight 5

Classical types in the public API are the types of G-check, as in Tables
1--7. The internal classical_formula() uses the manuscript's type of G.
All numbered references below are to main.pdf. Table constants are used
only as expected answers. Computations are exact Sage rational arithmetic.

"""

import argparse
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

sys.dont_write_bytecode = True

from sage.all import (
    QQ,
    ZZ,
    SymmetricGroup,
    binomial,
    factorial,
    multinomial,
    prod,
    matrix,
    vector,
    CartanType,
    CartanMatrix,
    LieAlgebra,
    RootSystem,
    PolynomialRing,
    ExteriorAlgebra,
    Subsets,
    identity_matrix,
    zero_matrix,
)
from sage.data_structures.bitset import FrozenBitset
from sage.version import version as sage_version


@dataclass(frozen=True)
class Invariants:
    d: object
    degree: object
    s: tuple
    t: tuple
    kappa: object  # rows: e_i; columns: f_j, including the block (5.36)


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
def classical_formula(kind, n, k):
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


def matrix_representation(L, generators):
    """Extend Sage's generator matrices using Sage's Chevalley brackets.

    This constructs dr_V, the representation used in (4.3), (4.5), (5.1),
    (5.2) and Theorem 4.4. No invariant is computed by this adapter.
    """
    roots = L.cartan_type().root_system().root_lattice()
    all_roots = set(roots.roots())
    images = {
        x.support()[0]: (y.matrix() if hasattr(y, "matrix") else y)
        for x, y in zip(L.gens(), generators)
    }

    @cache
    def on_basis(root):
        if root in images:
            return images[root]
        sign = 1 if sum(root.coefficients()) > 0 else -1
        for alpha in roots.simple_roots():
            alpha = sign * alpha
            beta = root - alpha
            if beta in all_roots:
                bracket = L.basis()[alpha].bracket(L.basis()[beta])
                scalar = bracket.monomial_coefficients().get(root, 0)
                if scalar:
                    A, B = on_basis(alpha), on_basis(beta)
                    return (A * B - B * A) / scalar
        raise ValueError(f"Cannot extend the supplied representation to {root}")

    def image(x):
        return sum(c * on_basis(root) for root, c in x.monomial_coefficients().items())

    return image


def exterior_action(A, k):
    """The coefficient of z in Sage's Lambda(I+z*A), on Lambda^k.

    Sections 5.1.4--5.1.7 use the differentiated exterior-power action.
    Sage's lift_morphism performs the wedge algebra and its signs.
    """
    R = PolynomialRing(QQ, "z")
    z = R.gen()
    exterior = ExteriorAlgebra(R, A.nrows(), "u")
    lift = exterior.lift_morphism(identity_matrix(R, A.nrows()) + z * A)
    indices = [FrozenBitset(I) for I in Subsets(range(A.nrows()), k)]
    columns = [lift(exterior.monomial(J)).monomial_coefficients() for J in indices]
    return matrix(
        QQ, [[column.get(I, R.zero())[1] for column in columns] for I in indices], sparse=True
    )


@dataclass
class Module:
    E: object
    low: object
    high: object  # column representing u^*; high.dot_product(v) is its evaluation
    d: object
    H2: object = None

    def __post_init__(self):
        # Repeated multiplication only evaluates powers in the printed sums.
        self.right = [self.low]
        self.left = [self.high]
        for _ in range(int(self.d)):
            self.right.append(self.E * self.right[-1])
            self.left.append(self.E.transpose() * self.left[-1])

    @property
    def degree(self):
        """The definition deg_lambda immediately following (5.2)."""
        return self.high.dot_product(self.right[self.d])

    def S(self, Y, dj):
        """Equation (5.1): <e^(d_lambda-d_j+1) Y u_-, u_+^*>."""
        power = self.d - dj + 1
        return QQ.zero() if power < 0 else self.left[power].dot_product(Y * self.low)

    def T(self, X, dj):
        """Equation (5.2), with zero terms outside the nilpotent range omitted."""
        power = self.d + dj - 1
        return sum(
            self.left[s].dot_product(X * self.right[power - s])
            for s in range(max(0, power - self.d), min(self.d, power) + 1)
        )

    def kappa(self, X, Y):
        """Equation (4.5), keeping the printed operator order e^s X Y e^(d-s)."""
        return sum(
            self.left[s].dot_product(X * (Y * self.right[self.d - s])) for s in range(self.d + 1)
        )

    def b(self):
        """Theorem 4.4: sum_s <e^s H_2 e^(d_lambda-s) u_-, u_+^*>."""
        if self.H2 is None:
            raise ValueError("H_2 has not been supplied")
        return sum(
            self.left[s].dot_product(self.H2 * self.right[self.d - s]) for s in range(self.d + 1)
        )

    def invariants(self, positive, negative, degrees):
        """(5.1), (5.2), (4.5); the supplied bases have kappa_min(f_j,e_i)=delta_ij."""
        return Invariants(
            self.d,
            self.degree,
            tuple(self.S(Y, dj) for Y, dj in zip(positive, degrees)),
            tuple(self.T(X, dj) for X, dj in zip(negative, degrees)),
            matrix(
                QQ,
                [
                    [self.kappa(X, Y) if di == dj else 0 for X, dj in zip(negative, degrees)]
                    for Y, di in zip(positive, degrees)
                ],
            ),
        )


@cache
def principal_setup(ct):
    """Section 4.2.1 and (4.4): principal triple and graded centralizers."""
    ct = CartanType(ct)
    L = LieAlgebra(QQ, cartan_type=ct)
    rank = ct.rank()
    sym = ct.symmetrizer()
    lacing = max(sym.values()) / min(sym.values())
    # Section 4.2.1: kappa_min(alpha_i,alpha_i)/2, roots (not coroots).
    a = vector(QQ, [sym[i] / min(sym.values()) for i in range(1, rank + 1)])
    c = CartanMatrix(ct).transpose().solve_right(vector(QQ, [2] * rank))
    e = sum(a[i - 1] * L.e(i) for i in range(1, rank + 1))
    f = sum(c[i - 1] / a[i - 1] * L.f(i) for i in range(1, rank + 1))
    h = e.bracket(f)
    if h.bracket(e) != 2 * e or h.bracket(f) != -2 * f:
        raise ArithmeticError("The principal sl_2 triple failed")

    def height(x):
        heights = {sum(root.coefficients()) for root in x.support()}
        if len(heights) != 1:
            raise ArithmeticError("A centralizer basis element is not homogeneous")
        return ZZ(heights.pop())

    positive = sorted(L.centralizer_basis(e), key=height)
    negative = sorted(L.centralizer_basis(f), key=lambda x: -height(x))
    degrees = tuple(height(x) + 1 for x in positive)
    # (4.4), using the Killing form supplied by Sage, not a trace substitute for (4.5).
    normalization = 2 * ct.dual_coxeter_number() * lacing
    pairing = matrix(
        QQ, [[L.killing_form(X, Y) / normalization for X in negative] for Y in positive]
    )
    # Sage solves the dual-basis change, including any repeated-degree block.
    inverse = pairing.inverse()
    negative = [sum(inverse[j, i] * negative[j] for j in range(rank)) for i in range(rank)]
    return L, e, f, h, positive, negative, degrees


def H2_operator(L, image, size):
    """H_2 in Theorem 4.4, literally the sum over roots of G.

    The roots of G are the coroots of G-check. h_G^vee belongs to the dual
    Cartan type. This distinction is essential in nonsimply-laced types.
    """
    ct = L.cartan_type()
    simple_h = [L.e(i).bracket(L.f(i)) for i in range(1, ct.rank() + 1)]
    result = zero_matrix(QQ, size, sparse=True)
    for coroot in RootSystem(ct).coroot_lattice().roots():
        H = sum(c * simple_h[i - 1] for i, c in coroot.monomial_coefficients().items())
        A = image(H)
        result += A * A
    return result / (
        4 * (RootSystem(ct).coroot_lattice().highest_root().associated_coroot().height() + 1)
    )


def exceptional_module(setup, name):
    """Sage's supplied minimal or adjoint representation, Tables 5--7."""
    L, e, f, h, positive, negative, degrees = setup
    ct = L.cartan_type()
    if name == "adjoint":
        image = lambda x: x.adjoint_matrix(sparse=True)
        keys = list(L.basis().keys())
        theta = ct.root_system().root_lattice().highest_root()
        low = vector(QQ, len(keys), {keys.index(-theta): 1})
        high = vector(QQ, len(keys), {keys.index(theta): 1})
        d = ZZ(2 * sum(theta.coefficients()))
    else:
        supplied = LieAlgebra(QQ, cartan_type=ct, representation="matrix")
        image = matrix_representation(L, supplied.gens())
        diagonal = image(h).diagonal()
        d = ZZ(max(diagonal))
        if diagonal.count(d) != 1 or diagonal.count(-d) != 1:
            raise ArithmeticError("The extreme principal-weight spaces are not lines")
        low = vector(QQ, len(diagonal), {diagonal.index(-d): 1})
        high = vector(QQ, len(diagonal), {diagonal.index(d): 1})
    E = image(e)
    module = Module(E, low, high, d, H2_operator(L, image, E.nrows()))
    # Section 4.2.1 fixes the sign by deg_lambda > 0; never scale to table data.
    if module.degree < 0:
        module = Module(E, -low, high, d, module.H2)
    return (
        module,
        image,
        module.invariants([image(Y) for Y in positive], [image(X) for X in negative], degrees),
    )


def polynomial_degrees(kind, rank):
    """The degree order in Sections 5.1.4--5.1.7, including (5.36)."""
    if kind == "A":
        return tuple(range(2, rank + 2))
    if kind == "D":
        return tuple(range(2, 2 * rank, 2)) + (rank,)
    return tuple(range(2, 2 * rank + 1, 2))


def prescribed_standard(kind, rank):
    """The primitive standard basis in Sections 5.1.5--5.1.7.

    The type is G-check. Sage's default type-B standard lattice scales
    one extreme vector differently, so use the manuscript's displayed e
    directly. Cartan diagonals need only Sage's basis-order permutation.
    """
    size = 2 * rank + 1 if kind == "B" else 2 * rank
    E = zero_matrix(QQ, size, sparse=True)
    for col in range(1, size):
        if kind == "C":  # Section 5.1.5 (type of G is B).
            E[col - 1, col] = 1 if col < rank else (2 if col == rank else -1)
        elif kind == "B":  # Section 5.1.6 (type of G is C).
            E[col - 1, col] = 2 if col <= rank else (-1 if col == rank + 1 else -2)
        elif col < rank:  # Section 5.1.7, displayed standard e.
            E[col - 1, col] = 1
        elif col == rank:
            E[col - 2, col] = 1
        elif col == rank + 1:
            E[col - 2, col] = -1
            E[col - 1, col] = -1
        else:
            E[col - 1, col] = -1
    standard = LieAlgebra(QQ, cartan_type=[kind, rank], representation="matrix")
    order = (
        list(range(rank))
        + ([2 * rank] if kind == "B" else [])
        + list(reversed(range(rank, 2 * rank)))
    )
    cartan = [
        matrix.diagonal(QQ, [standard.h(i).matrix()[j, j] for j in order])
        for i in range(1, rank + 1)
    ]
    return E, cartan


def classical_b(kind, rank, k, invariants):
    """Section 5.2: Proposition 5.13 for minuscule weights, otherwise Theorem 4.4.

    Sage supplies the exterior action and the coroots in the defining sum.
    """
    norm = None
    if kind == "A":
        norm = QQ(k * (rank + 1 - k)) / (rank + 1)  # Minuscule table, Section 5.2.2.
    elif kind == "B" and k == rank:
        norm = QQ(rank) / 2  # The same table, type-G C spin row.
    elif kind == "D" and k >= rank - 1:
        norm = QQ(rank) / 4  # The same table, type-G D half-spin row.
    if norm is not None:
        return norm * (invariants.d + 1) * invariants.degree / 2  # Proposition 5.13.

    ct = CartanType([kind, rank])
    E, cartan = prescribed_standard(kind, rank)
    coefficients = CartanMatrix(ct).transpose().solve_right(vector(QQ, [2] * rank))
    H = sum(coefficients[i - 1] * cartan[i - 1] for i in ct.index_set())
    # Section 4.2.1: e and h=2rho. Identify the primitive extreme wedge lines.
    subsets = list(Subsets(range(E.nrows()), k))
    weights = [sum(H[i, i] for i in subset) for subset in subsets]
    d = ZZ(max(weights))
    if weights.count(d) != 1 or weights.count(-d) != 1:
        raise ArithmeticError("The extreme principal-weight spaces are not lines")
    low = vector(QQ, len(subsets), {weights.index(-d): 1})
    high = vector(QQ, len(subsets), {weights.index(d): 1})
    simple_h = [exterior_action(H, k) for H in cartan]
    H2 = zero_matrix(QQ, len(subsets), sparse=True)
    coroots = RootSystem(ct).coroot_lattice()
    for root in coroots.roots():
        action = sum(c * simple_h[i - 1] for i, c in root.monomial_coefficients().items())
        H2 += action**2
    # Theorem 4.4: h_G^vee belongs to G, which is dual to the displayed type.
    h_dual_G = coroots.highest_root().associated_coroot().height() + 1
    H2 /= 4 * h_dual_G
    module = Module(exterior_action(E, k), low, high, d, H2)
    if module.degree < 0:
        module = Module(module.E, -low, high, d, H2)
    if (module.d, module.degree) != (invariants.d, invariants.degree):
        raise ArithmeticError("The exterior module disagrees with the Section 5 degree formula")
    return module.b()  # Theorem 4.4, original e^s H2 e^(d-s) operator sum.


@dataclass(frozen=True)
class Computed:
    kind: str
    rank: int
    representation: str
    degrees: tuple
    invariants: Invariants
    b: object

    def spectrum(self):
        """Corollary 4.3 and (5.36), with full equal-degree matrix blocks."""
        answer = {}
        for degree in sorted(set(self.degrees)):
            indices = [i for i, d in enumerate(self.degrees) if d == degree]
            block = self.invariants.kappa.matrix_from_rows_and_columns(indices, indices)
            answer[degree] = tuple(sorted(block.eigenvalues()))
        return answer

    def text(self):
        values = self.invariants
        # Sage supplies exact radical expressions for the quadratic D block.
        spectrum = {
            degree: tuple(
                value.radical_expression() if hasattr(value, "radical_expression") else value
                for value in eigenvalues
            )
            for degree, eigenvalues in self.spectrum().items()
        }
        lines = [
            f"{self.kind}{self.rank}, {self.representation}",
            f"  d_lambda = {values.d}; deg_lambda = {values.degree}; b_lambda = {self.b}",
            f"  s = {values.s}",
            f"  t = {values.t}",
            "  eigenweights by invariant degree: " + str(spectrum),
        ]
        if self.kind == "D" and self.rank % 2 == 0:
            indices = [i for i, d in enumerate(self.degrees) if d == self.rank]
            block = values.kappa.matrix_from_rows_and_columns(indices, indices)
            lines += [
                f"  degree-{self.rank} block (5.36):",
                str(block),
                f'  characteristic polynomial = {block.charpoly("x")}',
            ]
        return "\n".join(lines)


def compute(kind, rank, weight=None, representation="minimal"):
    """Classical fundamental weights or the exceptional Table 5 modules.

    Types denote G-check, as in Tables 1--7. Conversion to the type of G
    occurs only when calling the classical formulas.
    """
    ct = CartanType([kind, rank])
    if kind in "ABCD":
        minimum = {"A": 1, "B": 2, "C": 2, "D": 3}[kind]
        if rank < minimum or weight is None or not 1 <= weight <= rank:
            raise ValueError(f"Use rank >= {minimum} and a fundamental weight between 1 and rank")
        manuscript_kind = {"A": "A", "B": "C", "C": "B", "D": "D"}[kind]
        values = classical_formula(manuscript_kind, rank, weight)
        return Computed(
            kind,
            rank,
            f"fundamental weight {weight}",
            polynomial_degrees(kind, rank),
            values,
            classical_b(kind, rank, weight, values),
        )
    if weight is not None:
        raise ValueError("For exceptional types select --representation minimal or adjoint")
    names = {
        ("G", 2): "quasi-minuscule",
        ("F", 4): "quasi-minuscule",
        ("E", 6): "minuscule",
        ("E", 7): "minuscule",
    }
    if representation == "adjoint":
        name = "adjoint"
    elif (kind, rank) in names:
        name = names[kind, rank]
    else:
        raise ValueError("E8 has only an adjoint row in Tables 5--7; use --representation adjoint")
    setup = principal_setup(ct)
    module, image, values = exceptional_module(setup, name)
    return Computed(kind, rank, name, setup[-1], values, module.b())


# Expected values from Tables 1--7 follow. They enter ONLY comparisons.

CLASSICAL_TABLES = {
    1: [("A", 1, 1, 1, 1, (1,))],  # Table 1
    2: [  # Table 2
        ("A", 2, 1, 1, 2, (1, 1)),
        ("A", 2, 2, 1, 2, (1, 1)),
        ("B", 2, 1, 8, 4, (32, 32)),
        ("B", 2, 2, 2, 3, (4, 4)),
    ],
    3: [  # Table 3
        ("A", 3, 1, 1, 3, (1, 1, 1)),
        ("A", 3, 2, 2, 4, (4, 2, 4)),
        ("A", 3, 3, 1, 3, (1, 1, 1)),
        ("B", 3, 1, 32, 6, (128, 128, 128)),
        ("B", 3, 2, 10752, 10, (168960, 67584, 125952)),
        ("B", 3, 3, 16, 6, (64, 40, 64)),
        ("C", 3, 1, 2, 5, (4, 4, 4)),
        ("C", 3, 2, 56, 8, (384, 184, 328)),
        ("C", 3, 3, 336, 9, (3168, 1568, 2720)),
    ],
    4: [  # Table 4
        ("A", 4, 1, 1, 4, (1, 1, 1, 1)),
        ("A", 4, 2, 5, 6, (14, 6, 9, 13)),
        ("A", 4, 3, 5, 6, (14, 6, 9, 13)),
        ("A", 4, 4, 1, 4, (1, 1, 1, 1)),
        ("B", 4, 1, 128, 8, (512, 512, 512, 512)),
        ("B", 4, 2, 1757184, 14, (32800768, 11501568, 19169280, 23216128)),
        ("B", 4, 3, 2867724288, 18, (108973522944, 26417823744, 42420928512, 65536524288)),
        ("B", 4, 4, 768, 10, (5632, 2432, 3584, 5248)),
        ("C", 4, 1, 2, 7, (4, 4, 4, 4)),
        ("C", 4, 2, 528, 12, (4576, 1664, 3008, 3424)),
        ("C", 4, 3, 48048, 15, (777920, 232128, 349632, 555968)),
        ("C", 4, 4, 384384, 16, (7468032, 2376192, 3355392, 5392128)),
        ("D", 4, 1, 2, 6, (4, 4, 4, 2)),
        ("D", 4, 2, 168, 10, (1320, 528, 984, 528)),
        ("D", 4, 3, 2, 6, (4, 4, 4, 2)),
        ("D", 4, 4, 2, 6, (4, 4, 4, 2)),
    ],
}

# Table 5; rows (G-check Cartan type, name, degree, d).
EXCEPTIONAL_DEGREES = [
    (("G", 2), "quasi-minuscule", 18, 6),
    (("G", 2), "adjoint", 13608, 10),
    (("F", 4), "quasi-minuscule", 4992, 16),
    (("F", 4), "adjoint", 154791936, 22),
    (("E", 6), "minuscule", 78, 16),
    (("E", 6), "adjoint", 151164, 22),
    (("E", 7), "minuscule", 13110, 27),
    (("E", 7), "adjoint", 141430680, 34),
    (("E", 8), "adjoint", 126937516885200, 58),
]

# Table 6; degrees are ordered increasingly.
EXCEPTIONAL_EIGENWEIGHTS = {
    (("G", 2), "quasi-minuscule"): (108, 108),
    (("G", 2), "adjoint"): (320760, 239112),
    (("F", 4), "quasi-minuscule"): (52224, 27648, 32640, 47232),
    (("F", 4), "adjoint"): (4016652288, 1610956800, 2099613696, 2908827648),
    (("E", 6), "minuscule"): (408, 221, 216, 255, 299, 369),
    (("E", 6), "adjoint"): (1961256, 490314, 786600, 1025202, 967518, 1420326),
    (("E", 7), "minuscule"): (120060, 58320, 48024, 69920, 67176, 85376, 101744),
    (("E", 7), "adjoint"): (
        2530864800,
        643719960,
        1025744616,
        921331160,
        1301226984,
        1386811784,
        1807586936,
    ),
    (("E", 8), "adjoint"): (
        3503065990170600,
        1035140220518880,
        1116031452545520,
        1289228099378520,
        1602802318771080,
        1661879186158800,
        2045534982573600,
        2471069708566200,
    ),
}

# Table 7; these are checked solely by the sum in Theorem 4.4.
EXCEPTIONAL_B = {
    (("G", 2), "quasi-minuscule"): 108,
    (("G", 2), "adjoint"): 279936,
    (("F", 4), "quasi-minuscule"): 79872,
    (("F", 4), "adjoint"): 5318025216,
    (("E", 6), "minuscule"): 884,
    (("E", 6), "adjoint"): 3325608,
    (("E", 7), "minuscule"): 275310,
    (("E", 7), "adjoint"): 4808643120,
    (("E", 8), "adjoint"): 7362375979341600,
}


def canonical_spectrum(values, degrees):
    """Table entries are unordered within the equal-degree block (5.36)."""
    return {
        degree: tuple(sorted(values[i] for i, d in enumerate(degrees) if d == degree))
        for degree in sorted(set(degrees))
    }


def verify_tables(selection="all"):
    """Compute before comparing with the expected values in Tables 1--7."""
    lines = [f"SageMath {sage_version}: Section 5 table verification.", ""]
    passed = failed = 0

    def check(reference, actual, expected):
        nonlocal passed, failed
        ok = actual == expected
        passed += int(ok)
        failed += int(not ok)
        lines.append(
            f'{"PASS" if ok else "FAIL"} | {reference} | '
            f"computed: {actual} | expected: {expected}"
        )
        if not ok:
            print(lines[-1], flush=True)

    if selection in ["all", "classical"]:
        for number, rows in CLASSICAL_TABLES.items():
            for kind, rank, k, degree, d, eigenweights in rows:
                result = compute(kind, rank, weight=k)
                print(result.text(), flush=True)
                lines += [result.text()]
                label = f"Table {number}: {kind}{rank}, fundamental weight {k}"
                check(label + " d_lambda", result.invariants.d, d)
                check(label + " deg_lambda", result.invariants.degree, degree)
                check(
                    label + " eigenweights",
                    result.spectrum(),
                    canonical_spectrum(eigenweights, result.degrees),
                )
    if selection in ["all", "exceptional"]:
        for (kind, rank), name, degree, d in EXCEPTIONAL_DEGREES:
            result = compute(
                kind, rank, representation="adjoint" if name == "adjoint" else "minimal"
            )
            print(result.text(), flush=True)
            lines += [result.text()]
            label = f"{kind}{rank} {name}"
            check("Table 5: " + label + " d_lambda", result.invariants.d, d)
            check("Table 5: " + label + " deg_lambda", result.invariants.degree, degree)
            check(
                "Table 6: " + label + " eigenweights",
                result.spectrum(),
                canonical_spectrum(EXCEPTIONAL_EIGENWEIGHTS[(kind, rank), name], result.degrees),
            )
            check("Table 7: " + label + " b_lambda", result.b, EXCEPTIONAL_B[(kind, rank), name])
    lines += ["", f"{passed} passing comparisons; {failed} failing comparisons."]
    print(lines[-1], flush=True)
    return "\n".join(lines) + "\n", failed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--type", dest="kind", choices=list("ABCDEFG"), help="type of G-check, as in the tables"
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--weight", type=int, help="classical fundamental-weight index")
    parser.add_argument("--representation", choices=["minimal", "adjoint"], default="minimal")
    parser.add_argument(
        "--tables",
        choices=["all", "classical", "exceptional"],
        default="all",
        help="which tables to verify when --type is omitted",
    )
    parser.add_argument("--report", type=Path, help="save the calculation or verification report")
    args = parser.parse_args()
    if args.kind:
        if args.rank is None:
            parser.error("--rank is required with --type")
        try:
            text = compute(args.kind, args.rank, args.weight, args.representation).text() + "\n"
        except ValueError as error:
            parser.error(str(error))
        print(text, end="")
        failed = 0
    else:
        if args.rank is not None or args.weight is not None:
            parser.error("--rank and --weight require --type")
        text, failed = verify_tables(args.tables)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
