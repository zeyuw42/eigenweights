"""Independent evaluations of (4.5), Theorem 4.4, (5.1) and (5.2).

Sage supplies Lie algebras, Chevalley brackets, representations, exterior
algebras, centralizers, rational linear algebra and tensor products.
The small adapter below extends Sage's supplied simple-generator matrices
by Lie brackets. It is needed because Sage 10.7's matrix Lie algebras do
not implement the basis coordinates required by LieAlgebra.morphism().
"""

from dataclasses import dataclass
from functools import cache
from sage.all import (
    QQ,
    ZZ,
    CartanType,
    CartanMatrix,
    LieAlgebra,
    RootSystem,
    PolynomialRing,
    ExteriorAlgebra,
    Subsets,
    matrix,
    vector,
    identity_matrix,
    zero_matrix,
)
from sage.data_structures.bitset import FrozenBitset
from section5_formulas import Invariants


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


def exceptional_setup(ct):
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


def classical_standard(kind, n):
    """The explicitly prescribed matrices e and bases in Sections 5.1.4--5.1.7."""
    N = {"A": n + 1, "B": 2 * n, "C": 2 * n + 1, "D": 2 * n}[kind]
    E = zero_matrix(QQ, N, sparse=True)
    for col in range(1, N):
        if kind == "A":
            E[col - 1, col] = 1
        elif kind == "B":
            E[col - 1, col] = 1 if col < n else (2 if col == n else -1)
        elif kind == "C":
            E[col - 1, col] = 2 if col <= n else (-1 if col == n + 1 else -2)
        elif col < n:
            E[col - 1, col] = 1
        elif col == n:
            E[col - 2, col] = 1
        elif col == n + 1:
            E[col - 2, col] = -1
            E[col - 1, col] = -1
        else:
            E[col - 1, col] = -1
    if kind == "D":
        h = vector(QQ, [2 * (n - i - 1) for i in range(n)] + [-2 * i for i in range(n)])
    else:
        h = vector(QQ, [N - 1 - 2 * i for i in range(N)])
    H = matrix.diagonal(QQ, h)
    # Solve [e,f]=2rho in degree -1, using Sage rational linear algebra.
    candidates = []
    for i in range(N):
        for j in range(N):
            if h[i] - h[j] == -2:
                candidates.append(matrix(QQ, N, N, {(i, j): 1}, sparse=True))
    if kind == "A":
        constraints = [(E * A - A * E).list() for A in candidates]
        target = H.list()
    else:
        Q = matrix(
            QQ,
            N,
            N,
            {
                (i, N - 1 - i): (
                    -1 if kind == "B" and i >= n else (2 if kind == "C" and i == n else 1)
                )
                for i in range(N)
            },
        )
        constraints = [
            (E * A - A * E).list() + (A.transpose() * Q + Q * A).list() for A in candidates
        ]
        target = H.list() + [0] * (N * N)
    system = matrix(QQ, constraints).transpose()
    F = sum(c * A for c, A in zip(system.solve_right(vector(QQ, target)), candidates))
    if E * F - F * E != H or H * F - F * H != -2 * F:
        raise ArithmeticError("The prescribed principal triple failed")
    bound = n - 1 if kind == "D" else n
    powers = [j if kind == "A" else 2 * j - 1 for j in range(1, bound + 1)]
    positive = [E**p for p in powers]  # (5.8), (5.15), (5.22), (5.34)
    trace_norm = {"A": 1, "B": 2, "C": 4, "D": 2}[kind]
    negative = [trace_norm * (F**p) / ((F**p) * Y).trace() for p, Y in zip(powers, positive)]
    # (5.9), (5.16), (5.23), (5.35); the D bases lie in Span(f^(2j-1)).
    if kind == "D":
        en = matrix(
            QQ, N, N, {(0, n - 1): 1, (0, n): -1, (n - 1, N - 1): 1, (n, N - 1): -1}, sparse=True
        )
        fn = matrix(
            QQ,
            N,
            N,
            {
                (n - 1, 0): QQ(1) / 2,
                (n, 0): QQ(-1) / 2,
                (N - 1, n - 1): QQ(1) / 2,
                (N - 1, n): QQ(-1) / 2,
            },
            sparse=True,
        )
        positive.append(en)
        negative.append(fn)  # the two displayed bases before (5.36)
        powers.append(n - 1)
    for i, Y in enumerate(positive):
        for j, X in enumerate(negative):
            if (X * Y).trace() != (trace_norm if i == j else 0):
                raise ArithmeticError("The prescribed dual bases failed")
    return E, positive, negative, tuple(p + 1 for p in powers), h


def classical_exterior(kind, n, k, spin_branch=0):
    """Direct (5.1), (5.2), (4.5) on the prescribed exterior representation.

    For spin weights this checks twice the weight, using the embeddings
    explicitly prescribed in Sections 5.1.6 and 5.1.7.
    """
    E, positive, negative, degrees, h = classical_standard(kind, n)
    N = E.nrows()
    indices = [tuple(sorted(I)) for I in Subsets(range(N), k)]
    low_support = tuple(range(N - k, N))
    high_support = tuple(range(k))
    if kind == "D" and k == n:
        # Section 5.1.7: branches 0,1 mean 2varpi_(n-1), 2varpi_n.
        # The primed lowest vectors exchange roles when n is odd.
        lows = [tuple(range(n, 2 * n)), (n - 1,) + tuple(range(n + 1, 2 * n))]
        highs = [tuple(range(n)), tuple(range(n - 1)) + (n,)]
        low_support = lows[(spin_branch + n % 2) % 2]
        high_support = highs[spin_branch]
    low = vector(QQ, len(indices), {indices.index(low_support): 1})
    high = vector(QQ, len(indices), {indices.index(high_support): 1})
    if kind == "B" or kind == "D":
        low *= (-1) ** (k * (2 * n - k - 1) // 2)
    elif kind == "C":
        low *= (-1) ** (k * (2 * n - k + 1) // 2)
    d = ZZ(-sum(h[i] for i in range(N - k, N)))
    module = Module(exterior_action(E, k), low, high, d)
    return module, module.invariants(
        [exterior_action(Y, k) for Y in positive],
        [exterior_action(X, k) for X in negative],
        degrees,
    )


def tensor_modules(a, b):
    """Sage tensor products of operators, for checking Propositions 5.1 and 5.12."""
    Ia = identity_matrix(QQ, a.E.nrows(), sparse=True)
    Ib = identity_matrix(QQ, b.E.nrows(), sparse=True)
    E = a.E.tensor_product(Ib) + Ia.tensor_product(b.E)
    low = vector(QQ, a.low.column().tensor_product(b.low.column()).column(0))
    high = vector(QQ, a.high.column().tensor_product(b.high.column()).column(0))
    return Module(E, low, high, a.d + b.d)


def classical_spin(kind, n, k):
    """Sage's minuscule crystal module for the spin weights in Sections 5.1.6--5.1.7.

    Only the prescribed matrices are transported between bases. Sage
    constructs and acts on the irreducible highest-weight representation.
    """
    ct = CartanType(["B" if kind == "C" else "D", n])
    L = LieAlgebra(QQ, cartan_type=ct)
    supplied = LieAlgebra(QQ, cartan_type=ct, representation="matrix")
    standard = matrix_representation(L, supplied.gens())
    E, positive, negative, degrees, h = classical_standard(kind, n)
    N = E.nrows()
    # Sage orders the negative standard basis in the reverse order to the paper.
    order = list(range(n)) + ([2 * n] if kind == "C" else []) + list(reversed(range(n, 2 * n)))
    P = matrix(
        QQ,
        N,
        N,
        {
            (old, new): (QQ(1) / 2 if kind == "C" and new > n else QQ.one())
            for new, old in enumerate(order)
        },
    )
    keys = list(L.basis().keys())
    coordinate_matrix = matrix(QQ, [standard(L.basis()[r]).list() for r in keys]).transpose()

    def abstract(A):
        old = P * A * P.inverse()
        coordinates = coordinate_matrix.solve_right(vector(QQ, old.list()))
        return L.from_vector(coordinates)

    # The paper names the all-positive half-spin weight varpi_(n-1).
    # In Sage's Bourbaki ordering it is varpi_n (Section 5.1.7).
    sage_k = k if kind == "C" else 2 * n - 1 - k
    image, low, high = minuscule_representation(L, sage_k)
    Em = image(abstract(E))
    Hm = image(abstract(matrix.diagonal(QQ, h)))
    d = ZZ(max(Hm.diagonal()))
    module = Module(Em, low, high, d)
    if module.degree < 0:
        module = Module(Em, -module.low, module.high, d)
    module.H2 = H2_operator(L, image, Em.nrows())  # Theorem 4.4
    return module, module.invariants(
        [image(abstract(Y)) for Y in positive], [image(abstract(X)) for X in negative], degrees
    )


def classical_exterior_H2(kind, n, k):
    """Theorem 4.4 on the exterior representation prescribed in Section 5.1."""
    ct = CartanType([{"A": "A", "B": "C", "C": "B", "D": "D"}[kind], n])
    N = {"A": n + 1, "B": 2 * n, "C": 2 * n + 1, "D": 2 * n}[kind]
    simple = []
    for i in range(n):
        diagonal = [QQ.zero()] * N
        if kind == "A":
            diagonal[i] = 1
            diagonal[i + 1] = -1
        elif i < n - 1:
            diagonal[i] = 1
            diagonal[i + 1] = -1
            diagonal[N - 1 - i] = -1
            diagonal[N - 2 - i] = 1
        elif kind == "D":
            diagonal[n - 2] = diagonal[n - 1] = 1
            diagonal[n] = diagonal[n + 1] = -1
        else:
            diagonal[n - 1] = 2 if kind == "C" else 1
            diagonal[N - n] = -diagonal[n - 1]
        simple.append(matrix.diagonal(QQ, diagonal))
    size = len(Subsets(range(N), k))
    result = zero_matrix(QQ, size, sparse=True)
    for coroot in RootSystem(ct).coroot_lattice().roots():
        H = sum(c * simple[i - 1] for i, c in coroot.monomial_coefficients().items())
        A = exterior_action(H, k)
        result += A * A
    hdual = RootSystem(ct).coroot_lattice().highest_root().associated_coroot().height() + 1
    return result / (4 * hdual)


@cache
def minuscule_representation(L, k):
    """Sage LSPaths and representation modules for the spin modules of Section 5.1.

    A minuscule crystal has one vector per weight and strings of length at
    most one. Its Chevalley generators therefore act by the crystal edges,
    with coefficient one in the primitive integral basis. Sage supplies all
    edges, weights and brackets, and tests the resulting Lie action on EVERY
    pair of Lie algebra basis elements and EVERY module basis vector.

    This avoids the very large intermediate Verma modules used by Sage's
    generic simple_module() constructor. It changes only the construction
    of V_lambda; (4.5), (5.1), (5.2) and Theorem 4.4 are evaluated unchanged.
    """
    from sage.all import crystals

    ct = L.cartan_type()
    weights = ct.root_system().weight_space()
    crystal = crystals.LSPaths(weights.fundamental_weights()[k])
    basis = list(crystal)
    if len({b.weight() for b in basis}) != len(basis) or any(
        b.epsilon(i) > 1 or b.phi(i) > 1 for b in basis for i in ct.index_set()
    ):
        raise ValueError("This constructor requires a minuscule weight")
    position = {b: j for j, b in enumerate(basis)}

    def edge(i, direction):
        return matrix(
            QQ,
            len(basis),
            len(basis),
            {
                (position[target], j): 1
                for j, b in enumerate(basis)
                if (target := getattr(b, direction)(i)) is not None
            },
            sparse=True,
        )

    raising = [edge(i, "e") for i in ct.index_set()]
    lowering = [edge(i, "f") for i in ct.index_set()]
    cartan = [
        matrix.diagonal(QQ, [b.weight().scalar(weights.simple_coroots()[i]) for b in basis])
        for i in ct.index_set()
    ]
    image = matrix_representation(L, raising + lowering + cartan)
    V = L.representation(image, index_set=range(len(basis)))
    V._test_representation(elements=list(V.basis()), max_runs=L.dimension() ** 2)
    low = vector(QQ, len(basis), {position[crystal.lowest_weight_vectors()[0]]: 1})
    high = vector(QQ, len(basis), {position[crystal.highest_weight_vector()]: 1})
    return image, low, high
