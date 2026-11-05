from django.db.models import FloatField, Func, Value
from .. import Vector, HalfVector, SparseVector
from typing import Any, ClassVar


class DistanceBase(Func):
    """Base class for pgvector distance functions that operate on vector columns.

    Accepts either a Django ORM expression (any object with a
    ``resolve_expression`` method) or a raw vector value as ``list[float]``,
    :class:`~pgvector.Vector`, :class:`~pgvector.HalfVector`, or
    :class:`~pgvector.SparseVector`.  Raw values are converted to their
    PostgreSQL text representation before being passed to the query.
    """

    output_field = FloatField()  # type: ignore

    def __init__(self, expression: Any, vector: Any, /, **extra: Any) -> None:
        """Initialize a distance expression between *expression* and *vector*.

        Args:
            expression: A Django ORM expression referencing a vector column.
            vector: The comparison vector.  May be a ``list[float]``,
                :class:`~pgvector.Vector`, :class:`~pgvector.HalfVector`,
                :class:`~pgvector.SparseVector`, or any Django expression
                that resolves to a compatible vector type.
            **extra: Additional keyword arguments forwarded to
                :class:`django.db.models.Func`.
        """
        if not hasattr(vector, 'resolve_expression'):
            if isinstance(vector, (Vector, HalfVector, SparseVector)):
                vector = Value(vector.to_text())
            elif vector is not None:
                vector = Value(Vector(vector).to_text())

            # prevent error with unhashable types
            self._constructor_args = ((expression, vector), extra)

        super().__init__(expression, vector, **extra)


class BitDistanceBase(Func):
    """Base class for pgvector distance functions that operate on bit columns.

    Unlike :class:`DistanceBase`, the comparison value is used as-is (a plain
    Python value that Django wraps in a :class:`~django.db.models.Value`).
    """

    output_field = FloatField()  # type: ignore

    def __init__(self, expression: Any, vector: Any, /, **extra: Any) -> None:
        """Initialize a bit-distance expression between *expression* and *vector*.

        Args:
            expression: A Django ORM expression referencing a bit column.
            vector: The comparison bit string or any Django expression that
                resolves to a compatible bit type.
            **extra: Additional keyword arguments forwarded to
                :class:`django.db.models.Func`.
        """
        if not hasattr(vector, 'resolve_expression'):
            vector = Value(vector)
        super().__init__(expression, vector, **extra)


class L2Distance(DistanceBase):
    """L2 (Euclidean) distance via the pgvector ``<->`` operator.

    Returns the straight-line distance between two vectors in Euclidean space.
    Equivalent to ``sqrt(sum((a_i - b_i)^2))``.  Lower values indicate
    greater similarity.

    Example::

        Item.objects.annotate(dist=L2Distance('embedding', [1, 2, 3])).order_by('dist')
    """

    function: ClassVar[str] = ''
    arg_joiner: ClassVar[str] = ' <-> '


class MaxInnerProduct(DistanceBase):
    """Negative inner product via the pgvector ``<#>`` operator.

    Returns the *negated* dot product of two vectors so that ORDER BY gives
    the highest-similarity results first (pgvector stores it negated to
    support ``ORDER BY ASC``).  Equivalent to ``-sum(a_i * b_i)``.

    Use this instead of :class:`CosineDistance` when vectors are already
    unit-normalised (cosine similarity and inner product are equivalent in
    that case, but inner product is faster).

    Example::

        Item.objects.annotate(score=MaxInnerProduct('embedding', [1, 2, 3])).order_by('score')
    """

    function: ClassVar[str] = ''
    arg_joiner: ClassVar[str] = ' <#> '


class CosineDistance(DistanceBase):
    """Cosine distance via the pgvector ``<=>`` operator.

    Returns ``1 - cosine_similarity(a, b)``, ranging from 0 (identical
    direction) to 2 (opposite direction).  Magnitude-invariant, making it
    the typical choice for semantic-embedding similarity.

    Example::

        Item.objects.annotate(dist=CosineDistance('embedding', [1, 2, 3])).order_by('dist')
    """

    function: ClassVar[str] = ''
    arg_joiner: ClassVar[str] = ' <=> '


class L1Distance(DistanceBase):
    """L1 (Manhattan / taxicab) distance via the pgvector ``<+>`` operator.

    Returns ``sum(|a_i - b_i|)``.  More robust than L2 distance in
    high-dimensional spaces because it penalises large component differences
    less aggressively.

    Example::

        Item.objects.annotate(dist=L1Distance('embedding', [1, 2, 3])).order_by('dist')
    """

    function: ClassVar[str] = ''
    arg_joiner: ClassVar[str] = ' <+> '


class HammingDistance(BitDistanceBase):
    """Hamming distance between two bit strings via the pgvector ``<~>`` operator.

    Counts the number of bit positions at which the two values differ.
    Operates on ``BIT(n)`` columns (use :class:`~pgvector.django.bit.BitField`).

    Example::

        Item.objects.annotate(dist=HammingDistance('signature', b'\\x01')).order_by('dist')
    """

    function: ClassVar[str] = ''
    arg_joiner: ClassVar[str] = ' <~> '


class JaccardDistance(BitDistanceBase):
    """Jaccard distance between two bit strings via the pgvector ``<%%>`` operator.

    Returns ``1 - |A ∩ B| / |A ∪ B|``, i.e. 0 for identical bit strings and
    1 for disjoint ones.  Operates on ``BIT(n)`` columns.

    Example::

        Item.objects.annotate(dist=JaccardDistance('signature', b'\\x01')).order_by('dist')
    """

    function: ClassVar[str] = ''
    arg_joiner: ClassVar[str] = ' <%%> '
