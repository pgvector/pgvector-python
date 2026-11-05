from django.contrib.postgres.indexes import PostgresIndex
from typing import Any


class IvfflatIndex(PostgresIndex):
    """Django index class for the pgvector IVFFlat (Inverted File Flat) index type.

    IVFFlat partitions vectors into *lists* (Voronoi cells) at build time and
    restricts each query to the *probes* nearest lists.  It offers fast approximate
    nearest-neighbour search with a controllable recall/speed trade-off.

    Build-time cost is proportional to ``lists``; query-time cost is proportional to
    ``probes`` (a session-level GUC, ``ivfflat.probes``, not an index parameter).
    The index **requires** the table to contain data before it is created — an empty
    table produces degenerate clusters.

    Supported distance operators (set via ``opclasses``):

    - ``vector_l2_ops``    — L2 / Euclidean distance (``<->``)
    - ``vector_ip_ops``    — inner product (``<#>``)
    - ``vector_cosine_ops`` — cosine distance (``<=>``)
    - ``halfvec_l2_ops`` / ``halfvec_ip_ops`` / ``halfvec_cosine_ops`` — half-precision variants
    - ``sparsevec_l2_ops`` / ``sparsevec_ip_ops`` / ``sparsevec_cosine_ops`` — sparse-vector variants

    Example::

        from pgvector.django import IvfflatIndex

        class Meta:
            indexes = [
                IvfflatIndex(
                    name='item_embedding_ivfflat',
                    fields=['embedding'],
                    lists=100,
                    opclasses=['vector_cosine_ops'],
                )
            ]

    See also: https://github.com/pgvector/pgvector#ivfflat
    """

    suffix = 'ivfflat'

    def __init__(self, *expressions: Any, lists: int | None = None, **kwargs: Any) -> None:
        """Create an IVFFlat index.

        Args:
            *expressions: Column expressions to index (forwarded to
                :class:`~django.contrib.postgres.indexes.PostgresIndex`).
            lists: Number of lists (Voronoi cells) to partition the vector
                space into during index construction.  Controls the
                accuracy/speed trade-off at *build* time.

                - Higher values improve recall but increase build time and
                  memory consumption.
                - As a rule of thumb pgvector recommends ``rows / 1000`` for
                  up to 1 million rows, and ``sqrt(rows)`` beyond that.
                - Valid range: 1 – 32768.  pgvector default: ``100``.
                - ``None`` omits the parameter and lets the server use its
                  compiled-in default.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~django.contrib.postgres.indexes.PostgresIndex`
                (e.g. ``name``, ``fields``, ``opclasses``, ``condition``).
        """
        self.lists = lists
        super().__init__(*expressions, **kwargs)

    def deconstruct(self) -> tuple[Any, Any, Any]:
        """Return enough information to recreate the index via migrations.

        Returns:
            A ``(path, args, kwargs)`` tuple as required by Django's
            deconstruction protocol.  ``lists`` is included in *kwargs* only
            when it was explicitly set.
        """
        path, args, kwargs = super().deconstruct()
        if self.lists is not None:
            kwargs['lists'] = self.lists
        return path, args, kwargs

    def get_with_params(self) -> list[str]:
        """Return the ``WITH (...)`` storage parameters for the ``CREATE INDEX`` DDL.

        Returns:
            A list of ``'key = value'`` strings passed to the PostgreSQL
            ``WITH`` clause.  Empty list when no parameters were specified.
        """
        with_params = []
        if self.lists is not None:
            with_params.append('lists = %d' % self.lists)
        return with_params


class HnswIndex(PostgresIndex):
    """Django index class for the pgvector HNSW (Hierarchical Navigable Small World) index type.

    HNSW builds a multi-layer proximity graph over the vectors.  It generally
    provides higher recall and faster queries than :class:`IvfflatIndex` at the
    cost of more memory and a longer build time.  Unlike IVFFlat it can be built
    on an empty table and updated incrementally.

    Query-time recall is tuned with ``hnsw.ef_search`` (a session-level GUC),
    not an index parameter.

    Supported distance operators (set via ``opclasses``):

    - ``vector_l2_ops``    — L2 / Euclidean distance (``<->``)
    - ``vector_ip_ops``    — inner product (``<#>``)
    - ``vector_cosine_ops`` — cosine distance (``<=>``)
    - ``vector_l1_ops``    — L1 / Manhattan distance (``<+>``)
    - ``halfvec_*`` / ``sparsevec_*`` variants are also available

    Example::

        from pgvector.django import HnswIndex

        class Meta:
            indexes = [
                HnswIndex(
                    name='item_embedding_hnsw',
                    fields=['embedding'],
                    m=16,
                    ef_construction=64,
                    opclasses=['vector_cosine_ops'],
                )
            ]

    See also: https://github.com/pgvector/pgvector#hnsw
    """

    suffix = 'hnsw'

    def __init__(
        self,
        *expressions: Any,
        m: int | None = None,
        ef_construction: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an HNSW index.

        Args:
            *expressions: Column expressions to index (forwarded to
                :class:`~django.contrib.postgres.indexes.PostgresIndex`).
            m: Maximum number of bidirectional links (edges) created for each
                new node during graph construction.  Controls the graph
                connectivity at *build* time.

                - Higher values yield better recall and faster queries but
                  consume more memory (roughly ``m * 8`` bytes per vector per
                  layer beyond the base layer).
                - Valid range: 2 – 100.  pgvector default: ``16``.
                - ``None`` omits the parameter and lets the server use its
                  compiled-in default.
            ef_construction: Size of the dynamic candidate list maintained
                while inserting a new vector into the graph.  Controls the
                quality of the *build*-time graph structure.

                - Higher values produce a denser, better-connected graph with
                  improved recall, at the cost of a slower build.
                - Must satisfy ``ef_construction >= 2 * m``.
                - Valid range: 4 – 1000.  pgvector default: ``64``.
                - ``None`` omits the parameter and lets the server use its
                  compiled-in default.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~django.contrib.postgres.indexes.PostgresIndex`
                (e.g. ``name``, ``fields``, ``opclasses``, ``condition``).
        """
        self.m = m
        self.ef_construction = ef_construction
        super().__init__(*expressions, **kwargs)

    def deconstruct(self) -> tuple[Any, Any, Any]:
        """Return enough information to recreate the index via migrations.

        Returns:
            A ``(path, args, kwargs)`` tuple as required by Django's
            deconstruction protocol.  ``m`` and ``ef_construction`` are
            included in *kwargs* only when they were explicitly set.
        """
        path, args, kwargs = super().deconstruct()
        if self.m is not None:
            kwargs['m'] = self.m
        if self.ef_construction is not None:
            kwargs['ef_construction'] = self.ef_construction
        return path, args, kwargs

    def get_with_params(self) -> list[str]:
        """Return the ``WITH (...)`` storage parameters for the ``CREATE INDEX`` DDL.

        Returns:
            A list of ``'key = value'`` strings passed to the PostgreSQL
            ``WITH`` clause.  Empty list when no parameters were specified.
        """
        with_params = []
        if self.m is not None:
            with_params.append('m = %d' % self.m)
        if self.ef_construction is not None:
            with_params.append('ef_construction = %d' % self.ef_construction)
        return with_params
