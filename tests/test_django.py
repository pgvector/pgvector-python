import django
from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import OpClass
from django.core import serializers
from django.db import connection, migrations, models
from django.db.models import Avg, Sum, FloatField, DecimalField
from django.db.models.functions import Cast
from django.db.migrations.loader import MigrationLoader
from django.forms import ModelForm
from math import sqrt
import numpy as np
import os
import pgvector.django
from pgvector.django import VectorExtension, VectorField, HalfVectorField, BitField, SparseVectorField, IvfflatIndex, HnswIndex, L2Distance, MaxInnerProduct, CosineDistance, L1Distance, HammingDistance, JaccardDistance, HalfVector, SparseVector
from unittest import mock

settings.configure(
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'pgvector_python_test',
        }
    },
    DEBUG=('VERBOSE' in os.environ),
    LOGGING={
        'version': 1,
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler'
            }
        },
        'loggers': {
            'django.db.backends': {
                'handlers': ['console'],
                'level': 'DEBUG'
            },
            'django.db.backends.schema': {
                'level': 'WARNING'
            }
        }
    },
    # needed for OpClass
    # https://docs.djangoproject.com/en/5.1/ref/contrib/postgres/indexes/#opclass-expressions
    INSTALLED_APPS=[
        'django.contrib.postgres'
    ]
)
django.setup()


class Item(models.Model):
    embedding = VectorField(dimensions=3, null=True, blank=True)
    half_embedding = HalfVectorField(dimensions=3, null=True, blank=True)
    binary_embedding = BitField(length=3, null=True, blank=True)
    sparse_embedding = SparseVectorField(dimensions=3, null=True, blank=True)
    embeddings = ArrayField(VectorField(dimensions=3), null=True, blank=True)
    double_embedding = ArrayField(FloatField(), null=True, blank=True)
    numeric_embedding = ArrayField(DecimalField(max_digits=20, decimal_places=10), null=True, blank=True)

    class Meta:
        app_label = 'django_app'
        indexes = [
            IvfflatIndex(
                name='ivfflat_idx',
                fields=['embedding'],
                lists=100,
                opclasses=['vector_l2_ops']
            ),
            HnswIndex(
                name='hnsw_idx',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            ),
            HnswIndex(
                OpClass(Cast('embedding', HalfVectorField(dimensions=3)), name='halfvec_l2_ops'),
                name='hnsw_half_precision_idx',
                m=16,
                ef_construction=64
            )
        ]


class Migration(migrations.Migration):
    initial = True

    operations = [
        VectorExtension(),
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', pgvector.django.VectorField(dimensions=3, null=True, blank=True)),
                ('half_embedding', pgvector.django.HalfVectorField(dimensions=3, null=True, blank=True)),
                ('binary_embedding', pgvector.django.BitField(length=3, null=True, blank=True)),
                ('sparse_embedding', pgvector.django.SparseVectorField(dimensions=3, null=True, blank=True)),
                ('embeddings', ArrayField(pgvector.django.VectorField(dimensions=3), null=True, blank=True)),
                ('double_embedding', ArrayField(FloatField(), null=True, blank=True)),
                ('numeric_embedding', ArrayField(DecimalField(max_digits=20, decimal_places=10), null=True, blank=True)),
            ],
        ),
        migrations.AddIndex(
            model_name='item',
            index=pgvector.django.IvfflatIndex(fields=['embedding'], lists=1, name='ivfflat_idx', opclasses=['vector_l2_ops']),
        ),
        migrations.AddIndex(
            model_name='item',
            index=pgvector.django.HnswIndex(fields=['embedding'], m=16, ef_construction=64, name='hnsw_idx', opclasses=['vector_l2_ops']),
        ),
        migrations.AddIndex(
            model_name='item',
            index=pgvector.django.HnswIndex(OpClass(Cast('embedding', HalfVectorField(dimensions=3)), name='halfvec_l2_ops'), m=16, ef_construction=64, name='hnsw_half_precision_idx'),
        )
    ]


# probably a better way to do this
migration = Migration('initial', 'django_app')
loader = MigrationLoader(connection, replace_migrations=False)
loader.graph.add_node(('django_app', migration.name), migration)
sql_statements = loader.collect_sql([(migration, False)])

with connection.cursor() as cursor:
    cursor.execute("DROP TABLE IF EXISTS django_app_item")
    cursor.execute('\n'.join(sql_statements))


def create_items():
    Item(id=1, embedding=[1, 1, 1], half_embedding=[1, 1, 1], binary_embedding='000', sparse_embedding=SparseVector([1, 1, 1])).save()
    Item(id=2, embedding=[2, 2, 2], half_embedding=[2, 2, 2], binary_embedding='101', sparse_embedding=SparseVector([2, 2, 2])).save()
    Item(id=3, embedding=[1, 1, 2], half_embedding=[1, 1, 2], binary_embedding='111', sparse_embedding=SparseVector([1, 1, 2])).save()


class VectorForm(ModelForm):
    class Meta:
        model = Item
        fields = ['embedding']


class HalfVectorForm(ModelForm):
    class Meta:
        model = Item
        fields = ['half_embedding']


class BitForm(ModelForm):
    class Meta:
        model = Item
        fields = ['binary_embedding']


class SparseVectorForm(ModelForm):
    class Meta:
        model = Item
        fields = ['sparse_embedding']


class TestDjango:
    def setup_method(self, test_method):
        Item.objects.all().delete()

    def test_vector(self):
        Item(id=1, embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        assert np.array_equal(item.embedding, np.array([1, 2, 3]))
        assert item.embedding.dtype == np.float32

    def test_vector_l2_distance(self):
        create_items()
        distance = L2Distance('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_vector_max_inner_product(self):
        create_items()
        distance = MaxInnerProduct('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_vector_cosine_distance(self):
        create_items()
        distance = CosineDistance('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_vector_l1_distance(self):
        create_items()
        distance = L1Distance('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]

    def test_halfvec(self):
        Item(id=1, half_embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        assert item.half_embedding.to_list() == [1, 2, 3]

    def test_halfvec_l2_distance(self):
        create_items()
        distance = L2Distance('half_embedding', HalfVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_halfvec_max_inner_product(self):
        create_items()
        distance = MaxInnerProduct('half_embedding', HalfVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_halfvec_cosine_distance(self):
        create_items()
        distance = CosineDistance('half_embedding', HalfVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_halfvec_l1_distance(self):
        create_items()
        distance = L1Distance('half_embedding', HalfVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]

    def test_bit(self):
        Item(id=1, binary_embedding='101').save()
        item = Item.objects.get(pk=1)
        assert item.binary_embedding == '101'

    def test_bit_hamming_distance(self):
        create_items()
        distance = HammingDistance('binary_embedding', '101')
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [0, 1, 2]

    def test_bit_jaccard_distance(self):
        create_items()
        distance = JaccardDistance('binary_embedding', '101')
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        # assert [v.distance for v in items] == [0, 1/3, 1]

    def test_sparsevec(self):
        Item(id=1, sparse_embedding=SparseVector([1, 2, 3])).save()
        item = Item.objects.get(pk=1)
        assert item.sparse_embedding.to_list() == [1, 2, 3]

    def test_sparsevec_l2_distance(self):
        create_items()
        distance = L2Distance('sparse_embedding', SparseVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_sparsevec_max_inner_product(self):
        create_items()
        distance = MaxInnerProduct('sparse_embedding', SparseVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_sparsevec_cosine_distance(self):
        create_items()
        distance = CosineDistance('sparse_embedding', SparseVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_sparsevec_l1_distance(self):
        create_items()
        distance = L1Distance('sparse_embedding', SparseVector([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]

    def test_filter(self):
        create_items()
        distance = L2Distance('embedding', [1, 1, 1])
        items = Item.objects.alias(distance=distance).filter(distance__lt=1)
        assert [v.id for v in items] == [1]

    def test_vector_avg(self):
        avg = Item.objects.aggregate(Avg('embedding'))['embedding__avg']
        assert avg is None
        Item(embedding=[1, 2, 3]).save()
        Item(embedding=[4, 5, 6]).save()
        avg = Item.objects.aggregate(Avg('embedding'))['embedding__avg']
        assert np.array_equal(avg, np.array([2.5, 3.5, 4.5]))

    def test_vector_sum(self):
        sum = Item.objects.aggregate(Sum('embedding'))['embedding__sum']
        assert sum is None
        Item(embedding=[1, 2, 3]).save()
        Item(embedding=[4, 5, 6]).save()
        sum = Item.objects.aggregate(Sum('embedding'))['embedding__sum']
        assert np.array_equal(sum, np.array([5, 7, 9]))

    def test_halfvec_avg(self):
        avg = Item.objects.aggregate(Avg('half_embedding'))['half_embedding__avg']
        assert avg is None
        Item(half_embedding=[1, 2, 3]).save()
        Item(half_embedding=[4, 5, 6]).save()
        avg = Item.objects.aggregate(Avg('half_embedding'))['half_embedding__avg']
        assert avg.to_list() == [2.5, 3.5, 4.5]

    def test_halfvec_sum(self):
        sum = Item.objects.aggregate(Sum('half_embedding'))['half_embedding__sum']
        assert sum is None
        Item(half_embedding=[1, 2, 3]).save()
        Item(half_embedding=[4, 5, 6]).save()
        sum = Item.objects.aggregate(Sum('half_embedding'))['half_embedding__sum']
        assert sum.to_list() == [5, 7, 9]

    def test_serialization(self):
        create_items()
        items = Item.objects.all()
        for format in ['json', 'xml']:
            data = serializers.serialize(format, items)
            with mock.patch('django.core.serializers.python.apps.get_model') as get_model:
                get_model.return_value = Item
                for obj in serializers.deserialize(format, data):
                    obj.save()

    def test_vector_form(self):
        form = VectorForm(data={'embedding': '[1, 2, 3]'})
        assert form.is_valid()
        assert 'value="[1, 2, 3]"' in form.as_div()

    def test_vector_form_instance(self):
        Item(id=1, embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = VectorForm(instance=item)
        assert 'value="[1.0, 2.0, 3.0]"' in form.as_div()

    def test_vector_form_save(self):
        Item(id=1, embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = VectorForm(instance=item, data={'embedding': '[4, 5, 6]'})
        assert form.has_changed()
        assert form.is_valid()
        assert form.save()
        assert [4, 5, 6] == Item.objects.get(pk=1).embedding.tolist()

    def test_vector_form_save_missing(self):
        Item(id=1).save()
        item = Item.objects.get(pk=1)
        form = VectorForm(instance=item, data={'embedding': ''})
        assert form.is_valid()
        assert form.save()
        assert Item.objects.get(pk=1).embedding is None

    def test_halfvec_form(self):
        form = HalfVectorForm(data={'half_embedding': '[1, 2, 3]'})
        assert form.is_valid()
        assert 'value="[1, 2, 3]"' in form.as_div()

    def test_halfvec_form_instance(self):
        Item(id=1, half_embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = HalfVectorForm(instance=item)
        assert 'value="[1.0, 2.0, 3.0]"' in form.as_div()

    def test_halfvec_form_save(self):
        Item(id=1, half_embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = HalfVectorForm(instance=item, data={'half_embedding': '[4, 5, 6]'})
        assert form.has_changed()
        assert form.is_valid()
        assert form.save()
        assert [4, 5, 6] == Item.objects.get(pk=1).half_embedding.to_list()

    def test_halfvec_form_save_missing(self):
        Item(id=1).save()
        item = Item.objects.get(pk=1)
        form = HalfVectorForm(instance=item, data={'half_embedding': ''})
        assert form.is_valid()
        assert form.save()
        assert Item.objects.get(pk=1).half_embedding is None

    def test_bit_form(self):
        form = BitForm(data={'binary_embedding': '101'})
        assert form.is_valid()
        assert 'value="101"' in form.as_div()

    def test_bit_form_instance(self):
        Item(id=1, binary_embedding='101').save()
        item = Item.objects.get(pk=1)
        form = BitForm(instance=item)
        assert 'value="101"' in form.as_div()

    def test_bit_form_save(self):
        Item(id=1, binary_embedding='101').save()
        item = Item.objects.get(pk=1)
        form = BitForm(instance=item, data={'binary_embedding': '010'})
        assert form.has_changed()
        assert form.is_valid()
        assert form.save()
        assert '010' == Item.objects.get(pk=1).binary_embedding

    def test_bit_form_save_missing(self):
        Item(id=1).save()
        item = Item.objects.get(pk=1)
        form = BitForm(instance=item, data={'binary_embedding': ''})
        assert form.is_valid()
        assert form.save()
        assert Item.objects.get(pk=1).binary_embedding is None

    def test_sparsevec_form(self):
        form = SparseVectorForm(data={'sparse_embedding': '{1:1,2:2,3:3}/3'})
        assert form.is_valid()
        assert 'value="{1:1,2:2,3:3}/3"' in form.as_div()

    def test_sparsevec_form_instance(self):
        Item(id=1, sparse_embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = SparseVectorForm(instance=item)
        # TODO improve
        assert 'value="{1:1.0,2:2.0,3:3.0}/3"' in form.as_div()

    def test_sparsevec_form_save(self):
        Item(id=1, sparse_embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = SparseVectorForm(instance=item, data={'sparse_embedding': '{1:4,2:5,3:6}/3'})
        assert form.has_changed()
        assert form.is_valid()
        assert form.save()
        assert [4, 5, 6] == Item.objects.get(pk=1).sparse_embedding.to_list()

    def test_sparesevec_form_save_missing(self):
        Item(id=1).save()
        item = Item.objects.get(pk=1)
        form = SparseVectorForm(instance=item, data={'sparse_embedding': ''})
        assert form.is_valid()
        assert form.save()
        assert Item.objects.get(pk=1).sparse_embedding is None

    def test_clean(self):
        item = Item(id=1, embedding=[1, 2, 3], half_embedding=[1, 2, 3], binary_embedding='101', sparse_embedding=SparseVector([1, 2, 3]))
        item.full_clean()

    def test_get_or_create(self):
        Item.objects.get_or_create(embedding=[1, 2, 3])

    def test_missing(self):
        Item().save()
        assert Item.objects.first().embedding is None
        assert Item.objects.first().half_embedding is None
        assert Item.objects.first().binary_embedding is None
        assert Item.objects.first().sparse_embedding is None

    def test_vector_array(self):
        Item(id=1, embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]).save()

        with connection.cursor() as cursor:
            from pgvector.psycopg import register_vector
            register_vector(cursor.connection)

            # this fails if the driver does not cast arrays
            item = Item.objects.get(pk=1)
            assert item.embeddings[0].tolist() == [1, 2, 3]
            assert item.embeddings[1].tolist() == [4, 5, 6]

    def test_double_array(self):
        Item(id=1, double_embedding=[1, 1, 1]).save()
        Item(id=2, double_embedding=[2, 2, 2]).save()
        Item(id=3, double_embedding=[1, 1, 2]).save()
        distance = L2Distance(Cast('double_embedding', VectorField()), [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]
        assert items[1].double_embedding == [1, 1, 2]

    def test_numeric_array(self):
        Item(id=1, numeric_embedding=[1, 1, 1]).save()
        Item(id=2, numeric_embedding=[2, 2, 2]).save()
        Item(id=3, numeric_embedding=[1, 1, 2]).save()
        distance = L2Distance(Cast('numeric_embedding', VectorField()), [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]
        assert items[1].numeric_embedding == [1, 1, 2]

    def test_half_precision(self):
        create_items()
        distance = L2Distance(Cast('embedding', HalfVectorField(dimensions=3)), [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]
