import django
from django.conf import settings
from django.core import serializers
from django.db import connection, migrations, models
from django.db.models import Avg, Sum
from django.db.migrations.loader import MigrationLoader
from django.forms import ModelForm
from math import sqrt
import numpy as np
import pgvector.django
from pgvector.django import VectorExtension, VectorField, HalfvecField, BitField, SparsevecField, IvfflatIndex, HnswIndex, L2Distance, MaxInnerProduct, CosineDistance, L1Distance, HammingDistance, JaccardDistance, HalfVec, SparseVec
from unittest import mock

settings.configure(
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'pgvector_python_test',
        }
    }
)
django.setup()


class Item(models.Model):
    embedding = VectorField(dimensions=3, null=True, blank=True)
    half_embedding = HalfvecField(dimensions=3, null=True, blank=True)
    binary_embedding = BitField(length=3, null=True, blank=True)
    sparse_embedding = SparsevecField(dimensions=3, null=True, blank=True)

    class Meta:
        app_label = 'myapp'
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
                ef_construction=100,
                opclasses=['vector_l2_ops']
            )
        ]


class Migration(migrations.Migration):
    initial = True

    dependencies = [
    ]

    operations = [
        VectorExtension(),
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', pgvector.django.VectorField(dimensions=3, null=True, blank=True)),
                ('half_embedding', pgvector.django.HalfvecField(dimensions=3, null=True, blank=True)),
                ('binary_embedding', pgvector.django.BitField(length=3, null=True, blank=True)),
                ('sparse_embedding', pgvector.django.SparsevecField(dimensions=3, null=True, blank=True)),
            ],
        ),
        migrations.AddIndex(
            model_name='item',
            index=pgvector.django.IvfflatIndex(fields=['embedding'], lists=1, name='ivfflat_idx', opclasses=['vector_l2_ops']),
        ),
        migrations.AddIndex(
            model_name='item',
            index=pgvector.django.HnswIndex(fields=['embedding'], m=16, ef_construction=64, name='hnsw_idx', opclasses=['vector_l2_ops']),
        )
    ]


# probably a better way to do this
migration = Migration('initial', 'myapp')
loader = MigrationLoader(connection, replace_migrations=False)
loader.graph.add_node(('myapp', migration.name), migration)
sql_statements = loader.collect_sql([(migration, False)])

with connection.cursor() as cursor:
    cursor.execute("DROP TABLE IF EXISTS myapp_item")
    cursor.execute('\n'.join(sql_statements))


def create_items():
    vectors = [
        [1, 1, 1],
        [2, 2, 2],
        [1, 1, 2]
    ]
    for i, v in enumerate(vectors):
        item = Item(id=i + 1, embedding=v, half_embedding=v, sparse_embedding=SparseVec.from_dense(v))
        item.save()


class ItemForm(ModelForm):
    class Meta:
        model = Item
        fields = ['embedding']


class TestDjango:
    def setup_method(self, test_method):
        Item.objects.all().delete()

    def test_vector(self):
        item = Item(id=1, embedding=[1, 2, 3])
        item.save()
        item = Item.objects.get(pk=1)
        assert item.id == 1
        assert np.array_equal(item.embedding, np.array([1, 2, 3]))
        assert item.embedding.dtype == np.float32

    def test_halfvec(self):
        item = Item(id=1, half_embedding=[1, 2, 3])
        item.save()
        item = Item.objects.get(pk=1)
        assert item.id == 1
        assert item.half_embedding.to_list() == [1, 2, 3]

    def test_sparsevec(self):
        item = Item(id=1, sparse_embedding=SparseVec.from_dense([1, 2, 3]))
        item.save()
        item = Item.objects.get(pk=1)
        assert item.id == 1
        assert item.sparse_embedding.to_dense() == [1, 2, 3]

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

    def test_halfvec_l2_distance(self):
        create_items()
        distance = L2Distance('half_embedding', HalfVec([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_sparsevec_l2_distance(self):
        create_items()
        distance = L2Distance('sparse_embedding', SparseVec.from_dense([1, 1, 1]))
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_bit_hamming_distance(self):
        Item(id=1, binary_embedding='000').save()
        Item(id=2, binary_embedding='101').save()
        Item(id=3, binary_embedding='111').save()
        distance = HammingDistance('binary_embedding', '101')
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [0, 1, 2]

    def test_bit_jaccard_distance(self):
        Item(id=1, binary_embedding='000').save()
        Item(id=2, binary_embedding='101').save()
        Item(id=3, binary_embedding='111').save()
        distance = JaccardDistance('binary_embedding', '101')
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        # assert [v.distance for v in items] == [0, 1/3, 1]

    def test_filter(self):
        create_items()
        distance = L2Distance('embedding', [1, 1, 1])
        items = Item.objects.alias(distance=distance).filter(distance__lt=1)
        assert [v.id for v in items] == [1]

    def test_avg(self):
        avg = Item.objects.aggregate(Avg('embedding'))['embedding__avg']
        assert avg is None
        Item(embedding=[1, 2, 3]).save()
        Item(embedding=[4, 5, 6]).save()
        avg = Item.objects.aggregate(Avg('embedding'))['embedding__avg']
        assert np.array_equal(avg, np.array([2.5, 3.5, 4.5]))

    def test_sum(self):
        sum = Item.objects.aggregate(Sum('embedding'))['embedding__sum']
        assert sum is None
        Item(embedding=[1, 2, 3]).save()
        Item(embedding=[4, 5, 6]).save()
        sum = Item.objects.aggregate(Sum('embedding'))['embedding__sum']
        assert np.array_equal(sum, np.array([5, 7, 9]))

    def test_serialization(self):
        create_items()
        items = Item.objects.all()
        for format in ['json', 'xml']:
            data = serializers.serialize(format, items)
            with mock.patch('django.core.serializers.python.apps.get_model') as get_model:
                get_model.return_value = Item
                for obj in serializers.deserialize(format, data):
                    obj.save()

    def test_form(self):
        form = ItemForm(data={'embedding': '[1, 2, 3]'})
        assert form.is_valid()
        assert 'value="[1, 2, 3]"' in form.as_div()

    def test_form_instance(self):
        Item(id=1, embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = ItemForm(instance=item)
        assert 'value="[1.0, 2.0, 3.0]"' in form.as_div()

    def test_form_save(self):
        Item(id=1, embedding=[1, 2, 3]).save()
        item = Item.objects.get(pk=1)
        form = ItemForm(instance=item, data={'embedding': '[4, 5, 6]'})
        assert form.has_changed()
        assert form.is_valid()
        assert form.save()
        assert [4, 5, 6] == Item.objects.get(pk=1).embedding.tolist()

    def test_form_save_missing(self):
        Item(id=1).save()
        item = Item.objects.get(pk=1)
        form = ItemForm(instance=item, data={'embedding': ''})
        assert form.is_valid()
        assert form.save()
        assert Item.objects.get(pk=1).embedding is None

    def test_clean(self):
        item = Item(id=1, embedding=[1, 2, 3])
        item.full_clean()

    def test_get_or_create(self):
        Item.objects.get_or_create(embedding=[1, 2, 3])

    def test_missing(self):
        Item().save()
        assert Item.objects.first().embedding is None
