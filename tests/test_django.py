import django
from django.conf import settings
from django.core import serializers
from django.db import connection, migrations, models
from django.db.migrations.loader import MigrationLoader
from django.forms import ModelForm
from math import sqrt
import numpy as np
import pgvector.django
from pgvector.django import VectorExtension, VectorField, IvfflatIndex, HnswIndex, L2Distance, MaxInnerProduct, CosineDistance
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
    embedding = VectorField(dimensions=3)

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
                ('embedding', pgvector.django.VectorField(dimensions=3)),
            ],
        ),
        migrations.AddIndex(
            model_name='item',
            index=pgvector.django.IvfflatIndex(fields=['embedding'], lists=1, name='my_index', opclasses=['vector_l2_ops']),
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
        item = Item(id=i + 1, embedding=v)
        item.save()


class ItemForm(ModelForm):
    class Meta:
        model = Item
        fields = ['embedding']


class TestDjango:
    def setup_method(self, test_method):
        Item.objects.all().delete()

    def test_works(self):
        item = Item(id=1, embedding=[1, 2, 3])
        item.save()
        item = Item.objects.get(pk=1)
        assert item.id == 1
        assert np.array_equal(item.embedding, np.array([1, 2, 3]))
        assert item.embedding.dtype == np.float32

    def test_l2_distance(self):
        create_items()
        distance = L2Distance('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_max_inner_product(self):
        create_items()
        distance = MaxInnerProduct('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_cosine_distance(self):
        create_items()
        distance = CosineDistance('embedding', [1, 1, 1])
        items = Item.objects.annotate(distance=distance).order_by(distance)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_filter(self):
        create_items()
        distance = L2Distance('embedding', [1, 1, 1])
        items = Item.objects.alias(distance=distance).filter(distance__lt=1)
        assert [v.id for v in items] == [1]

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
