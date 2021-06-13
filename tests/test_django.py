import django
from django.conf import settings
from django.db import connection, migrations, models
from django.db.migrations.loader import MigrationLoader
import numpy as np
from pgvector.django import VectorExtension, VectorField, IvfflatIndex, L2Distance, MaxInnerProduct, CosineDistance

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
    factors = VectorField(dimensions=3)

    class Meta:
        app_label = 'myapp'
        indexes = [
            IvfflatIndex(
                name='my_index',
                fields=['factors'],
                lists=100,
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
                ('factors', VectorField(dimensions=3)),
            ],
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


class TestDjango(object):
    def test_works(self):
        item = Item(factors=[1, 2, 3])
        item.save()
        items = Item.objects.order_by(L2Distance('factors', [3, 1, 2]))[:5]
        assert items[0].id == 1
        assert np.array_equal(items[0].factors, np.array([1, 2, 3]))
