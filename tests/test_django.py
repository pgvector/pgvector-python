import django
from django.conf import settings
from django.db import models
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


class TestDjango(object):
    def test_works(self):
        VectorExtension()
        # item = Item(factors=[1, 2, 3])
        # item.save()
        # Item.objects.order_by(L2Distance('factors', [3, 1, 2]))[:5]
