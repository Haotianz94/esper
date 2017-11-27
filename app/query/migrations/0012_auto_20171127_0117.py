# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-11-27 09:17
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0011_auto_20171126_2127'),
    ]

    operations = [
        migrations.AddField(
            model_name='babycam_facefeatures',
            name='distto',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='babycam_posefeatures',
            name='distto',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='default_facefeatures',
            name='distto',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='default_posefeatures',
            name='distto',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='tvnews_facefeatures',
            name='distto',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='tvnews_posefeatures',
            name='distto',
            field=models.FloatField(null=True),
        ),
    ]
