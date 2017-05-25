# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-05-14 17:11
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion
import query.models
import scanner.types_pb2


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Face',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame', models.IntegerField()),
                ('bbox', query.models.ProtoField(scanner.types_pb2.BoundingBox)),
                ('features', models.BinaryField()),
            ],
        ),
        migrations.CreateModel(
            name='Identity',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
                ('classifier', models.BinaryField()),
                ('cohesion', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(max_length=256)),
                ('num_frames', models.IntegerField()),
                ('fps', models.FloatField()),
                ('width', models.IntegerField()),
                ('height', models.IntegerField()),
            ],
        ),
        migrations.AddField(
            model_name='face',
            name='identity',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='query.Identity'),
        ),
        migrations.AddField(
            model_name='face',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Video'),
        ),
    ]