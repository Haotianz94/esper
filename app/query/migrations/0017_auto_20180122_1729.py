# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-01-22 17:29
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0016_auto_20180122_1704'),
    ]

    operations = [
        migrations.CreateModel(
            name='tvnews_Segment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('min_frame', models.IntegerField()),
                ('max_frame', models.IntegerField()),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='segment', to='query.tvnews_Labeler')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Thing',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
                ('type', models.IntegerField()),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='tvnews_segment',
            name='things',
            field=models.ManyToManyField(related_query_name='segment', to='query.tvnews_Thing'),
        ),
        migrations.AddField(
            model_name='tvnews_segment',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='segment', to='query.tvnews_Video'),
        ),
    ]
