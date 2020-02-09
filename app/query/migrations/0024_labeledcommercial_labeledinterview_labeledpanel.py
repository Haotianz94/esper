# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-10-26 10:12
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0023_auto_20181026_1010'),
    ]

    operations = [
        migrations.CreateModel(
            name='LabeledCommercial',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start', models.FloatField()),
                ('end', models.FloatField()),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Video')),
            ],
        ),
        migrations.CreateModel(
            name='LabeledInterview',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start', models.FloatField()),
                ('end', models.FloatField()),
                ('interviewer1', models.CharField(blank=True, default=None, max_length=256, null=True)),
                ('interviewer2', models.CharField(blank=True, default=None, max_length=256, null=True)),
                ('guest1', models.CharField(blank=True, default=None, max_length=256, null=True)),
                ('guest2', models.CharField(blank=True, default=None, max_length=256, null=True)),
                ('original', models.BooleanField(default=True)),
                ('scattered_clips', models.BooleanField(default=False)),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Video')),
            ],
        ),
        migrations.CreateModel(
            name='LabeledPanel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start', models.FloatField()),
                ('end', models.FloatField()),
                ('num_panelists', models.IntegerField()),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Video')),
            ],
        ),
    ]