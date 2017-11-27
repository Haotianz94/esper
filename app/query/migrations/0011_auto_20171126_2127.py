# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-11-27 05:27
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0010_default_gender'),
    ]

    operations = [
        migrations.AddField(
            model_name='babycam_facetrack',
            name='max_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='babycam_facetrack',
            name='min_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='babycam_facetrack',
            name='video',
            field=models.ForeignKey(default=709, on_delete=django.db.models.deletion.CASCADE, related_query_name='facetrack', to='query.babycam_Video'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='babycam_posetrack',
            name='max_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='babycam_posetrack',
            name='min_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='babycam_posetrack',
            name='video',
            field=models.ForeignKey(default=709, on_delete=django.db.models.deletion.CASCADE, related_query_name='posetrack', to='query.babycam_Video'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='default_facetrack',
            name='max_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='default_facetrack',
            name='min_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='default_facetrack',
            name='video',
            field=models.ForeignKey(default=709, on_delete=django.db.models.deletion.CASCADE, related_query_name='facetrack', to='query.default_Video'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='default_posetrack',
            name='max_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='default_posetrack',
            name='min_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='default_posetrack',
            name='video',
            field=models.ForeignKey(default=709, on_delete=django.db.models.deletion.CASCADE, related_query_name='posetrack', to='query.default_Video'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tvnews_facetrack',
            name='max_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tvnews_facetrack',
            name='min_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tvnews_facetrack',
            name='video',
            field=models.ForeignKey(default=709, on_delete=django.db.models.deletion.CASCADE, related_query_name='facetrack', to='query.tvnews_Video'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tvnews_posetrack',
            name='max_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tvnews_posetrack',
            name='min_frame',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tvnews_posetrack',
            name='video',
            field=models.ForeignKey(default=709, on_delete=django.db.models.deletion.CASCADE, related_query_name='posetrack', to='query.tvnews_Video'),
            preserve_default=False,
        ),
    ]
