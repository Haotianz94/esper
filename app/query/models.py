from django.db import models
from . import base_models as base
import math
import numpy as np
import tempfile
import subprocess as sp


class Video(base.Video):
    time = models.DateTimeField()
    srt_extension = base.CharField(default='')

    def get_stride(self):
        return int(math.ceil(self.fps) / 2)

    def item_name(self):
        return '.'.join(self.path.split('/')[-1].split('.')[:-1])

    def url(self, duration='1d'):
        fetch_cmd = 'PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHONPATH gsutil signurl -d {} /app/service-key.json gs://esper/{} ' \
                    .format(duration, self.path)
        url = sp.check_output(fetch_cmd, shell=True).decode('utf-8').split('\n')[1].split('\t')[-1]
        return url


class Tag(models.Model):
    name = base.CharField()


class VideoTag(models.Model):
    video = models.ForeignKey(Video)
    tag = models.ForeignKey(Tag)


class Frame(base.Frame):
    tags = models.ManyToManyField(Tag)


class Labeler(base.Labeler):
    data_path = base.CharField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True, null=True, blank=True)

Labeled = base.Labeled(Labeler)
Track = base.Track(Labeler)


class Segment(Track):
    polarity = models.FloatField(null=True)
    subjectivity = models.FloatField(null=True)


class Shot(Track):
    pass


class Pose(Labeled, base.Pose, models.Model):
    frame = models.ForeignKey(Frame)


class Face(Labeled, base.BoundingBox, models.Model):
    frame = models.ForeignKey(Frame)
    shot = models.ForeignKey(Shot, null=True)
    probability = models.FloatField(default=1.)

    class Meta:
        unique_together = ('labeler', 'frame', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2')
        

class FaceTag(models.Model):
    face = models.ForeignKey(Face)
    score = models.FloatField(default=1.)
    tag = models.ForeignKey(Tag)
    labeler = models.ForeignKey(Labeler)
    
    class Meta:
        unique_together = ('labeler', 'face')


class FaceFeatures(Labeled, base.Features, models.Model):
    face = models.ForeignKey(Face)

    class Meta:
        unique_together = ('labeler', 'face')


class ScannerJob(models.Model):
    name = base.CharField()


class Object(base.BoundingBox, models.Model):
    frame = models.ForeignKey(Frame)
    label = models.IntegerField()
    probability = models.FloatField()