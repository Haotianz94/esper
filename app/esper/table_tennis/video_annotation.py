from esper.table_tennis.utils import *
from esper.widget import *
import cv2
import random
import pickle

from rekall.stdlib.ingest import attrgetter_accessor, ism_from_django_qs, ism_from_iterable_with_schema_bounds1D


def get_two_people_intrvlcol(video_id):
    two_people_intrvlcol_all = \
        ism_from_django_qs(
            Pose.objects.filter(frame__video__id=video_id) \
                .annotate(video_id=F('frame__video_id')) \
                .annotate(min_frame=F('frame__number')) \
                .annotate(max_frame=F('frame__number')),
        bounds_class=Bounds1D, \
        with_payload=lambda obj: [attrgetter_accessor(obj, 'id')]) \
    .coalesce(('t1', 't2'), bounds_merge_op=Bounds1D.span, payload_merge_op=payload_plus) \
    .filter(payload_satisfies(length_exactly(2)))

    two_people_intrvlcol_short = \
        two_people_intrvlcol_all \
            .dilate(10) \
            .coalesce(('t1', 't2'), bounds_merge_op=Bounds1D.span, payload_merge_op=payload_plus) \
            .dilate(-10) \
    
    two_people_intrvlcol_long = \
        two_people_intrvlcol_all \
            .dilate(5) \
            .coalesce(('t1', 't2'), bounds_merge_op=Bounds1D.span, payload_merge_op=payload_plus) \
            .dilate(-5) \
            .filter_length(min_length=100)

    return two_people_intrvlcol_short, two_people_intrvlcol_long



