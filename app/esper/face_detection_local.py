import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, Shot
from esper.prelude import Notifier


# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='mtcnn')
LABELED_TAG, _ = Tag.objects.get_or_create(name='mtcnn:labeled')


print("Prepare videos and frames")
db = scannerpy.Database()
videos = Video.objects.filter(path__contains='CS248').order_by('id')[1:]
frames = [[i for i in range(v.num_frames)] for v in videos]


print("Detecting faces")
faces = st.face_detection.detect_faces(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    cache=True,
    run_opts={
            'io_packet_size': 300,
            'work_packet_size': 20,
            'pipeline_instances_per_node': 2,
            'checkpoint_frequency': 1
        },
    device=scannerpy.DeviceType.GPU
)


print("Putting faces in database and tagging all the frames as being labeled")
frames_in_db_already = set([
    (f.video_id, f.number)
    for f in Frame.objects.filter(tags=LABELED_TAG).all()
])

new_faces = []
new_frame_tags = []
for video, framelist, facelist in zip(videos, frames, faces):
    frame_objs = Frame.objects.filter(video_id=video.id).filter(
            number__in=framelist).order_by('number')
    for frame, bboxlist in zip(frame_objs, facelist.load()):
        if (video.id, frame.number) in frames_in_db_already:
            continue
        face_detected = False
        for bbox in bboxlist:
            face_detected = True
            new_faces.append(Face(
                frame=frame,
                bbox_x1=bbox.x1,
                bbox_x2=bbox.x2,
                bbox_y1=bbox.y1,
                bbox_y2=bbox.y2,
                probability=bbox.score,
                labeler=LABELER))
        if face_detected:
            new_frame_tags.append(
                Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))
Face.objects.bulk_create(new_faces)
Frame.tags.through.objects.bulk_create(new_frame_tags)


print("Tagging all the video as being labeled")
videos_tagged_already = set([
    vtag.video_id
    for vtag in VideoTag.objects.filter(tag=LABELED_TAG).all()
])
new_videotags = [
    VideoTag(video=video, tag=LABELED_TAG)
    for video in videos
    if video.id not in videos_tagged_already
]
VideoTag.objects.bulk_create(new_videotags)


Notifier().notify('Done with face detection')