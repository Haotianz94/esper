import scannerpy 
import scannertools as st
import os
import sys
from django.db.models import Q
from query.models import Video, Face, Frame, Labeler, Tag, VideoTag, Pose
from esper.prelude import Notifier
# from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
from tqdm import tqdm

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='openpose')
LABELED_TAG, _ = Tag.objects.get_or_create(name='openpose:labeled')


print("Prepare videos and frames")
FACE_TAG, _ = Tag.objects.get_or_create(name='mtcnn:labeled')
db = scannerpy.Database()
videos = Video.objects.all().order_by('id')[:1]
frames = [[frame.number for frame in Frame.objects.filter(
    video_id=v.id, tags=FACE_TAG).order_by("number")] for v in videos]


print("Computing poses")
poses = st.pose_detection.detect_poses(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    cache=False,
    run_opts={
            'io_packet_size': 100,
            'work_packet_size': 20,
            'pipeline_instances_per_node': 2,
            'checkpoint_frequency': 1
        },
    device=scannerpy.DeviceType.GPU
)


print("Putting poses in database and tagging all the frames as being labeled")
frames_in_db_already = set([
    (f.video_id, f.number)
    for f in Frame.objects.filter(tags=LABELED_TAG).all()
])

for video, framelist, poseframelist in tqdm(zip(videos, frames, poses), total=len(videos)):
    new_poses = []
    new_frame_tags = []

    frame_objs = Frame.objects.filter(video_id=video.id, number__in=framelist).order_by('number')
    for frame, posesinframe in zip(frame_objs, poseframelist.load()):
        if (video.id, frame.number) in frames_in_db_already:
            continue
        pose_detected = False
        for pose in posesinframe:
            pose_detected = True
            new_pose = Pose(
                keypoints=pose.keypoints.tobytes(),
                labeler=LABELER,
                frame=frame
            )
            new_poses.append(new_pose)
        if pose_detected:
            new_frame_tags.append(
                Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))
    Pose.objects.bulk_create(new_poses, batch_size=10000)
    Frame.tags.through.objects.bulk_create(new_frame_tags, batch_size=10000)


# print("Tagging all the frames as being labeled")
# for video, framelist in tqdm(zip(videos, frames), total=len(videos)):
#     frame_objs = Frame.objects.filter(video_id=video.id, number__in=framelist)
#     for frame in frame_objs:
#         if (video.id, frame.number) in frames_in_db_already:
#             continue
#         new_frame_tags.append(
#             Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))


print("Tagging this video as being labeled")
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


Notifier().notify('Done with pose detection')