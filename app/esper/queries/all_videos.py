from esper.prelude import *
from .queries import query

@query("All videos")
def all_videos():
    from query.models import Video
    from esper.widget import qs_to_result
    return qs_to_result(Video.objects.all())

@query("All faces")
def all_faces():
    from query.models import Face
    from esper.widget import qs_to_result
    return qs_to_result(Face.objects.all(), stride=100, limit=10000)

@query("All poses")
def all_poses():
    from query.models import Face
    from esper.widget import qs_to_result
    return qs_to_result(Pose.objects.all(), stride=100, limit=10000)