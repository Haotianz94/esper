from query.models import Video
from scannertools import shot_detection, Pipeline
from esper.scannerutil import ScannerWrapper
import scannerpy 
from scannerpy import register_python_op
from scannerpy.stdlib import readers
import struct
from typing import Sequence
from esper.prelude import pcache, par_for
import numpy as np
import os


if __name__ == "__main__":
    print("Prepare videos and frames")
    db = scannerpy.Database()
    videos = Video.objects.filter(path__contains='Tabletennis').order_by('id')[0:1]
    # frames = [[i for i in range(v.num_frames)] for v in videos]
    video_ids = [video.id for video in videos]

    hists = shot_detection.compute_histograms(
        db,
        videos=[v.for_scannertools() for v in videos],
        cache=True,
        run_opts={
            'io_packet_size': 100,
            'work_packet_size': 10,
            'pipeline_instances_per_node': 5,
            'checkpoint_frequency': 1,
        })

    def load_hist(i):
        path = '/app/data/histogram/{:07d}.bin'.format(video_ids[i])
        hist = np.array(list(hists[i].load()), dtype=np.int)
        print(hist.shape)
        print(hist)
        with open(path, 'wb') as f:
            f.write(hist.tobytes())

    print('Loading...')
    par_for(load_hist, list(range(len(hists))), workers=8)
