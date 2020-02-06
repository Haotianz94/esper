from scannertools import audio
from scannertools.transcript_alignment import align_transcript_pipeline, align_transcript_pipeline2, TranscriptAligner
from query.models import Video
from esper.kube import make_cluster, cluster_config, worker_config
from esper.load_aligned_transcript import callback

import scannerpy
import os
import pickle
import math
import tempfile
import re
import sys

################################### Instructions ###################################
# 1. Please refer to /app/notebooks/aligment.ipynb for details of some preprocessing and post-processing steps
# 2. We use FIRST_RUN=True with ESTIMATE = False to run aligment for the first round
# 3. We use FIRST_RUN=False with ESTIMATE = True to run aligment for the second round
# 4. Do check the /opt/scannertools/scannertools/transcript_aligment.py to make sure the code matches the round(3 places in total), changes will be copied into cluster nodes


SEG_LENGTH = 60     # Window size(seconds) for running gentle
FIRST_RUN = True    # True if running alignment for the first time, False for re-run on bad aligned videos
ESTIMATE = False    # True for using estimate to tackle larger amount of mis-align
BATCH_LOAD = False  # True when loading results from scanner database
LOCAL_RUN = False   # True if running on local machine, False for running on Kubernete

video_list_file = '/app/data/video_list_kdd.txt'
additional_field_file = '/app/data/addtional_field_kdd.pkl'
result_file = '/app/result/align_stats_kdd.pkl'
subs_dir = 'tvnews/subs_kdd/'

if __name__ == "__main__":
    
#     video_start = int(sys.argv[1])
#     print(video_start)

    # Set test video list
    video_list = open(video_list_file, 'r').read().split('\n')
    video_list = [int(vid) for vid in video_list]
    videos = Video.objects.filter(id__in=video_list)
    
    # Remove videos have incomplete transcript
    addtional_field = pickle.load(open(additional_field_file, 'rb'))
    videos = [video for video in videos if addtional_field[video.id]['valid_transcript']]
    
    # Remove videos have inequal audio/frame time
    videos_valid = []
    for video in videos:
        audio_time = addtional_field[video.id]['audio_duration']
        frame_time = video.num_frames / video.fps
        if audio_time / frame_time < 1.1 and audio_time / frame_time > 0.9:
            videos_valid.append(video)
    videos = videos_valid
    print("Videos valid: ", len(videos))
    
    if FIRST_RUN:   
        # Remove already dumped videos
        if os.path.exists(result_file):
            res_stats = pickle.load(open(result_file, 'rb'))
        else:
            res_stats = {}
        videos = [video for video in videos if video.id not in res_stats]
        print('Videos unfinished:', len(videos))
    else:
        # Re-run bad align videos
        if os.path.exists(result_file):
            res_stats = pickle.load(open(result_file, 'rb'))
        else:
            res_stats = {}
        videos = [video for video in videos if video.id in res_stats and res_stats[video.id]['word_missing'] > 0.2]
        print("Videos unfinished: ", len(videos))
    
    db = scannerpy.Database()
    # Remove videos not saved in database
    if FIRST_RUN or BATCH_LOAD:
        meta = db._load_db_metadata()
        tables_in_db= {t.name for t in meta.tables}
        videos_uncommitted = []
        tables_committed = []
        for video in videos:
            table_name = '{}_align_transcript{}'.format(video.path, '2' if ESTIMATE else '')
            if table_name not in tables_in_db:
                videos_uncommitted.append(video)
            else:
                table = db.table(table_name)
                if not table.committed():
                    videos_uncommitted.append(video)
                else:
                    tables_committed.append(table)
        videos = videos_uncommitted
        print("Videos uncommitted:", len(videos_uncommitted))
        print("Videos committed:", len(tables_committed))

    exit()
    
############################################################################
#Dump results with batch
############################################################################
    if BATCH_LOAD:
        db.batch_load(tables_committed, 'align_transcript{}'.format('2' if ESTIMATE else ''), callback)
        exit()

############################################################################
# Prepare Alignment
############################################################################
    
    # Load audios from videos
    audios = [audio.AudioSource(video.for_scannertools(), 
                                frame_size=SEG_LENGTH, 
                                duration=addtional_field[video.id]['audio_duration']) 
              for video in videos]
    
    # Set up transcripts 
    captions = [audio.CaptionSource(subs_dir + video.item_name(), 
                                    max_time=addtional_field[video.id]['audio_duration'], 
                                    window_size=SEG_LENGTH) 
                for video in videos]
    
    # Set up run opts
    run_opts = {'pipeline_instances_per_node': 32, 'checkpoint_frequency': 5}
    
    # Set up align opts
    align_opts = {'win_size': 300,
                  'seg_length' : 60,
                  'max_misalign' : 10,
                  'num_thread' : 1,
                  'estimate' : True if ESTIMATE else False,
                  'align_dir' : None,
                  'res_path' : None,
#                   'align_dir' : '/app/result/aligned_transcript',
#                   'res_path' : '/app/result/test_align.pkl',
    }
    
############################################################################
# Run alignment (local)
############################################################################
    if LOCAL_RUN:
        if FIRST_RUN:
            result = align_transcript_pipeline(db=db, audio=audios, captions=captions, cache=False, 
                                                       run_opts=run_opts, align_opts=align_opts)
        else:
            result = align_transcript_pipeline2(db=db, audio=audios, captions=captions, cache=False, 
                                                       run_opts=run_opts, align_opts=align_opts)

        # Dump results for local run
    #     align_dir = align_opts['align_dir']
    #     res_path = align_opts['res_path']
    #     if align_dir is None or res_path is None:
    #         exit() 
    #     if os.path.exists(res_path):
    #         res_stats = pickle.load(open(res_path, 'rb'))
    #     else:
    #         res_stats = {}

    #     for idx, res_video in enumerate(result):
    #         video_name, srt_ext, video_id = videos[idx].item_name(), videos[idx].srt_extension, videos[idx].id
    #         if res_video is None:
    #             continue

    #         align_word_list = []
    #         num_word_aligned = 0
    #         num_word_total = 0
    #         res_video_list = res_video.load()
    #         for seg_idx, seg in enumerate(res_video_list):
    #             align_word_list += seg['align_word_list']
    #             num_word_aligned += seg['num_word_aligned']
    #             if 'num_word_total' in seg:
    #                 num_word_total += seg['num_word_total']
    #             else:
    #                 num_word_total += len(seg['align_word_list'])
    #             print(seg_idx, len(seg['align_word_list']))

    #         res_stats[video_id] = {'word_missing': 1 - 1. * num_word_aligned / num_word_total}
    #         print(idx, video_name)
    #         print('num_word_total: ', num_word_total)
    #         print('num_word_aligned: ', num_word_aligned)
    #         print('word_missing by total words: ', 1 - 1. * num_word_aligned / num_word_total)
    #         print('word_missing by total aligned: ', 1 - 1. * num_word_aligned / len(align_word_list))

    #         output_path = os.path.join(align_dir, '{}.{}.srt'.format(video_name, 'word'))
    #         TranscriptAligner.dump_aligned_transcript_byword(align_word_list, output_path)
    #         pickle.dump(res_stats, open(res_path, 'wb'))
    
    
############################################################################
# Run alignment (kubernete)
############################################################################
    else:
        cfg = cluster_config(
            num_workers=50,
            worker=worker_config('n1-standard-32'))

        with make_cluster(cfg, no_delete=True) as db_wrapper:
            db = db_wrapper.db
            if FIRST_RUN:
                align_transcript_pipeline(db=db, audio=audios, captions=captions, cache=False, 
                                                               run_opts=run_opts, align_opts=align_opts)
            else:
                align_transcript_pipeline2(db=db, audio=audios, captions=captions, cache=False, 
                                                               run_opts=run_opts, align_opts=align_opts)