import tempfile
import re
import pickle
import multiprocessing as mp
import os
import codecs
import pysrt

# ============== Basic help functions ==============  

def fid2second(fid, fps):
    second = 1. * fid / fps
    return second


def second2time(second, sep=','):
    h, m, s, ms = int(second) // 3600, int(second % 3600) // 60, int(second) % 60, int((second - int(second)) * 1000)
    return '{:02d}:{:02d}:{:02d}{:s}{:03d}'.format(h, m, s, sep, ms)


def time2second(time):
    if len(time) == 3:
        return time[0]*3600 + time[1]*60 + time[2]
    else:
        return time[0]*3600 + time[1]*60 + time[2] + time[3] / 1000.0
    

def get_detail_from_video_name(video_name):
    split = video_name.split('_')
    date = get_date_from_string(split[1], split[2])
    station = split[0]
    if station == 'CNN':
        show = video_name[21:]
    elif station == 'FOXNEWS':
        show = video_name[25:]
    elif station == 'MSNBC':
        show = video_name[23:]
    elif station == 'CNNW':
        show = video_name[22:]
        station = 'CNN'
    elif station == 'FOXNEWSW':
        show = video_name[26:]
        station = 'FOXNEWS'
    elif station == 'MSNBCW':
        show = video_name[24:]
        station = 'MSNBC'
    return date, station, show


def get_date_from_string(str1, str2):
    return (int(str1[:4]), int(str1[4:6]), int(str1[6:]), int(str2[:2]), int(str2[2:4]), int(str2[4:]))


def compare_date(date1, date2):
    if date1[0] < date2[0]:
        return -1
    elif date1[0] > date2[0]:
        return 1
    elif date1[1] < date2[1]:
        return -1
    elif date1[1] > date2[1]:
        return 1
    elif date1[2] < date2[2]:
        return -1
    elif date1[2] > date2[2]:
        return 1
    elif date1[2] == date2[2]:
        return 0

    
def par_for_process(function, param_list, num_workers=32):
    num_jobs = len(param_list)
    print("Total number of %d jobs" % num_jobs)
    if num_jobs == 0:
        return 
    if num_jobs <= num_workers:
        num_workers = num_jobs
        num_jobs_p = 1
    else:
        num_jobs_p = math.ceil(1. * num_jobs / num_workers)
    print("{} workers and {} jobs per worker".format(num_workers, num_jobs_p))
    
    process_list = []
    for i in range(num_workers):
        if i != num_workers - 1:
            param_list_p = param_list[i*num_jobs_p : (i+1)*num_jobs_p]
        else:
            param_list_p = param_list[i*num_jobs_p : ]
        p = mp.Process(target=function, args=(param_list_p,))
        process_list.append(p)

    for p in process_list:
        p.start()
#     for p in process_list:
#         p.join()
    

#=========== Compute audio_duration, bad_transcript, missing_transcript ===========

transcript_dir = '/app/data/subs/subs_kdd'

def get_audio_length(video):
    video_name = video.item_name()
    url = video.url()
    log_path = tempfile.NamedTemporaryFile(suffix='.txt').name
    cmd = 'ffprobe -show_streams -i ' + \
        '\"' + url + '\"' + ' > ' + log_path
    os.system(cmd)
    log = open(log_path, 'r')
    format_str = log.read()
    log.close()
    find = re.findall(r'\nduration=(.*)', format_str)
    if len(find) == 1:
        return float(re.findall(r'\nduration=(.*)', format_str)[0])
    else:
        return float(re.findall(r'\nduration=(.*)', format_str)[1])


def get_fps(video):
    video_name = video.item_name()
    url = video.url()
    log_path = tempfile.NamedTemporaryFile(suffix='.txt').name
    cmd = 'ffprobe -show_streams -i ' + \
        '\"' + url + '\"' + ' > ' + log_path
    os.system(cmd)
    log = open(log_path, 'r')
    format_str = log.read()
    log.close()
    fps_str = re.findall(r'\navg_frame_rate=(.*)', format_str)[0].split('/')
    return float(fps_str[0]) / float(fps_str[1])


def check_transcript_encoding(video):
    transcript_path = os.path.join(transcript_dir, video.item_name())
    try:
        file = codecs.open(transcript_path, encoding='utf-8', errors='strict')
        for line in file:
            pass
    except UnicodeDecodeError:
        return False
    return True


def check_transcript_complete(video):
    video_name = video.item_name()
    transcript_path = os.path.join(transcript_dir, video.item_name())
    transcript = []
    subs = pysrt.open(transcript_path)
    text_length = 0
    for sub in subs:
        transcript.append((sub.text, time2second(tuple(sub.start)[:3]), time2second(tuple(sub.end)[:3])))
        text_length += transcript[-1][2] - transcript[-1][1]
    video_length = video.num_frames / video.fps
    text_ratio = text_length / video_length
    return text_ratio > 0.3


def get_video_field(video):
    audio_length = get_audio_length(video)
    
    good_transcript = os.path.exists(os.path.join(transcript_dir, video.item_name()))
#     print(video.item_name())
    if good_transcript:    
        good_transcript = check_transcript_encoding(video)
        if good_transcript:
            good_transcript = check_transcript_complete(video)
    return audio_length, good_transcript 


def get_video_field_t(videos):
    pkl_path = tempfile.NamedTemporaryFile(suffix='.pkl').name
    print(pkl_path)
    result = {}
    for idx, video in enumerate(videos):
        audio_duration, valid_transcript = get_video_field(video)
        result[video.id] = {'audio_duration': audio_duration, 'valid_transcript': valid_transcript}
        if idx % 10 == 0:
            print(idx)
            pickle.dump(result, open(pkl_path, 'wb'))
    pickle.dump(result, open(pkl_path, 'wb'))
    return pkl_path
