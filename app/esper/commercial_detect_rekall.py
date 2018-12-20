from esper.prelude import *
from rekall.interval_list import IntervalList, Interval
from rekall.logical_predicates import not_pred, or_pred
from rekall.temporal_predicates import overlaps, equal, before, after
import pysrt
import copy
import time

"""
All thresholds 
"""
TRANSCRIPT_DELAY = 0

MIN_BLACKFRAME = 0.99
MIN_BLACKWINDOW = 1

MIN_BLANKWINDOW = 30
MAX_BLANKWINDOW = 270

MIN_LOWERTEXT = 0.5
MIN_LOWERWINDOW = 15
MAX_LOWERWINDOW_GAP = 60

MIN_COMMERCIAL_TIME = 10
MAX_COMMERCIAL_TIME = 270

MAX_MERGE_DURATION = 300
MAX_MERGE_GAP = 30
MIN_COMMERCIAL_GAP = 10
MAX_COMMERCIAL_GAP = 90

MIN_COMMERCIAL_TIME_FINAL = 30
MAX_ISOLATED_BLANK_TIME = 90

"""
Help functions
"""
def fid2second(fid, fps):
    second = 1. * fid / fps
    return second

def time2second(time):
    if len(time) == 3:
        return time[0]*3600 + time[1]*60 + time[2]
    elif len(time) == 4:
        return time[0]*3600 + time[1]*60 + time[2] + time[3] / 1000.0

def second2time(second, sep=','):
    h, m, s, ms = int(second) // 3600, int(second % 3600) // 60, int(second) % 60, int((second - int(second)) * 1000)
    return '{:02d}:{:02d}:{:02d}{:s}{:03d}'.format(h, m, s, sep, ms)

def load_transcript(transcript_path):
    """"
    Load transcript from *.srt file
    """
    transcript = []
    subs = pysrt.open(transcript_path)
    for sub in subs:
        transcript.append((sub.text, time2second(tuple(sub.start)[:4]), time2second(tuple(sub.end)[:4])))
    return transcript

def get_blackframe_list(video, histogram):
    """
    Get all black frames by checking the histogram list 
    """
    pixel_sum = video.height * video.width
    thresh = MIN_BLACKFRAME * pixel_sum
    blackframe_list = []
    for fid, hist in enumerate(histogram):
        if hist[0][0] > MIN_BLACKFRAME and hist[0][1] > MIN_BLACKFRAME and hist[0][2] > MIN_BLACKFRAME:
            blackframe_list.append(fid)
    return IntervalList([
        (fid2second(fid, video.fps),
        fid2second(fid + 1, video.fps),
        0) for fid in blackframe_list])

def get_text_intervals(word, transcript):
    #Todo: remove ones in {}
    return IntervalList([
        (start_sec - TRANSCRIPT_DELAY, end_sec - TRANSCRIPT_DELAY, 0)
        for text, start_sec, end_sec in transcript
        if word in text and '{' not in text
    ]).coalesce()

def get_lowercase_intervals(transcript):
    def is_lower_text(text):
        lower = [c for c in text if c.islower()]
        alpha = [c for c in text if c.isalpha()]
        if len(alpha) == 0:
            return False
        if 1. * len(lower) / len(alpha) > MIN_LOWERTEXT:
            return True
        else:
            return False
        
    return IntervalList([
        (start_sec, end_sec, 0)
        for text, start_sec, end_sec in transcript
        if is_lower_text(text)]) \
            .dilate(1) \
            .coalesce() \
            .dilate(-1) \
            .filter_length(min_length=MIN_LOWERWINDOW)

def detect_commercial_rekall(video, transcript_path, blackframe_list=None, histogram=None, debug=True, verbose=False):
    """
    API for detecting commercial blocks from TV news video using rekall
    
    @video: django query set
    @transcript_path: transcript_path
    @blackframe_list: list of black frames index
    @histogram: list of histogram 16x3 bin for each frame, not used if blackframe_list is provided  
    
    Return: commercial_list (list of tuple((start_fid, start_sec), (end_fid, end_sec)), None if failed)
    """
    
    transcript = load_transcript(transcript_path)
    if blackframe_list is None:
        blackframe_intervallist = get_blackframe_list(histogram)
    else:
        blackframe_intervallist = IntervalList([(fid2second(fid, video.fps),
                                                fid2second(fid + 1, video.fps),
                                                0) for fid in blackframe_list])

    black_windows = blackframe_intervallist \
            .dilate(1. / video.fps) \
            .coalesce() \
            .dilate(-1. / video.fps)
#             .filter_length(min_length=MIN_BLACKWINDOW * 1. / video.fps)

    if verbose:
        print("black window: ({})\n".format(black_windows.size()))
        for idx, win in enumerate(black_windows.get_intervals()):
            print(idx, win)
    
    # get all instances of >>, Announcer:, and  >> Announcer: in transcript
    arrow_text = get_text_intervals(">>", transcript)
    announcer_text = get_text_intervals("Announcer:", transcript)
    arrow_announcer_text = get_text_intervals(">> Announcer:", transcript)

    if verbose:
        print("arrow_text: ({})\n".format(arrow_text.size()), arrow_text)
        print("announcer_text: ({})\n".format(announcer_text.size()))
        print("arrow_announcer_text: ({})\n".format(arrow_announcer_text.size()))
    
    # get an interval for the whole video
    whole_video = IntervalList([(0., video.num_frames/video.fps, 0)])

    # whole video minus black windows to get segments in between black windows
    # then filter out anything that overlaps with ">>" as long as it's not
    #   ">> Announcer:"
    # then coalesce, as long as it doesn't get too long
    def fold_fn(stack, interval):
        if len(stack) == 0:
            stack.append(interval)
        else:
            last = stack.pop()
            if or_pred(overlaps(), after(max_dist=1), arity=2)(interval, last):
                if last.merge(interval).length() > MAX_COMMERCIAL_TIME:
#                     if last.length() > MAX_COMMERCIAL_TIME:
                    stack.append(Interval(
                        last.start, 
                        last.start + MAX_COMMERCIAL_TIME, 
                        last.payload))
#                     else:
#                         stack.append(last)
#                     stack.append(interval)
                else:
                    stack.append(last.merge(interval))
            else:
                stack.append(last)
                stack.append(interval)
        return stack
    
    all_blocks = whole_video.minus(black_windows)
    non_commercial_blocks = all_blocks.filter_against(
        arrow_text.minus(arrow_announcer_text),
        predicate=overlaps()
    )
    commercial_blocks = whole_video.minus(non_commercial_blocks.set_union(black_windows))
    if verbose:
        print("commercial blocks candidates: ({})\n".format(commercial_blocks.size()))
        for idx, win in enumerate(commercial_blocks.get_intervals()):
            print(idx, win)
    
    commercials = commercial_blocks \
        .fold_list(fold_fn, []) \
        .filter_length(min_length = MIN_COMMERCIAL_TIME)
    if verbose:
        print("commercials from blackwindow:\n", commercials)
    if debug:
        commercials_raw = copy.deepcopy(commercials)
    
    # add in lowercase intervals
    lowercase_intervals = get_lowercase_intervals(transcript)
    if verbose:
        print("lowercase intervals:\n", lowercase_intervals)
    commercials = commercials \
            .set_union(lowercase_intervals) \
            .dilate(MIN_COMMERCIAL_GAP / 2) \
            .coalesce() \
            .dilate(-MIN_COMMERCIAL_GAP / 2)
    if verbose:
        print("commercials merge with lowercase:\n", commercials)
    
    
    # get blank intervals
    blank_intervals = whole_video.minus(IntervalList([
            (start_sec, end_sec, 0)
            for text, start_sec, end_sec in transcript
            ]).dilate(1)  \
              .coalesce() \
              .dilate(-1) \
        ).filter_length(min_length=MIN_BLANKWINDOW, max_length=MAX_BLANKWINDOW)
    blank_intervals = IntervalList(blank_intervals.get_intervals()[:-1])
    
    if verbose:
        print("blank intervals:\n", blank_intervals)

    # add in blank intervals, but only if adding in the new intervals doesn't get too long & remove small gaps
    commercials = commercials.set_union(blank_intervals) \
            .dilate(MIN_COMMERCIAL_GAP / 2) \
            .coalesce() \
            .dilate(-MIN_COMMERCIAL_GAP / 2)
        
#     commercials = commercials.merge(blank_intervals,
#             predicate=or_pred(before(max_dist=MAX_MERGE_GAP),
#                 after(max_dist=MAX_MERGE_GAP), arity=2),
#             working_window=MAX_MERGE_GAP
#             ) \
#             .filter_length(max_length=MAX_MERGE_DURATION) \
#             .set_union(commercials) \
#             .dilate(MIN_COMMERCIAL_GAP / 2) \
#             .coalesce() \
#             .dilate(-MIN_COMMERCIAL_GAP / 2)
    if verbose:
        print("commercials merge with blank intervals:\n", commercials)
        
    # merge with small gaps, but only if that doesn't make things too long
#     commercials = commercials \
#             .coalesce() \
#             .filter_length(max_length=MAX_COMMERCIAL_TIME) \
#             .set_union(commercials) \
#             .coalesce()
    

#     # post-process commercials to get rid of gaps, small commercials, and
#     #   islated blocks
#     small_gaps = whole_video \
#             .minus(commercials) \
#             .filter_length(max_length = MAX_COMMERCIAL_GAP) \
#             .filter_against(
#                     arrow_text.filter_against(
#                         announcer_text,
#                         predicate=not_pred(overlaps()),
#                         working_window=1.0
#                     ), predicate=not_pred(overlaps()),
#                     working_window=1.0)
    
#     # merge with small gaps, but only if that doesn't make things too long
#     commercials = commercials \
#             .set_union(small_gaps.dilate(0.1)) \
#             .coalesce() \
#             .filter_length(max_length=MAX_COMMERCIAL_TIME) \
#             .set_union(commercials) \
#             .coalesce()

#     # get isolated commercials
#     not_isolated_commercials = commercials.filter_against(commercials,
#             predicate=or_pred(before(max_dist=MAX_COMMERCIAL_TIME),
#                 after(max_dist=MAX_COMMERCIAL_TIME), arity=2),
#             working_window=MAX_COMMERCIAL_TIME)
#     isolated_commercials = commercials.minus(not_isolated_commercials)
#     commercials_to_delete = isolated_commercials \
#             .filter_length(max_length=MIN_COMMERCIAL_TIME_FINAL) \
#             .set_union(isolated_commercials \
#                 .filter_against(blank_intervals, predicate=equal()) \
#                 .filter_length(max_length=MAX_ISOLATED_BLANK_TIME))
#     commercials = commercials.minus(commercials_to_delete)

    commercial_list = [(i.start, i.end) for i in commercials.get_intervals()]
    
    if debug:
        result = {'black': black_windows.dilate(2),
                  'arrow': arrow_text.dilate(2),
                  'commercials_raw': commercials_raw,
                  'lowercase': lowercase_intervals,
                  'blank': blank_intervals,
                  'commercials': commercials,
                  }
        return result
    else:
        return commercial_list


