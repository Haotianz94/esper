import math
import sys
import os
from multiprocessing import Pool
from pathlib import Path

from esper.prelude import par_for
from query.models import Video

import captions.util as caption_util
from captions import Documents, Lexicon, CaptionIndex, MetadataIndex


INDEX_DIR = '/app/data/index10'
DOCUMENTS_PATH = os.path.join(INDEX_DIR, 'docs.list')
LEXICON_PATH = os.path.join(INDEX_DIR, 'words.lex')
INDEX_PATH = os.path.join(INDEX_DIR, 'index.bin')
METADATA_PATH = os.path.join(INDEX_DIR, 'meta.bin')

print('Loading the document list and lexicon', file=sys.stderr)
try:
    DOCUMENTS
    LEXICON
    INDEX
except NameError:
    DOCUMENTS = Documents.load(DOCUMENTS_PATH)
    LEXICON = Lexicon.load(LEXICON_PATH)
    INDEX = CaptionIndex(INDEX_PATH, LEXICON, DOCUMENTS)

    
def is_word_in_lexicon(word):
    return word in LEXICON

    
def _get_video_name(p):
    """Only the filename without exts"""
    return Path(p).name.split('.')[0]


def _init_doc_id_to_vid_id():
    video_name_to_id = {_get_video_name(v.path) : v.id for v in Video.objects.all()}
    doc_id_to_vid_id = {}
    num_docs_with_no_videos = 0
    for d in DOCUMENTS:
        video_name = _get_video_name(d.name)
        video_id = video_name_to_id.get(video_name, None)
        if video_id is not None:
            doc_id_to_vid_id[d.id] = video_id
        else:
            num_docs_with_no_videos += 1
    print('Matched {} documents to videos'.format(len(doc_id_to_vid_id)), file=sys.stderr)
    print('{} documents have no videos'.format(num_docs_with_no_videos), file=sys.stderr)
    print('{} videos have no documents'.format(len(video_name_to_id) - len(doc_id_to_vid_id)),
          file=sys.stderr)
    return doc_id_to_vid_id


DOCUMENT_ID_TO_VIDEO_ID = _init_doc_id_to_vid_id()
VIDEO_ID_TO_DOCUMENT_ID = {v: k for k, v in DOCUMENT_ID_TO_VIDEO_ID.items()}


def get_document(video_id: int):
    doc_id = VIDEO_ID_TO_DOCUMENT_ID[video_id]
    return DOCUMENTS[doc_id]


def convert_doc_ids_to_video_ids(results):
    def wrapper(document_results):
        for d in document_results:
            video_id = DOCUMENT_ID_TO_VIDEO_ID.get(d.id, None)
            if video_id is not None:
                yield d._replace(id=video_id)
    return wrapper(results)


def convert_video_ids_to_doc_ids(vid_ids, verbose=False):
    if vid_ids is None:
        return None
    else:
        doc_ids = []
        for v in vid_ids:
            d = VIDEO_ID_TO_DOCUMENT_ID.get(v, None)
            if d is not None:
                doc_ids.append(d)
            elif verbose:
                print('Document not found for video id={}'.format(v))
        assert len(doc_ids) > 0
        return doc_ids


def topic_search(phrases, window_size=60, video_ids=None):
    if not isinstance(phrases, list):
        raise TypeError('phrases should be a list of phrases/n-grams')
    documents = convert_video_ids_to_doc_ids(video_ids)
    return convert_doc_ids_to_video_ids(
        caption_util.topic_search(
            phrases, INDEX, window_size, documents))


def phrase_search(query, video_ids=None):
    documents = convert_video_ids_to_doc_ids(video_ids)
    return convert_doc_ids_to_video_ids(
        INDEX.search(query, documents=documents))


# Set before forking, this is a hack
LOWER_CASE_ALPHA_IDS = None


def _get_lowercase_segments(video_id, dilate=1, verbose=False):
    doc_id = VIDEO_ID_TO_DOCUMENT_ID.get(video_id, None)
    if doc_id is None:
        if verbose:
            print('No document for video id: {}'.format(video_id), file=sys.stderr)
        return []

    def has_lowercase(posting):
        tokens = INDEX.tokens(doc_id, posting.idx, posting.len)
        for t in tokens:
            if t in LOWER_CASE_ALPHA_IDS:
                return True
        return False

    lowercase_segments = []
    curr_interval = None
    for interval in INDEX.intervals(doc_id, 0, 2 ** 31):
        if has_lowercase(interval):
            if curr_interval is None:
                curr_interval = (interval.start - dilate, interval.end + dilate)
            else:
                curr_start, curr_end = curr_interval
                if min(interval.end + dilate, curr_end) - max(interval.start - dilate, curr_start) > 0:
                    curr_interval = (
                        min(interval.start - dilate, curr_start), 
                        max(interval.end + dilate, curr_end)
                    )
                else:
                    lowercase_segments.append(curr_interval)
                    curr_interval = (interval.start - dilate, interval.end + dilate)
    if curr_interval is not None:
        lowercase_segments.append(curr_interval)
    return lowercase_segments


def get_lowercase_segments(video_ids=None):
    if video_ids is None:
        video_ids = [v.id for v in Video.objects.filter(threeyears_dataset=True)]
    elif not isinstance(video_ids, list):
        video_ids = list(video_ids)

    def has_lower_alpha(word):
        for c in word:
            if c.isalpha() and c.islower():
                return True
        return False

    lowercase_alpha_ids = {w.id for w in LEXICON if has_lower_alpha(w.token)}
    global LOWER_CASE_ALPHA_IDS
    LOWER_CASE_ALPHA_IDS = lowercase_alpha_ids
    with Pool(os.cpu_count()) as pool:
        results = pool.map(_get_lowercase_segments, video_ids)
    return zip(video_ids, results)
