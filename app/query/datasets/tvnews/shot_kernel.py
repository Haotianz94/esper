from scannerpy import Kernel
from scannerpy.stdlib import parsers
import numpy as np
from scipy.spatial import distance
from unionfind import unionfind
from timeit import default_timer as now
import pickle
import traceback

WINDOW_SIZE = 500
GROUP_THRESHOLD = 10
STD_DEV_FACTOR = 1
MAGNITUDE_THRESHOLD = 5000

class ShotDetectionKernel(Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        self.hists = []
        pass

    def close(self):
        pass

    def reset(self):
        del self.hists
        self.hists = []

    def execute(self, input_columns):
        self.hists.extend([parsers.histograms([buf], self.protobufs) for buf in input_columns[0]])
        assert(len(self.hists) > 0)

        try:
            print('Processing {} hists'.format(len(self.hists)))
            diffs = np.array([
                np.mean([distance.chebyshev(self.hists[i - 1][j], self.hists[i][j]) for j in range(3)])
                for i in range(1, len(self.hists))
            ])
            diffs = np.insert(diffs, 0, 0)
            n = len(diffs)

            # Do simple outlier detection to find boundaries between shots
            boundaries = []
            for i in range(1, n):
                window = diffs[max(i - WINDOW_SIZE / 2, 0):min(i + WINDOW_SIZE / 2, n)]
                if diffs[i] > MAGNITUDE_THRESHOLD and \
                   diffs[i] - np.mean(window) > STD_DEV_FACTOR * np.std(window):
                    boundaries.append(i)

            u = unionfind(len(boundaries))
            for i, bi in enumerate(boundaries):
                for j, bj in enumerate(boundaries):
                    if abs(bi - bj) < GROUP_THRESHOLD:
                        u.unite(i, j)
                        break

            grouped_boundaries = [boundaries[g[len(g) / 2]] for g in u.groups()]

            black_frames = []
            count = float(3 * sum(self.hists[0][0]))
            for i, h in enumerate(self.hists):
                if (h[0][0] + h[1][0] + h[2][0]) / count > 0.9:
                    black_frames.append(i)

            print('Done!')
            return [['_' for _ in range(len(input_columns[0]) - 1)] + \
                    [pickle.dumps((grouped_boundaries, black_frames))]]
        except Exception:
            traceback.print_exc()
            return [['_' for _ in range(len(input_columns[0]))]]



KERNEL = ShotDetectionKernel
