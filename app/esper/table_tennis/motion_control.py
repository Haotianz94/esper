def group_motion(interval, fid2openpose, step_size=5, fps=25):
    _, sfid, efid, _ = interval
    x_list = [fid2openpose_A[fid]._format_keypoints()[Pose.Neck, 0] for fid in range(sfid, efid) if fid in fid2openpose]
    x_smooth = savgol_filter(x_list, fps, 2)
    motion_dict = {'left': [], 'right': [], 'still': []}
   
    for i in range(0, efid - sfid, step_size):
        for j in range(i + fps, efid - sfid, step_size):
            x_seg = x_smooth[i : j]
            xmin, xmax = min(x_seg), max(x_seg)
            argmin = np.argmin(x_seg)
            argmax = np.argmax(x_seg)
            if xmax - xmin < 0.01:
                avgx = np.average(x_seg)
                motion_dict['still'] += [{'start_x': avgx, 'end_x': avgx, 'start_fid': i+sfid, 'end_fid' : j+sfid, 
                                          'duration': (j-i)/video.fps}]
            elif (xmax - xmin) > 0.05 and argmin == 0 and argmax == len(x_seg) - 1:
                # decide whether the slope is smooth
                valid = True
                for k in range(0, len(x_seg) - fps, fps):
                    if x_seg[k+fps] - x_seg[k] < 0.01:
                        valid = False
                        break
                if valid:
                    motion_dict['right'] += [{'start_x': xmin, 'end_x': xmax, 'start_fid': i+sfid, 'end_fid' : j+sfid, 
                                          'duration': (j-i)/video.fps}]
            elif (xmax - xmin) > 0.04 and argmax == 0 and argmin == len(x_seg) - 1:
                # decide whether the slope is smooth
                valid = True
                for k in range(0, len(x_seg) - fps, fps):
                    if x_seg[k] - x_seg[k+fps] < 0.01:
                        valid = False
                        break
                if valid:
                    motion_dict['left'] += [{'start_x': xmax, 'end_x': xmin, 'start_fid': i+sfid, 'end_fid' : j+sfid, 
                                          'duration': (j-i)/video.fps}] 
    def merge_overlap_seg(motion_list):
        if len(motion_list) == 0:
            return []
        motion_list.sort(key = lambda x : x['start_fid'])
        motion_list_merge = []
        old_sfid = motion_list[0]['start_fid']
        for seg in motion_list:
            if seg['start_fid'] == old_sfid:
                longest_seg = seg
            elif old_sfid != -1:
                motion_list_merge.append(longest_seg)
                old_sfid = -1
            if seg['start_fid'] > longest_seg['end_fid']:
                old_sfid = seg['start_fid']
                longest_seg = seg
        if len(motion_list_merge) == 0:
            motion_list_merge.append(longest_seg)
        elif longest_seg['start_fid'] != motion_list_merge[-1]['start_fid']:
            motion_list_merge.append(longest_seg)
        return motion_list_merge
    
    motion_dict_filter = {'left': merge_overlap_seg(motion_dict['left']), 'right': merge_overlap_seg(motion_dict['right']), 
                     'still': merge_overlap_seg(motion_dict['still'])}

    return motion_dict_filter