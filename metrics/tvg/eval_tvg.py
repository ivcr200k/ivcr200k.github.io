import json
import os
import time
import sys
import argparse
import pdb
import logging
def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def iou(A, B):
    if len(B) == 0:
        return 0
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def toSec(timeStr):
    t = time.strptime(timeStr, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def captiondata_modify(steps):
    modify_data = {}
    for i, step in enumerate(steps[0]):
        for key in step["step"].keys():
            name = step["step"][key]["query_idx"]
            modify_data[name] = [[step['step'][key]["startime"], step['step'][key]["endtime"]]]
        
    return modify_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="/data/longshaohua/TimeChat/output/test_for_final_ivcr_tvg/IVCR_train_epoch10_2w_accgrad16_vfrm12_changeloss_001--2024_05_28_11_01/xpool_clip_cp7_final_top1/fmt_IVCR_test_f96_result.json")
    parser.add_argument('--gt_file', type=str, default='/data/longshaohua/TimeChat/data_processing/IVCR-200k/test_data/xpool-clip/test_tvg.json')
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()
    '''
    {
        "query_idx": [start_time, end_time],
        ...
    }
    '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(args.pred_file)
    logging.info(args.gt_file)
    answer = read_json(args.gt_file)
    # answer = answer["annotations"]
    gt_timestamps = {}
    for jterm in answer:
        gt_timestamps[jterm["id"]] = jterm['gt_se'] #jterm["timestamp"]
        
    submission = read_json(args.pred_file)
    pred_timestamps = {}
    count = 0
    #use count as flag
    for qid, jterm in submission.items():
        if jterm.get('intent') == -1:
            count+=1
            pred_timestamps[int(qid)] = [[]]
            continue
        # pred_timestamps[int(qid)] = jterm['timestamp']
        if 'temporal video grounding' in jterm.get('intent') or 'video moment' in jterm.get('intent'):
            pred_timestamps[int(qid)] = jterm['timestamp']
        else:
            count+=1
            pred_timestamps[int(qid)] = [[]]
    print(count)
    if args.sample:
        new = {}
        for qid in pred_timestamps.keys():
            new[qid] = gt_timestamps[qid]
        gt_timestamps = new
    all_data_len  = read_json('/data/longshaohua/TimeChat/data_processing/IVCR-200k/test_data/test_tvg_dup_new_caption_data_1153.json')
    num = len(all_data_len)
    print(f"# pred video timestamps {len(pred_timestamps)}; # gt video timestamps {len(gt_timestamps)}")
    print(num)
    assert len(gt_timestamps) == len(pred_timestamps)
    Result = {0.3:0, 0.5:0, 0.7:0}
    for c_iou in [0.3, 0.5, 0.7]:
        for key in gt_timestamps.keys():
            if len(pred_timestamps[key]) < 1:
                continue
            if(iou(gt_timestamps[key], pred_timestamps[key][0]) >= c_iou):
                Result[c_iou] = Result[c_iou] + 1
    print("IOU 0.3: {0}\nIOU 0.5: {1}\nIOU 0.7: {2}".format(Result[0.3]*100/num, Result[0.5]*100/num, Result[0.7]*100/num))