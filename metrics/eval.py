import json
import os
import time
import sys
import argparse
import pdb
import logging
import re
def extract_time(paragraph):
    prompt = 'A specific example is : 20.8 - 30.0 seconds'.lower()
    paragraph = paragraph.lower()
    paragraph = paragraph.replace(prompt, '')

    pattern = r'\.{2,}'
    paragraph = re.sub(pattern, '.', paragraph)

    # Split text into sentences based on common delimiters
    sentences = re.split(r'[!?\n]', paragraph)
    
    candidates = []
    for sentence in sentences:
        # If sentence contains one of the keywords
        # if any(keyword in sentence for keyword in keywords):
        candidates.append(sentence)
            
    timestamps = []
    # Check for The given query happens in m - n (seconds)
    patterns = [
        r"(\d+\.*\d*)s?\s*-\s*(\d+\.*\d*)s?"
    ]
    
    for time_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]  

    if len(sentences) == 0:
        return []
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r'\b(\d+\.\d+\b|\b\d+)\b') # time formats (e.g., 18, 18.5)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                time_in_sec = float(time[0])
                times.append(time_in_sec)
        times = times[:len(times)//2*2]
        timestamps = [(times[i], times[i+1]) for i in range(0, len(times), 2)]
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r'\b((\d{1,2}:\d{2}:\d{2}))\b') # time formats (e.g., 18:00, 00:18:05)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                t = time[0]
            else:
                continue
            # If time is in HH:MM:SS format, convert to seconds
            if t.count(':') == 2:
                h, m, s = map(int, t.split(':'))
                time_in_sec = h * 3600 + m * 60 + s
            elif t.count(':') == 1:
                m, s = map(int, t.split(':'))
                time_in_sec = m * 60 + s
            times.append(time_in_sec)
        times = times[:len(times)//2*2]
        timestamps = [(times[i], times[i+1]) for i in range(0, len(times), 2)]
    results = []
    for (start, end) in timestamps:
        if end > start:
            results.append([start, end])
        else:
            results.append([end, start])
    if len(results) > 1:
        results = results[:1]
    if results == []:
        results = [[0,0]]
    return results

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


def extract_video_numbers_first_only(text):
    """
    只返回第一个匹配的数字
    
    Args:
        text (str): 输入的字符串
    
    Returns:
        int or None: 第一个找到的数字，如果没有找到则返回None
    """
    pattern = r'(?i)\bvideo\s+(10|[0-9])\b'
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    return None

def extract_time_ranges(text):
    """
    从文本中提取xxs-yys格式的时间段，返回[开始时间, 结束时间]的列表
    
    Args:
        text (str): 输入的文本字符串
    
    Returns:
        list: 包含[start_time, end_time]的列表，时间为浮点数（秒）
    
    Examples:
        >>> extract_time_ranges("Query content found in video 1 at 5s-10s.")
        [5.0, 10.0]
        >>> extract_time_ranges("Content at 1.5s-3.2s and 10s-15.5s")
        [1.5, 3.2, 10.0, 15.5]
    """
    # 正则表达式匹配 数字s-数字s 的模式
    # \d+(?:\.\d+)? 匹配整数或浮点数
    pattern = r'(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s'
    
    # 找到所有匹配的时间段
    matches = re.findall(pattern, text)
    
    # 将匹配结果转换为浮点数并展开为一维列表
    result = []
    for start_str, end_str in matches:
        result.extend([float(start_str), float(end_str)])
    
    return result


def cal_reslut(pre_data,gt_data,all_data):
    reslut = dict()
    count = 0
    sum = 0
    tongji = dict()
    jieguo = dict()
    for i, gt in enumerate(gt_data):
        user_q = gt[-2]
        assert user_q['role'] == 'user'
        assis_a = gt[-1]
        assert assis_a['role'] == 'assistant'
        if user_q['content'].count("VIDEOTOKEN") == 10:
            if 'vr' not in reslut:
                reslut['vr'] = dict(right=0,all=0)
            else:
                gt_index = extract_video_numbers_first_only(assis_a['content'])
                pre = pre_data[i]['output']
                pre_index = extract_video_numbers_first_only(pre)
                if pre_index not in [1,2,3,4,5,6,7,8,9,10]:
                    count+=1
                if gt_index not in tongji:
                    tongji[gt_index] = 0
                if gt_index not in jieguo:
                    jieguo[gt_index] = {'right':0,'error':0}
                tongji[gt_index] += 1
                if gt_index == pre_index:
                    jieguo[gt_index]['right']+=1
                    reslut['vr']['right'] += 1
                else:
                    jieguo[gt_index]['error'] += 1
        else:
            if 'vmr' not in reslut:
                reslut['vmr'] = dict(r3=0,r5=0,r7=0,all=0)
            else:
                gt_time = extract_time_ranges(assis_a['content'])
                pre = pre_data[i]['output']
                pre_time = extract_time(pre)
                if pre_time[0] == [0,0]:
                    sum += 1
                    print(f"sum:{sum}")
                cal_iou = iou(gt_time,pre_time[0])
                if cal_iou>=0.3:
                    reslut['vmr']['r3'] += 1
                if cal_iou>=0.5:
                    reslut['vmr']['r5'] += 1
                if cal_iou>=0.7:
                    reslut['vmr']['r7'] += 1
    for sample in all_data:
        if sample['split'] == 'test' and sample['type'] != [0]:
            convs = sample['conversations']
            for conv in convs:
                if conv['from'] == 'gpt' and conv['gt_se'] == [-1,-1]:
                    reslut['vr']['all'] += 1
                elif conv['from'] == 'gpt' and conv['gt_se'] != [-1, -1]:
                    reslut['vmr']['all'] += 1
    reslut['vr']['right'] = reslut['vr']['right']/reslut['vr']['all']
    reslut['vmr']['r3'] = reslut['vmr']['r3']/reslut['vmr']['all']
    reslut['vmr']['r5'] = reslut['vmr']['r5']/reslut['vmr']['all']
    reslut['vmr']['r7'] = reslut['vmr']['r7']/reslut['vmr']['all']
    return reslut

def find_sentence(user_q, pre_data):
    for pre in pre_data:
        user_q1 = pre['user_q']
        if user_q == user_q1:
            return pre['output']
    return None

def cal_multi_reslut(pre_data,all_data):
    result = dict()
    sum = 0
    for sample in all_data:
        if sample['split'] == 'test' and sample['type'] != [0]:
            convs = sample['conversations']
            for i,conv in enumerate(convs):
                if conv['from'] == 'human':
                    if int((i+2)/2) not in result:
                        result[int((i+2)/2)] = dict(vr_right = 0,vr_all= 0,vmr_r3 = 0,vmr_r5=0,vmr_r7=0,vmr_all=0)
                    user_q = conv['value']
                    pre_result = find_sentence(user_q,pre_data)
                    if pre_result is not None:
                        if convs[i+1]['gt_se'] == [-1,-1]:
                            gt_index = extract_video_numbers_first_only(convs[i+1]['value'])
                            pre_index = extract_video_numbers_first_only(pre_result)
                            if gt_index == pre_index:
                                result[int((i+2)/2)]['vr_right'] += 1
                                result[int((i+2)/2)]['vr_all'] += 1
                            else:
                                result[int((i+2)/2)]['vr_all'] += 1
                        elif convs[i+1]['gt_se'] != [-1,-1]:
                            gt_time = extract_time_ranges(convs[i+1]['value'])
                            pre_time = extract_time(pre_result)
                            cal_iou = iou(gt_time,pre_time[0])
                            if cal_iou>=0.3:
                                result[int((i+2)/2)]['vmr_r3'] += 1
                            if cal_iou>=0.5:
                                result[int((i+2)/2)]['vmr_r5'] += 1
                            if cal_iou>=0.7:
                                result[int((i+2)/2)]['vmr_r7'] += 1
                            result[int((i+2)/2)]['vmr_all'] += 1
                    else:
                        if convs[i+1]['gt_se'] == [-1,-1]:
                            result[int((i+2)/2)]['vr_all'] += 1
                        elif convs[i+1]['gt_se'] != [-1,-1]:
                            result[int((i+2)/2)]['vmr_all'] += 1
             
    return result
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="/home/longshaohua/IVCR_Result/Eval_Result/2025_06_16_21_21/2/result_new.json")
    parser.add_argument('--gt_file', type=str, default='/home/longshaohua/Dataset/use_xpool_top10/IVCR_no_type0_no_zero_dialogues_test_up7_new.json')
    parser.add_argument('--all_file',type=str,default='/home/longshaohua/Dataset/use_xpool_top10/IVCR-200K_new.json')
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
    pre_data = read_json(args.pred_file)
    gt_data = read_json(args.gt_file)
    all_data = read_json(args.all_file)
    # print(len(pre_data))
    # print(len(gt_data))
    multi_result = cal_multi_reslut(pre_data, all_data)
    print(f"多轮检索结果")
    print(f"Turns     R@1    R@1(Iou=0.3)    R@1(Iou=0.5)    R@1(Iou=0.7)")
    for k,v in multi_result.items():
        if k<=7:
            print(f"{k}:   {v['vr_right']/v['vr_all']*100}    {v['vmr_r3']/v['vmr_all']*100}    {v['vmr_r5']/v['vmr_all']*100}    {v['vmr_r7']/v['vmr_all']*100}\n")
    result= cal_reslut(pre_data,gt_data,all_data)
    print(f"测试结果")
    print(f"R@1: {result['vr']['right']*100}\n")
    print(f"IOU 0.3: {result['vmr']['r3']*100}\nIOU 0.5: {result['vmr']['r5']*100}\nIOU 0.7: {result['vmr']['r7']*100}")