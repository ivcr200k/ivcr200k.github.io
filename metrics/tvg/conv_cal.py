import json
import os
from eval_video import find_number
from eval_tvg import iou

def read(path):
    with open(path,'r') as file:
        data = json.load(file)
    return data

#当对话中出现前轮有视频检索且检索结果为0,则将后续轮次全部归0
def deal_conv_result(conv):
    new_conv = []
    flag = True
    for sub_conv in conv:
        new_sub_conv = dict()
        for k,v in sub_conv.items():
            if flag:
                if isinstance(v,bool):
                    if v:
                        new_sub_conv[k] = v
                    else:
                        flag=False
                        new_sub_conv[k] = v
                else:
                    new_sub_conv[k]= v
            else:
                if isinstance(v,bool):
                    new_sub_conv[k] = False
                else:
                    new_sub_conv[k] = 0.
        flag = True
        new_conv.append(new_sub_conv)
    return new_conv

def test_conv_caption(conv,tvg,video):
    flag = 0
    query_set = set()
    for conv in conv:
        for sub_conv in conv:
            query = sub_conv.get('Q')
            query_set.add(query)
            if sub_conv.get('gt_se') is not None:
                if tvg.get(query) is None:
                    flag +=1
                    # print(query)
            elif sub_conv.get('top10_list') is not None:
                if video.get(query) is None:
                    # print(query)
                    flag+=1
    print(len(query_set))
    return flag

def format_print(cal_conv_result):
    for k,v in cal_conv_result.items():
        print(k)
        print(v)

def val_result(new_result,old_result):
    hell=True
    for item in old_result:
        if len(item)==6:
            for k,v in item.items():
                if v==False:
                    print(item)
                    hell = False
                    break
        if hell is False:
            break
    flag=True
    for item in new_result:
        if len(item)==6:
            for k,v in item.items():
                if v==False:
                    print(item)
                    flag = False
                    break
        if flag is False:
            break
# def print_result_lunci(conv,f):


if __name__ == "__main__":
    conv_path = "./data_processing/IVCR-200k/test_data/test_conv_data_final.json"
    video_pre_path = "./output/test_for_final_ivcr_video_retrieval/IVCR_train_epoch10_2w_accgrad16_vfrm12_changeloss_001--2024_05_28_11_01/xpool_blip2_cp7_final_recall10/fmt_IVCR_test_f96_result.json"
    tvg_pre_path = "./output/test_for_final_ivcr_tvg/IVCR_train_epoch10_2w_accgrad16_vfrm12_changeloss_001--2024_05_28_11_01/xpool_blip2_cp7_final_top1/fmt_IVCR_test_f96_result.json"
    video_all_path = "./data_processing/IVCR-200k/test_data/xpool-blip2/test_video_dup_data_add_top10_1283.json"
    tvg_all_path = "./data_processing/IVCR-200k/test_data/xpool-blip2/all_test_tvg.json"
    tvg_pre_data = read(tvg_pre_path)
    video_pre_path = read(video_pre_path)
    video_all_path = read(video_all_path)
    tvg_all_data = read(tvg_all_path)
    tvg_query2se = dict()
    video_query2index = dict()
    for k,v in video_pre_path.items():
        query = v.get('query')
        pre_id = find_number(v.get('video_id'))
        video_query2index[query] = pre_id
    
    for k,v in tvg_pre_data.items():
        query = v.get('query')
        if len(v.get('timestamp'))==1:
            pre_se= v.get('timestamp')[0]
        else:
            pre_se = [0.,0.]
        tvg_query2se[query] = pre_se

    for item in tvg_all_data:
        query = item.get('Q')
        if query not in tvg_query2se:
            tvg_query2se[query] = item.get('gt_se')

    for item in video_all_path:
        query = item.get('Q')
        if query not in video_query2index:
            # print(item.get('video_top10_list'))
            video_query2index[query] = item.get('video_top10_list')

    result = []
    conv_data = read(conv_path)
    for conv in conv_data:
        sub_result = dict()
        for i,sub_conv in enumerate(conv):
            if sub_conv.get('gt_se') is not None:
                if not sub_conv.get('gt_se'):
                    sub_result[i] = 0.
                else:
                    predict_se = tvg_query2se[sub_conv.get('Q')]
                    pre_iou = iou(predict_se, sub_conv.get('gt_se'))
                    sub_result[i] = pre_iou
            elif sub_conv.get('top10_list') is not None:
                if sub_conv.get('top10_list') == 0:
                    sub_result[i] = False
                else:
                    gt_video = sub_conv.get('video_path')
                    top10 = sub_conv.get('top10_list')
                    gt_index = top10.index(gt_video)+1
                    pre_video = top10[video_query2index[sub_conv.get('Q')]-1]
                    # pre_index = video_query2index[sub_conv.get('Q')]
                    if gt_video == pre_video:
                        sub_result[i] = True
                    else:
                        sub_result[i] = False
        result.append(sub_result)
    
    new_result = deal_conv_result(result)
    # new_result =result
    #验证result的修改是否有效
    val_result(new_result,result)
    # new_result = result
    max = 0
    for sub_result in new_result:
        if len(sub_result) > max:
            max = len(sub_result)

    cal_conv_result = dict()
    for i in range(6):
        tvg_conv_result = dict(iou3=0,iou5=0,iou7=0)
        video_conv_result = 0
        total_tvg = 0
        total_video = 0
        for sub_result in new_result:
            if len(sub_result)>=i+1:
            # if len(sub_result) == 5:
                get_result = sub_result[i]
                if isinstance(get_result, bool):
                    total_video+=1
                    if get_result:
                        video_conv_result+=1
                else:
                    total_tvg+=1
                    if get_result>=0.3:
                        tvg_conv_result['iou3'] +=1
                    if get_result >= 0.5:
                        tvg_conv_result['iou5'] +=1
                    if get_result >= 0.7:
                        tvg_conv_result['iou7'] +=1
        print(f"第{i+1}轮次的视频片段检索总数目{total_tvg}")
        print(f"第{i+1}轮次的视频检索总数目{total_video}")
        tvg_conv_result['iou3'] = tvg_conv_result['iou3']*100/total_tvg
        tvg_conv_result['iou5'] = tvg_conv_result['iou5']*100/total_tvg
        tvg_conv_result['iou7'] = tvg_conv_result['iou7']*100/total_tvg
        if total_video==0:
            total_video=1
        video_conv_result = video_conv_result*100/total_video
        cal_conv_result[i+1] =  dict(tvg3=tvg_conv_result['iou3'],
                                    tvg5 = tvg_conv_result['iou5'],
                                    tvg7 = tvg_conv_result['iou7'],
                                    video = video_conv_result)
    
    format_print(cal_conv_result)




