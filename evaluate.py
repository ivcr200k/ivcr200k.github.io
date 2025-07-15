import copy
import os
import torch
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
# from ivcr.common.logger import setup_logger
from utils.logger import setup_logger
from ivcr.common.config import Config
from ivcr.common.dist_utils import get_rank
from ivcr.common.registry import registry
from ivcr.conversation.conversation_video_batch import Chat, Conversation, default_conversation, SeparatorStyle, \
    conv_llava_llama_2
import decord
import einops
import ivcr.tasks as tasks
decord.bridge.set_bridge('torch')
import logging
from torchvision.transforms.functional import InterpolationMode
import sys
from transformers import AutoTokenizer
from torchvision import transforms
import pdb
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random

random.seed(42)
from utils.format_tvg import format_tvg_output


def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    with open(anno_path, 'r') as f:
        # data = json.load(f)["annotations"]
        data = json.load(f)

    if args.debug:
        data = data[:10]
    return data


def save_result(output_dir, results):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(results, f,indent=4)
    return


def format_intent(gcap):
    if '.' in gcap:
        start_index = gcap.index('.') + 1
        first_part = gcap[:start_index]
        sub_gcap = gcap[start_index:]
        if '.' in sub_gcap:
            second_index = sub_gcap.index('.') + 1
            second_part = sub_gcap[:second_index]
            return first_part,second_part
        else:
            return -1,-1
    else:
        return -1,-1

def format_video(datas):
    fmt_datas = {}
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        intent,video_id = format_intent(gcap=gcap)
        if intent==-1 and video_id == -1:
            continue
        else:
            fmt_datas[i] = {"video_id": video_id,"intent":intent, "query": query, "vid": vid}
    return fmt_datas

def format_tvg(datas):
    fmt_datas = {}
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = int(jterm["id"])
        timestamps = format_tvg_output(gcap)
        intent,second_part = format_intent(gcap=gcap)
        if intent == -1:
            continue
        if len(timestamps) == 0:
            cnt += 1
            print(vid, query + "\n", gcap, "\n")
        fmt_datas[qid] = {"timestamp": timestamps,"intent":intent, "query": query, "vid": vid}
    print(f'parse failed number: {cnt}')
    return fmt_datas


def generate(chat, gr_videos, user_messages, num_beams, temperature, top_p, n_frms,task, chat_states=None, img_lists=None):
    N = len(user_messages)
    if chat_states is None:
        chat_states = []
        for i in range(N):
            chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            chat_states.append(chat_state)
    if img_lists is None:
        if task == 'format_video':
            img_lists = [[] for i in range(N)]
            llm_message = chat.upload_top10_video(gr_videos, chat_states, img_lists, n_frms=12)
        else:
            img_lists = [[] for i in range(N)]
            llm_message = chat.upload_video_without_audio(gr_videos, chat_states, img_lists, n_frms=n_frms)

    for user_message, chat_state in zip(user_messages, chat_states):
        chat.ask(user_message, chat_state)

    responses,interval,_= chat.answer(convs=chat_states,
                            img_lists=img_lists,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=512,
                            max_length=3000)
    return responses, chat_states, img_lists,interval


def main(args):
    eval_start_time = time.time()

    # load model
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    cfg = Config(args)
    model_config = cfg.model_cfg
    run_cfg = cfg.run_cfg

    # set after init_distributed_mode() to only log on master.
    Path(run_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger('ivcr.test',run_cfg.output_dir,'log_test.txt')
    cfg.pretty_print()
    message = '\n' + '\n'.join([f'{k:<25}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    model_cls = registry.get_model_class(model_config.arch)
    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model, use_fast=False)
    model = model_cls.from_config(model_config,tokenizer)
    model = model.to("cuda")
    model.eval()

    # 加载数据
    task = tasks.setup_task(cfg)
    tokenizer.pad_token = tokenizer.eos_token
    datasets = task.build_datasets(cfg,tokenizer)
    datasets = datasets['ivcr_instruct']
    datasets = datasets['train']
    len_eval_data = datasets.get_eval_len()
    video_token = model.llama_tokenizer.get_vocab()['<VIDEOTOKEN>']
    all_outputs = []
    print(len_eval_data)
    for i in tqdm(range(len_eval_data)):
        eval_data = datasets.get_eval_data(i)
        video_frames = eval_data["image"]
        user_q = eval_data['sentence']
        input_ids = eval_data['text_input'].to('cuda')
        img_embeds_list = []
        for img, timestamp in zip(video_frames, eval_data["timestamps"]):
            img = img.unsqueeze(0)
            img = img.to('cuda')
            if len(img.size()) == 4:
                ti = 1
                img = einops.repeat(img, 'b c h w -> b c t h w', t=ti)
            img_embeds, atts_img = model.encode_videoQformer_visual(img.to(dtype=torch.float16), 
                                                                    timestamp=timestamp.to('cuda'),
                                                                    is_video_clip=True)
            img_embeds = img_embeds.to('cuda')                                                        
            img_embeds_list.append(img_embeds)
        img_embeds = img_embeds_list
        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == video_token] = 0
        temp_input_embedding = model.llama_model.get_base_model().model.embed_tokens(temp_input_ids)
        new_input_embeds = []
        cur_input_embeds = temp_input_embedding.to('cuda')
        token_pos = torch.where(input_ids == video_token)[0]
        token_pos_list = token_pos.tolist()
        cur_new_input_embeds = None
        pre_pos  = -1
        # print(f"img_embeds:{len(img_embeds)}")
        # print(f"token_pos_list:{token_pos_list}")
        for idx, pos in enumerate(token_pos_list):
            if idx == 0 and idx != len(token_pos_list)-1:
                cur_new_input_embeds = torch.cat((cur_input_embeds[:pos],img_embeds[idx].squeeze(0)),dim=0)
            elif idx == 0 and idx == len(token_pos_list)-1:
                cur_new_input_embeds = torch.cat((cur_input_embeds[:pos],img_embeds[idx].squeeze(0),
                                                  cur_input_embeds[pos+1:]),dim=0)
            elif idx == len(token_pos_list) -1:
                cur_new_input_embeds = torch.cat((cur_new_input_embeds, cur_input_embeds[pre_pos+1:pos],
                                                    img_embeds[idx].squeeze(0),cur_input_embeds[pos+1:]), dim=0)
            else:
                cur_new_input_embeds = torch.cat((cur_new_input_embeds, cur_input_embeds[pre_pos+1:pos],img_embeds[idx].squeeze(0)), dim=0)
            pre_pos = pos
        new_input_embeds.append(cur_new_input_embeds)
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        input_emb = inputs_embeds.to('cuda')
        attn_mask = torch.ones(input_emb.shape[0], input_emb.shape[1], device='cuda')
        attn_mask = attn_mask.int()
        with torch.no_grad():
            outputs = model.llama_model.generate(
                inputs_embeds=input_emb.to(torch.float16),
                attention_mask = attn_mask.to(torch.float16),
                max_new_tokens=1024, #注意使用llama3.2的时候一定要加上这个参数
            )
        output_text = model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(output_text)
        all_outputs.append({'user_q':user_q,'output':output_text})
    save_result(run_cfg.output_dir, all_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./eval_configs/ivcr.yaml')
    args = parser.parse_args()
    main(args)

