import math
import os
from ivcr.datasets.datasets.base_dataset import BaseDataset
from ivcr.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import re
import sys
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import logging
import transformers
import re
import pathlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,LlamaForCausalLM
import copy
from ivcr.processors import transforms_video, AlproVideoTrainProcessor,AlproVideoEvalProcessor
from torchvision import transforms
from ivcr.processors.video_processor import ToTHWC, ToUint8, load_video
from ivcr.conversation.conversation_video import Conversation, SeparatorStyle

video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

llama_v2_video_conversation = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

IGNORE_INDEX = -100

#数据读取代码
class Video_Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor,v_frm, vis_root, ann_root,vid_vname, num_video_query_token=32,
                data_type='video', model_type='vicuna', num_frm=8,
                 sample_type='rand', max_txt_len=512, stride=32,tokenizer = None):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """

        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        
        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            messages_list = json.load(f)
        with open(vid_vname,'r') as file:
            vid_2_name = json.load(file)
        self.messages_list = []
        for mess in messages_list:
            sam_mess = mess.copy()
            if len(mess) <= 15:
                self.messages_list.append(sam_mess)
        self.vid_2_name = []
        for vid in vid_2_name:
            vid_copy = vid.copy()
            if len(vid_copy) <= 7:
                self.vid_2_name.append(vid_copy)
        

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = num_frm
        self.v_frm = v_frm
        self.tokenizer = tokenizer
        
        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms=self.num_frm
        ).transform
        self.eval_transform = AlproVideoEvalProcessor(
            image_size=self.resize_size, n_frms=self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type
        self.sample_type = sample_type
        self.max_txt_len = max_txt_len
        self.stride = stride
        
        print(f"数据长度{len(self.messages_list)}")
        
    def _get_dataset_name_video_name(self, path_name):
        # 使用正则表达式提取所需的部分，允许不同的文件扩展名
        match = re.search(r'.*/([^/]+)/([^/]+)\..*$', path_name)

        if match:
            dataset_name = match.group(1)  # 提取“ActivityNet”
            video_name = match.group(2)     # 提取“v__n9eNF1WaFU”
        else:
            print("No match found.")
        return dataset_name, video_name
    

    def _get_video_path(self, sample):
        # rel_video_fp = sample['video'].split('/')[-1]
        rel_video_fp = sample['video_path']
        full_video_fp = os.path.join(self.vis_root, rel_video_fp)
        gt_value = sample['gt_se']
        return full_video_fp,gt_value

    def _get_video_list_path(self, sample):
        rel_video_fp = sample['video_top10_list']
        gt_video = sample['video_path']
        index = rel_video_fp.index(gt_video)+1
        full_video_fp = [os.path.join(self.vis_root, rel_video) for rel_video in rel_video_fp  ]
        return full_video_fp,index
    
    def _get_video_id(self,sentence):
        result = re.findall(r'VID\d+', sentence)
        return result

    def _get_video_num(self, sentence):
        # result = re.findall(r'\d+', sentence)
        # 提取第一个句点前的内容（若无句点则取整个字符串）
        if "." in sentence:
            first_part = sentence.split(".")[0].strip()
        else:
            first_part = sentence.strip()

        # 分割并过滤空字符
        split_parts = first_part.split(">")
        result = [num.strip() for num in split_parts if num.strip()]

        # 安全转换为整数（处理非数字字符）
        # try:
        #     result = list(map(int, cleaned_parts))
        # except ValueError:
        #     result = []
        #     print("存在非数字字符！")

        # print(result)  # 输出: [4, 2, 1, 3, 5]
        return result

    def get_eval_len(self):
        return len(self.messages_list)

    def get_eval_data(self,index):
        # rerank_sample = self.rerank_messages[index]
        sample = self.messages_list[index]
        video_list = self.vid_2_name[index]
        # new_sample = self.new_messages_list[index]
        video_frm_list = []
        cur_n_frms = []
        time_message_list = []
        assert len(sample) -1 == 2*len(video_list)
        for idx,videos in enumerate(video_list):
            if isinstance(videos, list):  #如果是视频检索，则有top10，故为list类型
                for v in videos:
                    video_path = os.path.join(self.vis_root,v)
                    videos, msg = load_video(
                        video_path=video_path,
                        n_frms=self.num_frm,
                        height=self.resize_size,
                        width=self.resize_size,
                        sampling=self.sample_type, return_msg=True
                    )
                    videos = self.transform(videos)
                    video_frm_list.append(videos)
                    cur_n_frms.append(videos.shape[1])
                    all_timestamp = msg.split('sampled at')[1].replace('seconds.','').strip().split(',')
                    all_timestamp = [f'This frame is sampled at {t.strip()} second.' for t in all_timestamp]

                    all_timestamp = self.tokenizer(
                        all_timestamp,
                        return_tensors="pt",
                        padding="longest",
                        max_length=32,
                        truncation=True,
                    )
                    time_message_list.append(all_timestamp)
            else:
                video_path = os.path.join(self.vis_root,videos)
                videos, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling=self.sample_type, return_msg=True
                )
                videos = self.transform(videos)
                video_frm_list.append(videos)
                cur_n_frms.append(videos.shape[1])
                all_timestamp = msg.split('sampled at')[1].replace('seconds.','').strip().split(',')
                all_timestamp = [f'This frame is sampled at {t.strip()} second.' for t in all_timestamp]

                all_timestamp = self.tokenizer(
                    all_timestamp,
                    return_tensors="pt",
                    padding="longest",
                    max_length=32,
                    truncation=True,
                )
                time_message_list.append(all_timestamp)
                flag = (idx+1)*2
                message = sample[flag-1]
                assert message['role'] == 'user'
                part1, part2 = message.get('content').split("</VIDEO>")
                sample[flag-1]['content'] = part1 + "</VIDEO>. " + msg.strip() + part2
        
        model_input = self.tokenizer.apply_chat_template(sample[:-1], tokenize=False,add_generation_prompt=True)

        input_ids = self.tokenizer([model_input],return_tensors = "pt",add_special_tokens=False).input_ids
        sentence = sample[-2]['content']
        assert sample[-2]['role'] == 'user'
        return {
            "sentence": sentence,
            "image": video_frm_list,
            "text_input": input_ids[0],
            "timestamps": time_message_list,
        }
    
    def extract_video_numbers_first_only(self, text):
        """
        只返回第一个匹配的数字
        
        Args:
            text (str): 输入的字符串
        
        Returns:
            int or None: 第一个找到的数字，如果没有找到则返回None
        """
        pattern = r'(?i)\bvideo\s+(10|[1-9])\b'
        match = re.search(pattern, text)
        
        if match:
            return int(match.group(1))
        return None
    
    def extract_time_ranges(self,text):
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
    def get_gt_index(self,sample):
        sam = sample[-1]
        assert sam['role'] == 'assistant'
        assert sample[-2]['role'] == 'user'
        sam2 = sample[-2]
        if 'Candidate videos' in sam2['content']:
            result = self.extract_video_numbers_first_only(sam['content'])
            return result
        elif 'Current video' in sam2['content']:
            result = self.extract_time_ranges(sam['content'])
            return result
        
    def __getitem__(self, index):
        sample = self.messages_list[index]
        video_list = self.vid_2_name[index]
        # new_sample = self.new_messages_list[index]
        video_frm_list = []
        cur_n_frms = []
        time_message_list = []
        assert len(sample) -1 == 2*len(video_list)
        for idx,videos in enumerate(video_list):
            if isinstance(videos, list):  #如果是视频检索，则有top10，故为list类型
                for v in videos:
                    video_path = os.path.join(self.vis_root,v)
                    videos, msg = load_video(
                        video_path=video_path,
                        n_frms=self.num_frm,
                        height=self.resize_size,
                        width=self.resize_size,
                        sampling=self.sample_type, return_msg=True
                    )
                    videos = self.transform(videos)
                    video_frm_list.append(videos)
                    cur_n_frms.append(videos.shape[1])
                    all_timestamp = msg.split('sampled at')[1].replace('seconds.','').strip().split(',')
                    all_timestamp = [f'This frame is sampled at {t.strip()} second.' for t in all_timestamp]

                    all_timestamp = self.tokenizer(
                        all_timestamp,
                        return_tensors="pt",
                        padding="longest",
                        max_length=32,
                        truncation=True,
                    )
                    time_message_list.append(all_timestamp)
            else:
                video_path = os.path.join(self.vis_root,videos)
                videos, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling=self.sample_type, return_msg=True
                )
                videos = self.transform(videos)
                video_frm_list.append(videos)
                cur_n_frms.append(videos.shape[1])
                all_timestamp = msg.split('sampled at')[1].replace('seconds.','').strip().split(',')
                all_timestamp = [f'This frame is sampled at {t.strip()} second.' for t in all_timestamp]

                all_timestamp = self.tokenizer(
                    all_timestamp,
                    return_tensors="pt",
                    padding="longest",
                    max_length=32,
                    truncation=True,
                )
                time_message_list.append(all_timestamp)
                flag = (idx+1)*2
                message = sample[flag-1]
                assert message['role'] == 'user'
                part1, part2 = message.get('content').split("</VIDEO>")
                sample[flag-1]['content'] = part1 + "</VIDEO>. " + msg.strip() + part2
        cur_token_len = [self.num_video_query_token * math.ceil(
                    cur_n_frm / self.stride) if self.stride > 0 else self.num_video_query_token for cur_n_frm in cur_n_frms]
        model_input = self.tokenizer.apply_chat_template(sample, tokenize=False,add_generation_prompt=False)
        input_ids = self.tokenizer([model_input],return_tensors = "pt",add_special_tokens=False).input_ids
        labels,length = get_label(input_ids, cur_token_len,self.tokenizer)
        flag_content = sample[-1].get('content')
        video_gt_index = self.get_gt_index(sample)
        return {
            "image": video_frm_list,
            "text_input": input_ids[0],
            "labels": labels[0],
            "length": length,
            "timestamps": time_message_list,
            "gt_value":video_gt_index,
        }
    def __len__(self):
        return len(self.messages_list)

    def collater(self, instances):
        input_ids, labels, timestamps,length,gt_value = tuple([instance[key] for instance in instances]
                                              for key in ("text_input", "labels", "timestamps","length","gt_value"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            length = length,
            gt_value = gt_value,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            # target_vid = target_vid,
        )
        images = [instance['image'] for instance in instances]
        batch['images'] = images
        batch_timestamps = []
        for timestamp in timestamps[0]:
            batch_timestamps.append(
                {'input_ids': timestamp['input_ids'], 'attention_mask': timestamp['attention_mask']})
        batch['timestamps'] = batch_timestamps
        return batch


def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from': 'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from': 'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources
def preprocess_for_test(conversation_list,tokenizer):
    msg = "There are 10 videos."
    text = "<Video>" + "<ImageHere>" + "</Video>" + msg+conversation_list[0]['q']
    conv = copy.deepcopy(llama_v2_video_conversation.copy())
    # conv.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    conv.system = ""
    conv.append_message('USER',text)
    prompt = [conv.get_prompt()]
    input_test = tokenizer(
                    prompt, 
                    return_tensors="pt").input_ids
    return input_test


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_txt_len: int = 512, ) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_txt_len,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        max_txt_len: int,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer, max_txt_len)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer, max_txt_len)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_for_llama_v2(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        max_txt_len: int = 512,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    conv = copy.deepcopy(llama_v2_video_conversation.copy())
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        header = f"<s>[INST] <<SYS>>\n{conv.system}\n</SYS>>\n\n"

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        logger = logging.getLogger('ivcr.train')
        logger.info(conv.get_prompt())
        # print(f"conversations: {conversations}")

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=max_txt_len,
        truncation=True,
    ).input_ids
    
    
    targets = copy.deepcopy(input_ids)
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # 为什么减去2,speical token 的数目

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def get_label(input_ids,cur_token_len,tokenizer):
    target = copy.deepcopy(input_ids)
    target = target[0].tolist()
    #use llama2-chat-7b-hf
    inst_token_id = tokenizer.encode("[/INST]", add_special_tokens=False,return_tensors='pt')

    # use llama3.2-3b-instruct
    # inst_token_id = tokenizer(['<|end_header_id|>'],add_special_tokens=False, return_tensors='pt').input_ids

    inst_token_id = inst_token_id[0].tolist()
    inst_token_len = len(inst_token_id)
    last_pos = -1 
    for i in range(len(target) - inst_token_len + 1):
        if target[i:i+inst_token_len] == inst_token_id:
            last_pos = i + inst_token_len -1
    assistant_response = target[last_pos + 1:]
    target[:last_pos+1] = [IGNORE_INDEX] * len(target[:last_pos+1])

    add_token = []
    for i in cur_token_len:
        for j in range(i-1):
            add_token.append(IGNORE_INDEX)
    target = target[:last_pos+1] + add_token + assistant_response
    target = [target]
    target = torch.tensor(target)
    assistant_len = len(assistant_response)
    return target,assistant_len

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
