import torch
import numpy as np
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import pandas as pd

from transformers import logging

logging.set_verbosity_error()
import warnings
import os
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers", lineno=1656)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity_warning()
logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    disable_torch_init()

    model_path = 'output_folder/Hawkeye'
    # model_path = 'LanguageBind/video-llava-7b'
    model_base = 'lmsys/vicuna-7b-v1.5'
    # model_base = None
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit,
                                                                     load_4bit, device=device)
    video_processor = processor['video']

    for video_id_folder in tqdm(os.listdir('dataset/vid_split/test')):
        # if video_id_folder == '295_Ekman6_anger_932':
        print(video_id_folder)
        video_id_folder_path = os.path.join('dataset/vid_split/test_new', video_id_folder)
        video_list = []
        name = []
        res = []
        for video in os.listdir(video_id_folder_path):
            video_list.append(video)
        video_list = sorted(video_list, key=lambda x: int(x.split("_")[1].split(".")[0]))

        for video in tqdm(video_list):
            name.append(video)
            video_path = os.path.join(video_id_folder_path, video)
            pose_path = os.path.join('dataset/pose_feat/test/{}'.format(video_id_folder),
                                    'frame_{}.npy'.format(int(video.split('.')[0])))

            pose_feature = torch.from_numpy(np.load(pose_path))
            if pose_feature.size(0) < 5:
                padding_size = 5 - pose_feature.size(0)
                pose_feature_pad = torch.cat((pose_feature, torch.zeros((padding_size, 17, 5))), dim=0)
            else:
                pose_feature_pad = pose_feature[:5, :, :]

            scene_path = os.path.join('dataset/graph_feat/test/{}'.format(video_id_folder),
                                    'frame_{}.npy'.format(int(video.split('.')[0])))

            scene_feature = torch.from_numpy(np.load(scene_path))
            if scene_feature.size(0) < 5:
                padding_size = 5 - scene_feature.size(0)
                scene_feature_pad = torch.cat((scene_feature, torch.zeros((padding_size, 353))), dim=0)
            else:
                scene_feature_pad = scene_feature[:5, :]

            inp = 'Your prompt here'

            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles

            video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
            if type(video_tensor) is list:
                tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
            else:
                tensor = video_tensor.to(model.device, dtype=torch.float16)

            if type(pose_feature_pad) is list:
                tensor_pose = [pose.to(model.device, dtype=torch.float16) for pose in pose_feature_pad]
            else:
                tensor_pose = pose_feature_pad.to(model.device, dtype=torch.float16)

            if type(scene_feature_pad) is list:
                tensor_scene = [scene.to(model.device, dtype=torch.float16) for scene in scene_feature_pad]
            else:
                tensor_scene = scene_feature_pad.to(model.device, dtype=torch.float16)
            key = ['video']

            inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(
                0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[tensor, [tensor_pose], [tensor_scene], key],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=16,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # print(outputs)
            res.append(outputs)
        df = pd.DataFrame({
            'file': name,
            'output': res
        })
        save_path = 'dataset/saved_result/test_res/{}.csv'.format(video_id_folder)
        df.to_csv(save_path, index=False)

    for video_id_folder in tqdm(os.listdir('dataset/Ucf/Ucfcrime_split')):
        if 'Normal' not in video_id_folder:
            video_id_folder_path = os.path.join('dataset/Ucf/Ucfcrime_split', video_id_folder)
            video_list = []
            name = []
            res = []
            for video in os.listdir(video_id_folder_path):
                video_list.append(video)
            video_list = sorted(video_list, key=lambda x: int(x.split("_")[1].split(".")[0]))

            for video in tqdm(video_list):
                name.append(video)
                video_path = os.path.join(video_id_folder_path, video)

                pose_path = os.path.join('dataset/Ucf/pose_feat/{}'.format(video_id_folder),
                                        'frame_{}.npy'.format(int(video.split('.')[0])))

                pose_feature = torch.from_numpy(np.load(pose_path))
                
                if pose_feature.size(0) < 5:
                    padding_size = 5 - pose_feature.size(0)
                    pose_feature_pad = torch.cat((pose_feature, torch.zeros((padding_size, 17, 5))), dim=0)
                else:
                    pose_feature_pad = pose_feature[:5, :, :]

                scene_path = os.path.join('dataset/Ucf/graph_feat/{}'.format(video_id_folder),
                                    'frame_{}.npy'.format(int(video.split('.')[0])))

                scene_feature = torch.from_numpy(np.load(scene_path))
                if scene_feature.size(0) < 5:
                    padding_size = 5 - scene_feature.size(0)
                    scene_feature_pad = torch.cat((scene_feature, torch.zeros((padding_size, 353))), dim=0)
                else:
                    scene_feature_pad = scene_feature[:5, :]

                inp = 'Your prompt here'

                conv_mode = "llava_v1"
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles

                video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
                if type(video_tensor) is list:
                    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                else:
                    tensor = video_tensor.to(model.device, dtype=torch.float16)
                    
                if type(pose_feature_pad) is list:
                    tensor_pose = [pose.to(model.device, dtype=torch.float16) for pose in pose_feature_pad]
                else:
                    tensor_pose = pose_feature_pad.to(model.device, dtype=torch.float16)

                if type(scene_feature_pad) is list:
                    tensor_scene = [scene.to(model.device, dtype=torch.float16) for scene in scene_feature_pad]
                else:
                    tensor_scene = scene_feature_pad.to(model.device, dtype=torch.float16)

                key = ['video']

                inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(
                    0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=[tensor, [tensor_pose], [tensor_scene], key],
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=32,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                # print(outputs)
                res.append(outputs)
            df = pd.DataFrame({
                'file': name,
                'output': res
            })
            save_path = 'dataset/saved_result/test_res/{}.csv'.format(video_id_folder)
            df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
