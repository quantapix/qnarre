# Copyright 2021 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# conditional text generation with (GPT/GPT-2/CTRL/Transformer-XL/XLNet)

import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

MAX_LENGTH = int(10000)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def adjust_length_to_model(x, lim):
    if x < 0 and lim > 0:
        x = lim
    elif 0 < lim < x:
        x = lim
    elif x < 0:
        x = MAX_LENGTH
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_language", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    ps = parser.parse_args()
    ps.device = torch.device("cuda" if torch.cuda.is_available() and not ps.no_cuda else "cpu")
    ps.n_gpu = 0 if ps.no_cuda else torch.cuda.device_count()
    log.warning(f"device: {ps.device}, n_gpu: {ps.n_gpu}, 16-bits training: {ps.fp16}")

    def set_seed(ps):
        np.random.seed(ps.seed)
        torch.manual_seed(ps.seed)
        if ps.n_gpu > 0:
            torch.cuda.manual_seed_all(ps.seed)

    def prepare_ctrl_input(ps, _, tokenizer, prompt):
        if ps.temperature > 0.7:
            log.info("CTRL typically works better with lower temperatures (and lower top_k).")
        y = tokenizer.encode(prompt, add_special_tokens=False)
        if not any(y[0] == x for x in tokenizer.control_codes.values()):
            log.info(
                "WARNING! You are not starting your generation from a control code so you won't get good results"
            )
        return prompt

    def prepare_xlm_input(ps, model, tokenizer, prompt):
        # kw = {"language": None, "MSK_TOK": None}
        use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
        if hasattr(model.config, "lang2id") and use_lang_emb:
            ls = model.config.lang2id.keys()
            if ps.xlm_language in ls:
                l = ps.xlm_language
            else:
                l = None
                while l not in ls:
                    l = input("Using XLM. Select language in " + str(list(ls)) + " >>> ")
            model.config.LANG = model.config.lang2id[l]
            # kw["language"] = tokenizer.lang2id[l]
        return prompt

    def prepare_xlnet_input(ps, _, tokenizer, prompt):
        x = ps.prefix if ps.prefix else ps.padding_text if ps.padding_text else PREFIX
        prompt = x + prompt
        return prompt

    def prepare_transfoxl_input(ps, _, tokenizer, prompt):
        x = ps.prefix if ps.prefix else ps.padding_text if ps.padding_text else PREFIX
        prompt = x + prompt
        return prompt

    PREPROCESSING_FUNCTIONS = {
        "ctrl": prepare_ctrl_input,
        "xlm": prepare_xlm_input,
        "xlnet": prepare_xlnet_input,
        "transfo-xl": prepare_transfoxl_input,
    }

    set_seed(ps)
    try:
        ps.model_type = ps.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[ps.model_type]
    except KeyError:
        raise KeyError("Model {}  is not supported")
    tokenizer = tokenizer_class.from_pretrained(ps.model_name)
    model = model_class.from_pretrained(ps.model_name)
    model.to(ps.device)
    if ps.fp16:
        model.half()
    ps.length = adjust_length_to_model(ps.length, lim=model.config.n_pos)
    log.info(ps)
    x = ps.prompt if ps.prompt else input("Model prompt >>> ")
    if ps.model_type in PREPROCESSING_FUNCTIONS.keys():
        prep = PREPROCESSING_FUNCTIONS.get(ps.model_type)
        y = prep(ps, model, tokenizer, x)
        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kws = {"add_space_before_punct_symbol": True}
        else:
            kws = {}
        prompt = tokenizer.encode(y, add_special_tokens=False, return_tensors="pt", **kws)
    else:
        prefix = ps.prefix if ps.prefix else ps.padding_text
        prompt = tokenizer.encode(prefix + x, add_special_tokens=False, return_tensors="pt")
    prompt = prompt.to(ps.device)
    if prompt.size()[-1] == 0:
        ins = None
    else:
        ins = prompt
    out = model.generate(
        input_ids=ins,
        max_len=ps.length + len(prompt[0]),
        temperature=ps.temperature,
        top_k=ps.k,
        top_p=ps.p,
        repetition_penalty=ps.repetition_penalty,
        do_sample=True,
        num_return_sequences=ps.num_return_sequences,
    )
    if len(out.shape) > 2:
        out.squeeze_()
    ys = []
    for i, x in enumerate(out):
        print(f"=== GENERATED SEQUENCE {i + 1} ===")
        x = x.tolist()
        y = tokenizer.decode(x, clean_up_tokenization_spaces=True)
        y = y[: y.find(ps.stop_token) if ps.stop_token else None]
        y = x + y[len(tokenizer.decode(prompt[0], clean_up_tokenization_spaces=True)) :]
        ys.append(y)
        print(y)

    return ys


if __name__ == "__main__":
    main()


"""
python gen.py \
    --model_type=gpt2 \
    --model_name=gpt2
"""
