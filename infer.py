#!/usr/bin/env python
# coding: utf-8
import torch
import model_handling
from data_handling import DataCollatorForNormSeq2Seq
from model_handling import EncoderDecoderSpokenNorm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from qs_model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer, pipeline, RobertaForQuestionAnswering
import torch
from nltk import word_tokenize
from transformers.models.auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING

# Init tokenizer and model

tokenizer = model_handling.init_tokenizer()
model = EncoderDecoderSpokenNorm.from_pretrained(
    "./model_spoken", cache_dir=model_handling.cache_dir
)
data_collator = DataCollatorForNormSeq2Seq(tokenizer)

# Infer sample
# bias_list = ['scotland', 'covid', 'delta', 'beta']
# input_str = 'ngày hai tám tháng tư cô vít bùng phát ở sờ cốt lờn chiếm tám mươi phần trăm là biến chủng đen ta và bê ta'


def process_text(text_input, bias_list):
    inputs = tokenizer([text_input])
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    text_output = ""
    if len(bias_list) > 0:
        bias = data_collator.encode_list_string(bias_list)
        bias_input_ids = bias["input_ids"]
        bias_attention_mask = bias["attention_mask"]
    else:
        bias_input_ids = None
        bias_attention_mask = None

    inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "bias_input_ids": bias_input_ids,
        "bias_attention_mask": bias_attention_mask,
    }

    # Format input text **with** bias phrases

    outputs = model.generate(
        **inputs, output_attentions=True, num_beams=1, num_return_sequences=1
    )

    for output in outputs.cpu().detach().numpy().tolist():
        # print('\n', tokenizer.decode(output, skip_special_tokens=True).split(), '\n')
        text_output = tokenizer.sp_model.DecodePieces(
            tokenizer.decode(output, skip_special_tokens=True).split()
        )
    # output: 28/4 covid bùng phát ở scotland chiếm 80 % là biến chủng delta và beta
    return text_output


def tokenize_function(example, tokenizer):
    question_word = word_tokenize(example["question"])
    context_word = word_tokenize(example["context"])

    question_sub_words_ids = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in question_word
    ]
    context_sub_words_ids = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in context_word
    ]
    valid = True
    if (
        len([j for i in question_sub_words_ids + context_sub_words_ids for j in i])
        > tokenizer.max_len_single_sentence - 1
    ):
        valid = False

    question_sub_words_ids = (
        [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    )
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > tokenizer.max_len_single_sentence + 2:
        valid = False

    words_lengths = [
        len(item) for item in question_sub_words_ids + context_sub_words_ids
    ]

    return {"input_ids": input_ids, "words_lengths": words_lengths, "valid": valid}


def data_collator(samples, tokenizer):
    if len(samples) == 0:
        return {}

    def collate_tokens(
        values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
        return res

    input_ids = collate_tokens(
        [torch.tensor(item["input_ids"]) for item in samples],
        pad_idx=tokenizer.pad_token_id,
    )
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(samples)):
        attention_mask[i][: len(samples[i]["input_ids"])] = 1
    words_lengths = collate_tokens(
        [torch.tensor(item["words_lengths"]) for item in samples], pad_idx=0
    )

    batch_samples = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "words_lengths": words_lengths,
    }

    return batch_samples


def extract_answer(inputs, outputs, tokenizer):
    plain_result = []
    for sample_input, start_logit, end_logit in zip(
        inputs, outputs.start_logits, outputs.end_logits
    ):
        sample_words_length = sample_input["words_lengths"]
        input_ids = sample_input["input_ids"]
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = sum(sample_words_length[: torch.argmax(start_logit)])
        # Get the most likely end of answer with the argmax of the score
        answer_end = sum(sample_words_length[: torch.argmax(end_logit) + 1])

        if answer_start <= answer_end:
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            )
            if answer == tokenizer.bos_token:
                answer = ""
        else:
            answer = ""

        score_start = (
            torch.max(torch.softmax(start_logit, dim=-1))
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
        score_end = (
            torch.max(torch.softmax(end_logit, dim=-1)).cpu().detach().numpy().tolist()
        )
        plain_result.append(
            {"answer": answer, "score_start": score_start, "score_end": score_end}
        )
    return plain_result
