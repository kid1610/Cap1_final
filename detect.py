from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

# import kenlm
from infer import process_text
from os import getenv
from transformers import Wav2Vec2Processor
import onnxruntime as rt
import numpy as np


class Detect:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("./s2t_model")
        self.model = Wav2Vec2ForCTC.from_pretrained("./s2t_model")
        self.lm_file = "./s2t_model/vi_lm_4grams.bin"
        self.bias_list = []
        ONNX_PATH = getenv("model_path", "./s2t_model/transformers_vi.onnx")
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = rt.InferenceSession(ONNX_PATH, sess_options)
        with open("vocab.txt", "r", encoding="utf8") as f:
            vocabs = f.read().split("\n")
        for vocab in vocabs:
            self.bias_list.append(vocab)

    def map_to_array(self, batch):
        speech, _ = librosa.load(batch["file"], sr=16000)
        batch["speech"] = speech
        return batch

    # def get_decoder_ngram_model(self, tokenizer, ngram_lm_path):
    #     vocab_dict = tokenizer.get_vocab()
    #     sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    #     vocab = [x[1] for x in sort_vocab][:-2]
    #     vocab_list = vocab
    #     # convert ctc blank character representation
    #     vocab_list[tokenizer.pad_token_id] = ""
    #     # replace special characters
    #     vocab_list[tokenizer.unk_token_id] = ""
    #     # vocab_list[tokenizer.bos_token_id] = ""
    #     # vocab_list[tokenizer.eos_token_id] = ""
    #     # convert space character representation
    #     vocab_list[tokenizer.word_delimiter_token_id] = " "
    #     # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    #     alphabet = Alphabet.build_alphabet(
    #         vocab_list, ctc_token_idx=tokenizer.pad_token_id
    #     )
    #     lm_model = kenlm.Model(ngram_lm_path)
    #     decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    #     return decoder

    # using with n-gram model
    # def process_data(self, path):
    #     ngram_lm_model = self.get_decoder_ngram_model(
    #         self.processor.tokenizer, self.lm_file
    #     )
    #     ds = self.map_to_array({"file": path})
    #     input_values = self.processor(
    #         ds["speech"], return_tensors="pt", padding="longest"
    #     ).input_values
    #     logits = self.model(input_values).logits[0]
    #     beam_search_output = ngram_lm_model.decode(
    #         logits.cpu().detach().numpy(), beam_width=500
    #     )
    #     text_output = process_text(str(beam_search_output), self.bias_list)
    #     return beam_search_output, text_output

    def process_data(self, path):
        ds = self.map_to_array({"file": path})
        input_values = self.processor(
            ds["speech"], return_tensors="pt", padding="longest"
        ).input_values
        onnx_outputs = self.session.run(
            None, {self.session.get_inputs()[0].name: input_values.numpy()}
        )[0]
        predicted_ids = np.argmax(onnx_outputs, axis=-1)
        transcription = self.processor.decode(predicted_ids.squeeze().tolist())
        return transcription

    # don't use n-gram model
    # def process_data(self,path):
    #     ds = self.map_to_array({"file": path })
    #     input_values = self.processor(ds["speech"], return_tensors="pt", padding="longest").input_values
    #     logits = self.model(input_values).logits
    #     predicted_ids = torch.argmax(logits, dim=-1)
    #     transcription = self.processor.batch_decode(predicted_ids)
    #     return transcription[0],len(transcription[0])
