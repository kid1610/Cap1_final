from transformers import AutoTokenizer, T5ForConditionalGeneration
import re

model = T5ForConditionalGeneration.from_pretrained(
    "./SC_checking/content/checkpoint-18500"
)
tokenizer = AutoTokenizer.from_pretrained("./SC_checking/content/checkpoint-18500")
global SPECIAL
SPECIAL = []


def get_correct(sentences):
    sentences = format_sent(sentences)
    inputs = tokenizer(sentences, max_length=512, return_tensors="pt", truncation=True)
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,
    )
    sentences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    sentences = format_after(str(sentences[0]))
    SPECIAL = []
    return sentences


def is_special(token):
    return bool(
        re.match(
            r"[^\u0000-\u05C0\u2100-\u214Fa-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+",
            token,
        )
    )


# def replace_special(token):


def format_sent(sentence):
    sentence = re.sub(r"""\s([?.!,:;](?:\s|$))""", r"\1", sentence)
    sentence = re.sub(r'\(\s*([^"]*?)\s*\)', r"(\1)", sentence)
    sentence = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', sentence)
    sentence = re.sub(r"\s*\\\s*", r"\\", sentence)
    sentence = re.sub(r"\s*/\s*", r"/", sentence)
    sentence = re.sub(r"(?<=\d[.]) (?=\d)", "", sentence)
    sentence = re.sub(r"(?=\s[.,])", "", sentence)
    sentence = re.sub(r"(?<=\\)\s", "", sentence)
    sentence = re.sub(r"(?<=\w[,])", " ", sentence)
    sentence = re.sub(r"(?<=[?!])\.", "", sentence)
    sentence = re.sub(r"[.][.][.]*", "…", sentence)
    sentence = re.sub(r" - ", "-", sentence)
    sentence = re.sub(r"(?<=[-])\s", "", sentence)
    sentence = re.sub(r"\s(?=[-])", "", sentence)
    sentence = re.sub(r"(?<=[–])", " ", sentence)
    sentence = re.sub(r"(?=[–])", " ", sentence)
    sentence = re.sub(r"\s(?=\s)", "", sentence)
    token = sentence.split()
    for i in range(len(token)):
        # print(token[i])
        # print(i)
        if is_special(token[i]) == True:
            SPECIAL.append(token[i])
            token[i] = re.sub(
                r"[^\u0000-\u05C0\u2100-\u214Fa-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+",
                "SPECIAL",
                token[i],
            )
    sentence = " ".join(token)
    return sentence


def format_after(sentence):
    j = 0
    token = sentence.split()
    for i in range(len(token)):
        if token[i] == "SPECIAL":
            token[i] = re.sub("SPECIAL", SPECIAL[j], token[i])
            j = j + 1
    sentence = " ".join(token)
    return sentence


if __name__ == "__main__":
    sent = """
        ☺☺☺☺☺☺☺☺☺☺☺☺☺ ddaay, la moot url ☺
    """
    # # print(format_sent(sent))
    # # senten_before  = format_sent(sent)
    # # print(format_after(senten_before))
    # print(get_correct(sent))
