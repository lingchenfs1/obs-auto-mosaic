"""生成「血腥 / 暴力」检测用的 CLIP 文字向量文件 data/scene/clip_prompts.bin。

插件运行时只跑 CLIP 的图像模型，文字描述的向量在这里预先算好（需要分词器和文字模型，只在开发时用）。
改了下面的描述后重跑本脚本，并提交生成的 .bin。

依赖（建议放在虚拟环境里）：pip install tokenizers onnxruntime numpy
模型：https://huggingface.co/Xenova/clip-vit-base-patch32（MIT）的 tokenizer.json 和 onnx/text_model_int8.onnx
用法：python tools/make-clip-prompts.py <tokenizer.json> <text_model_int8.onnx> <输出 .bin>

文件格式（小端）：8 字节 "CLIPPRM1"，int32 维度，int32 血腥条数、暴力条数、正常条数，
随后是各条描述归一化后的 float32 向量（按 血腥、暴力、正常 的顺序）。
"""
import struct
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 血腥：流血、伤口、尸体
GORE = [
    "a photo of a bloody wound",
    "a gory image with a lot of blood",
    "a person covered in blood",
    "a severe injury with blood",
    "a mutilated body",
    "a dead body",
]
# 暴力：打斗、殴打、持械威胁、暴乱
VIOLENCE = [
    "a photo of people fighting violently",
    "a person being beaten",
    "a violent brawl",
    "a person pointing a gun at another person",
    "a person attacking someone with a knife",
    "a riot with violent clashes",
]
# 正常画面（对照组）：特意放进容易混淆的体育对抗、射击游戏、红色食物和衣服、抗议人群等
SAFE = [
    "a photo of a person",
    "a person talking to the camera",
    "a person sitting at a desk",
    "a group of people",
    "a selfie",
    "people playing sports",
    "a boxing or kickboxing match",
    "a screenshot of a video game",
    "a first person shooter video game",
    "a screenshot of a computer screen with text",
    "a plate of food",
    "a landscape",
    "a city street",
    "a person in a swimsuit at the beach",
    "a bodybuilder posing",
    "a concert crowd",
    "an animal",
    "a cartoon",
    "people wearing red clothes",
    "a crowd of people at a protest",
]


def main():
    tok_path, model_path, out_path = sys.argv[1:4]
    tok = Tokenizer.from_file(tok_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    vecs = []
    for p in GORE + VIOLENCE + SAFE:
        ids = np.array([tok.encode(p).ids], np.int64)
        e = sess.run(None, {"input_ids": ids})[0][0].astype(np.float32)
        vecs.append(e / np.linalg.norm(e))
    vecs = np.stack(vecs)
    with open(out_path, "wb") as f:
        f.write(b"CLIPPRM1")
        f.write(struct.pack("<4i", vecs.shape[1], len(GORE), len(VIOLENCE), len(SAFE)))
        f.write(vecs.astype("<f4").tobytes())
    print(f"{out_path}：{len(vecs)} 条描述，{vecs.shape[1]} 维")


if __name__ == "__main__":
    main()
