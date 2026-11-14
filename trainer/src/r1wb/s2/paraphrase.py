"""More wordings for the training instructions, written by a language model and checked by it.

For every instruction and language the model is asked for N other ways to say it (meaning unchanged: same action,
direction and speed). Each answer is then judged by the same model in a separate question ("does this sentence ask
for exactly ...?"); only confirmed ones are kept, and anything equal to a held-out phrasing is dropped (the held-out
set stays unseen). Output: a json {meaning: {language: [phrasings]}} that r1wb.s2.instructions.extra_phrases reads.

  python -m r1wb.s2.paraphrase --backbone <vision-language model dir> --out <phrases.json> [--count 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch

from r1wb.s2 import instructions

DESCRIBE = {
  "ko": {"stand": "제자리에 멈춰 서서 움직이지 않기", "rise": "수그린 자세에서 똑바로 일어서기",
         "walk": "{d} {s} 걷기", "turn": "제자리에서 {t} {s} 돌기", "crouch": "{c} 수그리기(몸을 낮추기)"},
  "en": {"stand": "stop and stand still without moving", "rise": "stand up straight from a crouch",
         "walk": "walk {d} {s}", "turn": "turn {t} in place {s}", "crouch": "crouch (lower the body) {c}"},
}
ASK = {
  "ko": "로봇에게 내리는 지시문입니다. 뜻(동작, 방향, 속도)은 그대로 두고 표현만 바꾼 한국어 문장 {n}개를 써 주세요. "
        "말투와 단어를 다양하게 하되 한 줄에 한 문장만, 번호나 설명 없이 쓰세요.\n지시의 뜻: {meaning}\n예: {example}",
  "en": "This is an instruction given to a robot. Write {n} other English sentences with exactly the same meaning (same "
        "action, direction and speed), varied in wording and tone. One sentence per line, no numbering, no explanations.\n"
        "Meaning: {meaning}\nExample: {example}",
}
JUDGE = {
  "ko": "로봇 지시문: \"{text}\"\n이 문장이 정확히 다음을 요구합니까? {meaning}\n방향이나 속도가 다르거나 빠져 있으면 '아니오'입니다. '예' 또는 '아니오'로만 답하세요.",
  "en": "Robot instruction: \"{text}\"\nDoes this sentence ask for exactly this: {meaning}?\nIf the direction or speed differs or is missing, the answer is no. Answer only yes or no.",
}


def describe(item: dict, language: str) -> str:
  words = instructions.KOREAN if language == "ko" else instructions.ENGLISH
  kind = item["kind"]
  fields = {}
  if kind == "walk":
    fields = {"d": words["direction"][item["direction"]], "s": words["speed_other"][item["speed"]]}
  elif kind == "turn":
    fields = {"t": instructions.TURN_WORDS[language][0][item["side"]], "s": instructions.TURN_SPEED[language][0][item["rate"]]}
  elif kind == "crouch":
    fields = {"c": instructions.CROUCH_WORDS[language][0][item["depth"]]}
  return " ".join(DESCRIBE[language][kind].format(**fields).split())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--count", type=int, default=20)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  from transformers import AutoModelForImageTextToText, AutoProcessor
  processor = AutoProcessor.from_pretrained(args.backbone)
  processor.tokenizer.padding_side = "left"
  model = AutoModelForImageTextToText.from_pretrained(args.backbone, dtype=torch.bfloat16).to(args.device).eval()

  def answer(prompts: list[str], tokens: int, sample: bool) -> list[str]:
    texts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
    batch = processor(text=texts, padding=True, return_tensors="pt").to(args.device)
    with torch.no_grad():
      out = model.generate(**batch, max_new_tokens=tokens, do_sample=sample, temperature=0.9 if sample else None,
                           top_p=0.95 if sample else None)
    return processor.tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

  def clean(line: str) -> str:
    return " ".join(re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"“”').split())

  result, kept, asked = {}, 0, 0
  for item in instructions.every_instruction():
    name = instructions.identifier(item)
    result[name] = {}
    for language in ("ko", "en"):
      meaning = describe(item, language)
      known = set(instructions.all_phrases(item, language)) | set(instructions.all_phrases(item, language, held_out=True))
      example = instructions.all_phrases(item, language)[0]
      raw = answer([ASK[language].format(n=args.count, meaning=meaning, example=example)] * 2, 40 * args.count, True)
      lines = []
      for text in raw:
        for line in text.splitlines():
          line = clean(line)
          if 2 <= len(line) <= 80 and line not in known and line not in lines:
            lines.append(line)
      verdicts = answer([JUDGE[language].format(text=line, meaning=meaning) for line in lines], 4, False) if lines else []
      yes = ("예", "네") if language == "ko" else ("yes",)
      good = [line for line, verdict in zip(lines, verdicts) if verdict.strip().lower().startswith(yes)]
      result[name][language] = good
      asked += len(lines); kept += len(good)
      print(f"{name:28s} {language}: {len(good)}/{len(lines)} kept  e.g. {good[:3]}", flush=True)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
  print(f"saved {args.out}: {kept} phrasings kept of {asked} written")


if __name__ == "__main__":
  main()
