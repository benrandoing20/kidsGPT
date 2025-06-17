import json, re
import textstat
from tqdm import tqdm

def preprocess(input_txt, output_jsonl, max_grade=8):
    with open(input_txt, encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            t = line.strip()
            if len(t) < 20: continue
            grade = textstat.flesch_kincaid_grade(t)
            if grade > max_grade: continue
            fout.write(json.dumps({"text": t, "grade": round(min(12, max(0, grade)))}) + "\n")

if __name__ == "__main__":
    preprocess("scrape/all_kid_corpus.txt", "data/data.jsonl")
