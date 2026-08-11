#!/usr/bin/env python3
"""
add_prompt_test.py
~~~~~~~~~~~~~~~~~~
A script to add prompts specifically to the test dataset retrieved inside `dataset/subgraph_logos/test.json`.
It replaces the reference to entity embeddings with a confidence directive:
"Only choose an entity if you are really sure about it, if not just choose the first entity"
"""

import os
import json
import argparse

def add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, bkg):
    rel = raw['triple'][1]
    query_entity = raw['query_entity']
    rank_entities = raw['rank_entities']
    pred_type = raw['type']
    answer_options = "(" + ", ".join([f"'{name}'" for name in rank_entities]) + ")"
    
    if pred_type == "predicted_tail":
        question_template = relation_questions_A_to_B.get(rel, "What is related to {}?")
    elif pred_type == "predicted_head":
        question_template = relation_questions_B_to_A.get(rel, "What is related to {}?")
    question = question_template.format(query_entity)

    # Replaced: "\nYou can refer to the entity embeddings: " + refer_str + "."
    # with: "Only choose an entity if you are really sure about it, if not just choose the first entity"
    if bkg:
        prompt = (
            "You are a biomedical scientist. The task is to predict the answer based on the given question, "
            "and you only need to answer one entity. The answer must be in " + answer_options + ". "
            "You must output the exact string of the entity. Do not output any numbers or explanations. "
            "Only choose an entity if you are really sure about it, if not just output the exact name of the first entity"
            ".\n\nQuestion: " + question + "\nAnswer: "
        )
    else:
        prompt = (
            "You are an excellent linguist. The task is to predict the answer based on the given question, "
            "and you only need to answer one entity. The answer must be in " + answer_options + ". "
            "You must output the exact string of the entity. Do not output any numbers or explanations. "
            "Only choose an entity if you are really sure about it, if not just output the exact name of the first entity"
            ".\n\nQuestion: " + question + "\nAnswer: "
        )

    if pred_type == "predicted_tail":
        answer = raw['triple'][2]
    elif pred_type == "predicted_head":
        answer = raw['triple'][0]
    
    raw['input'] = prompt
    raw['output'] = answer

def main():
    parser = argparse.ArgumentParser(description="Add prompt to test.json with modified instructions.")
    parser.add_argument("--test_json_path", type=str, default="dataset/subgraph_logos/test.json", help="Path to test.json")
    parser.add_argument("--output_json_path", type=str, default="dataset/subgraph_logos/test.json", help="Path to save output")
    parser.add_argument("--tail_pred_lex", type=str, default="lexicon/primekg_tail_prediction.json", help="Tail lexicon")
    parser.add_argument("--head_pred_lex", type=str, default="lexicon/primekg_head_prediction.json", help="Head lexicon")
    parser.add_argument("--bkg", action="store_true", default=True, help="Use biomedical context prompt")
    parser.add_argument("--no_bkg", action="store_false", dest="bkg", help="Disable biomedical context prompt")

    args = parser.parse_args()

    print(f"Loading test data from {args.test_json_path}...")
    if not os.path.exists(args.test_json_path):
        raise FileNotFoundError(f"Test dataset not found at {args.test_json_path}")

    with open(args.test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading lexicons...")
    if not os.path.exists(args.tail_pred_lex):
        raise FileNotFoundError(f"Tail lexicon not found at {args.tail_pred_lex}")
    if not os.path.exists(args.head_pred_lex):
        raise FileNotFoundError(f"Head lexicon not found at {args.head_pred_lex}")

    with open(args.tail_pred_lex, 'r', encoding='utf-8') as f:
        relation_questions_A_to_B = json.load(f)
    with open(args.head_pred_lex, 'r', encoding='utf-8') as f:
        relation_questions_B_to_A = json.load(f)

    print(f"Processing {len(data)} items and adding prompts...")
    for item in data:
        add_prompt(item, relation_questions_A_to_B, relation_questions_B_to_A, args.bkg)

    print(f"Saving modified data to {args.output_json_path}...")
    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
    with open(args.output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Success! Prompt addition completed.")

if __name__ == "__main__":
    main()
