import os

import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from rouge_score import rouge_scorer
from collections import Counter

from LLMs.utils.load import load_config


def evaluate_model(data, tokenizer, model):
    total_correct = 0
    f1_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for item in data:
        inputs = tokenizer.encode_plus(item["question"], item["context"], return_tensors='pt')
        with torch.no_grad():
            output = model(**inputs)

        # Get the predicted answer
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits) + 1
        predicted_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        # Evaluate accuracy
        total_correct += int(predicted_answer.strip().lower() == item["true_answer"].strip().lower())

        # Evaluate F1 score
        pred_tokens = predicted_answer.lower().split()
        true_tokens = item["true_answer"].lower().split()
        common = Counter(pred_tokens) & Counter(true_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(true_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)


        # Evaluate ROUGE score
        rouge_score = scorer.score(item["true_answer"], predicted_answer)
        rouge_scores.append(rouge_score['rougeL'].fmeasure)

    # Print the results
    print(f"Accuracy: {total_correct / len(data)}")
    print(f"Average F1 Score: {sum(f1_scores) / len(f1_scores)}")
    print(f"Average ROUGE-L F1 Score: {sum(rouge_scores) / len(rouge_scores)}")

def main():
    cfg: dict = load_config(os.path.join('LLMs', 'metrics', 'question_answering', 'config.json'))

    tokenizer = DistilBertTokenizer.from_pretrained(cfg['model']['name'])
    model = DistilBertForQuestionAnswering.from_pretrained(cfg['model']['name'])
    model.eval()
    evaluate_model(cfg['data'], tokenizer, model)


if __name__ == "__main__":
    main()
