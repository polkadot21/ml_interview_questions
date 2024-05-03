import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from torch.quantization import quantize_dynamic
import time

from LLMs.utils.load import load_config


def get_answer(model, inputs, tokenizer):
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0, answer_start:answer_end]))

    return answer


def measure_performance(model, inputs, tokenizer):
    start_time = time.time()
    answer = get_answer(model, inputs, tokenizer)
    end_time = time.time()
    return end_time - start_time, answer


def main():
    cfg: dict = load_config()
    tokenizer = DistilBertTokenizer.from_pretrained(cfg['model']['name'])
    model = DistilBertForQuestionAnswering.from_pretrained(cfg['model']['name'])

    inputs = tokenizer(cfg['data'][0]['question'], cfg['data'][0]['context'], return_tensors="pt")

    model_quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    original_time, original_answer = measure_performance(model, inputs, tokenizer)
    quantized_time, quantized_answer = measure_performance(model_quantized, inputs, tokenizer)

    print(f"Original model time: {original_time:.3f}s, Answer: {original_answer}")
    print(f"Quantized model time: {quantized_time:.3f}s, Answer: {quantized_answer}")


if __name__ == "__main__":
    main()
