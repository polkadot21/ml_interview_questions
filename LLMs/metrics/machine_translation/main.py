import os

from transformers import FSMTForConditionalGeneration, FSMTTokenizer, MarianTokenizer, MarianMTModel
from nltk.translate.bleu_score import corpus_bleu

from LLMs.utils.load import load_config


def translate_and_evaluate(data, tokenizer, model):
    translations = []
    references = []

    for item in data:
        input_ids = tokenizer.encode(item["english"], return_tensors="pt")
        outputs = model.generate(input_ids)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translation)

        refs = [ref.split() for ref in item["french_refs"]]
        references.append(refs)

    bleu_score = corpus_bleu(references, [t.split() for t in translations])

    for eng, trans in zip(data, translations):
        print(f'English: {eng["english"]} - Translated: {trans}')
    print(f'\nBLEU score: {bleu_score}')


def main():
    cfg: dict = load_config(os.path.join('LLMs', 'metrics', 'machine_translation', 'config.json'))

    tokenizer = MarianTokenizer.from_pretrained(cfg['model']['name'])
    model = MarianMTModel.from_pretrained(cfg['model']['name'])
    translate_and_evaluate(cfg['data'], tokenizer, model)


if __name__ == "__main__":
    main()
