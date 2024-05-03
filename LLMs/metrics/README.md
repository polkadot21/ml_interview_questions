# Metrics for LLMs

Measuring the performance of Large Language Models (LLMs) involves several metrics that depend on the specific tasks
they are designed to perform. The performance metrics can broadly be categorized into quantitative and qualitative 
types. Here’s an overview of the primary metrics used:


1. Perplexity (PPL)
 - **Description**: Perplexity is a measure of how well a probability model predicts a sample. In the context of language models, it quantifies how well the model predicts a sequence of tokens. A lower perplexity indicates that the model is more certain about its predictions.
 - **Application**: Typically used in the evaluation of models on language modeling tasks. For example, a language model tasked with predicting the next word in a sentence will have its perplexity calculated based on its probability distribution over the vocabulary.
 - **Example**: If a model has a perplexity of 10 on a test set, it means, on average, the model is as confused as if it had to choose uniformly and independently among 10 possibilities for each next word.

2. Accuracy
 - **Description**: Accuracy measures the fraction of predictions the model got right. In the context of LLMs, it's often used for tasks with discrete outputs, like classification tasks.
 - **Application**: Used in tasks such as sentiment analysis or multiple-choice question answering, where outputs can be clearly defined and matched against ground truth.
 - **Example**: In a sentiment analysis task, if the model correctly identifies the sentiment of 80 out of 100 samples, its accuracy is 80%.

3. F1 Score
 - **Description**: The F1 score is the harmonic mean of precision and recall. It is particularly useful when the class distribution is imbalanced.
 - **Application**: Commonly used in tasks like named entity recognition (NER) or any other scenario where true positives, false negatives, and false positives need distinct consideration.
 - **Example**: In an NER task, F1 can balance the importance of precision (avoiding labeling non-entities as entities) and recall (not missing actual entities).

4. BLEU Score (Bilingual Evaluation Understudy)
 - **Description**: BLEU is a metric used to evaluate a generated sentence to a reference sentence. It compares n-grams of the generated text to the n-grams of the reference text and counts the number of matches.
 - **Application**: Widely used in machine translation and text generation tasks to assess the quality of generated text in comparison to a human reference.
 - **Example**: A machine translation model outputs a sentence translated from English to French, which is then compared to a reference translation to compute the BLEU score.

5. ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)
 - **Description**: ROUGE measures the quality of summaries by comparing them to reference summaries. It includes measures such as ROUGE-N (overlap of N-grams), ROUGE-L (longest common subsequence), and ROUGE-S (skip-bigram co-occurrence statistics).
 - **Application**: Commonly used in text summarization to evaluate the overlap between the generated summary and reference summaries.
 - **Example**: In evaluating a summarization model, ROUGE-N can be used to check how many N-grams in the generated summaries appear in the human-written reference summaries.

6. METEOR (Metric for Evaluation of Translation with Explicit ORdering)
 - **Description**: METEOR is another metric used in machine translation that considers word order, synonymy, and stemming. It aligns generated text with reference texts based on exact, stem, synonym, and paraphrase matches, and then calculates a score based on these alignments.
 - **Application**: Used primarily in translation to evaluate how well the translated output from a model matches the reference, with adjustments for fluent word choice and proper ordering.
 - **Example**: In translating from English to German, METEOR would score the output higher if it correctly uses synonyms or grammatical derivations that reflect the same underlying semantics as the reference text.