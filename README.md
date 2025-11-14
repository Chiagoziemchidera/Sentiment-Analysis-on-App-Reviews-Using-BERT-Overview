# Sentiment Analysis on App Reviews Using BERT

## Overview
This project implements a **sentiment analysis pipeline** using **BERT (Bidirectional Encoder Representations from Transformers)** to classify user reviews into **negative, neutral, or positive** sentiment. Leveraging the state-of-the-art NLP capabilities of BERT, the model can understand contextual relationships in text and provide highly accurate sentiment predictions.  

The dataset consists of app reviews with metadata such as user names, review content, scores, and timestamps. Review scores are mapped to three sentiment categories to train the classifier.  

## Features
- Preprocessing of textual data, including **tokenization**, **special tokens (`[CLS]` and `[SEP]`)**, **padding**, and **attention masks** for BERT input.
- Custom **PyTorch Dataset and DataLoader** for efficient batching of tokenized reviews.
- Fine-tuning of **BERT Base Cased** model for multi-class sentiment classification.
- **Dropout regularization** and **fully connected output layer** for robust predictions.
- Training with **AdamW optimizer** and **linear learning rate scheduler**.
- Evaluation using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrices**.
- Ability to predict sentiment on **raw text reviews**.

## Technical Stack
- **Python 3**
- **PyTorch** for model building and training
- **Hugging Face Transformers** for BERT
- **Scikit-learn** for train-test split and evaluation metrics
- **Pandas & NumPy** for data manipulation
- **Seaborn & Matplotlib** for visualization

## How It Works
1. **Data Preparation**: Reviews are loaded and mapped to sentiment classes (`negative`, `neutral`, `positive`) based on ratings.
2. **Tokenization**: Each review is tokenized with BERT tokenizer, special tokens are added, sequences are padded to a fixed length (160 tokens), and attention masks are created.
3. **Dataset & Dataloader**: Tokenized reviews are wrapped in a custom PyTorch Dataset to feed batches to the model.
4. **Model Architecture**: A BERT model is fine-tuned with a dropout layer and a linear classifier to predict sentiment.
5. **Training & Evaluation**: The model is trained on the training set, validated, and tested. Performance metrics are tracked for analysis.
6. **Prediction**: Sentiment can be predicted for individual reviews or large batches of review text.

## Business Benefits
Implementing this sentiment analysis pipeline can provide **actionable insights from customer reviews**, helping businesses:  
- **Enhance product development**: Identify negative feedback trends and improve product features.
- **Optimize customer experience**: Detect issues quickly and respond to dissatisfied users.
- **Refine marketing strategies**: Understand what features or experiences drive positive sentiment.
- **Support decision-making**: Generate dashboards or automated reports summarizing customer sentiment.
- **Increase engagement and retention**: By acting on insights from reviews, businesses can improve user satisfaction and loyalty.

## Example Usage

```python
review_text = "I love completing my todos! Best app ever!!!"

encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=160,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
)

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)

print(f"Review text: {review_text}")
print(f"Sentiment  : {class_names[prediction]}")
```
## Performance
- **Test Accuracy:** ~88%
- **F1-Scores:** 
  - Negative: 0.88  
  - Neutral: 0.84  
  - Positive: 0.92
- **Confusion Matrix:** Indicates the model is slightly less confident on neutral reviews, but overall performance is strong.

## Future Improvements
- Integrate **multi-lingual BERT models** for international app reviews.
- Expand to **aspect-based sentiment analysis** to detect sentiment for specific features.
- Deploy as a **real-time sentiment analysis API** for continuous feedback monitoring.
