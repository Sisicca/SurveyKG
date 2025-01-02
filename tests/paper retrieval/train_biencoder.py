import argparse

from datasets import load_dataset
from transformers.integrations import TensorBoardCallback
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

parser = argparse.ArgumentParser(description="Semantic search with SentenceTransformer")
parser.add_argument(
    "--model_path", 
    type=str, 
    default="allenai-specter",
    help="Path to the SentenceTransformer model"
)
parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/train_data.json",
        help="Path to the training data JSON file"
    )
arguments = parser.parse_args()

model_path = arguments.model_path
data_path = arguments.data_path

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    model_path
)

# 3. Load a dataset to finetune on
dataset = load_dataset("json", data_files=data_path)
full_dataset = dataset["train"]
train_test_split = full_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"/root/autodl-tmp/models_during_training/{model_path}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    logging_dir = f'./autodl-tmp/logs_{model_path}',
    run_name="allenai-specter",  # Will be used in W&B if `wandb` is installed
)

# # 6. (Optional) Create an evaluator & evaluate the base model
# dev_evaluator = TripletEvaluator(
#     anchors=eval_dataset["anchor"],
#     positives=eval_dataset["positive_example"],
#     name="data-dev",
# )
# dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    # evaluator=dev_evaluator,
    callbacks=[TensorBoardCallback()],
)
trainer.train()

# # (Optional) Evaluate the trained model on the test set
# test_evaluator = TripletEvaluator(
#     anchors=test_dataset["anchor"],
#     positives=test_dataset["positive"],
#     negatives=test_dataset["negative"],
#     name="all-nli-test",
# )
# test_evaluator(model)

# 8. Save the trained model
model.save_pretrained(f"/root/autodl-tmp/models/{model_path}-survey/final")

# # # 9. (Optional) Push it to the Hugging Face Hub
# # model.push_to_hub("mpnet-base-all-nli-triplet")