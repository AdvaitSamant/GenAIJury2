"""
Training and Comprehensive Evaluation Pipeline for Intent Classification.
Optimized for NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) and Scaled Massive Datasets.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import (
    DATASET_CHOICE,
    DATASET_CONFIGS,
    CLASSIFIER_BACKBONE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    LABEL_SMOOTHING_FACTOR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    USE_FP16,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    SEED,
    DEVICE,
    GPU_NAME,
    TOTAL_VRAM_GB,
)
from evaluation.metrics_and_plots import (
    compute_comprehensive_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_class_performance,
    plot_metrics_summary,
)


def load_and_prepare_dataset(choice: str = DATASET_CHOICE, seed: int = SEED) -> Tuple[DatasetDict, List[str]]:
    """
    Loads large-scale intent datasets:
    - 'banking200k': Massive Banking Intent Corpus (220,000 utterances, 110 intents)
    - 'massive': Amazon MASSIVE (16,521 utterances, 60 intents)
    - 'banking77': PolyAI Banking77 (13,083 utterances, 77 fine-grained intents)
    - 'merged': Unified 29.6k row multi-domain intent benchmark (137 intents)
    """
    print(f"\n[Dataset Loader] Preparing dataset: '{choice}'...")

    if choice == "banking200k":
        parquet_path = Path(__file__).resolve().parent / "data" / "banking_intent_200k.parquet"
        if not parquet_path.exists():
            from generate_banking_200k import build_banking_200k_dataset, export_dataset
            df, tax = build_banking_200k_dataset(samples_per_intent=2000)
            export_dataset(df, tax)

        raw_ds = load_dataset("parquet", data_files=str(parquet_path))["train"]
        unique_intents = sorted(list(set(raw_ds["label_text"])))
        intent_names = unique_intents

        # Stratified/random split: 80% train (176,000), 10% validation (22,000), 10% test (22,000)
        split_1 = raw_ds.train_test_split(test_size=0.20, seed=seed)
        split_2 = split_1["test"].train_test_split(test_size=0.50, seed=seed)
        dataset = DatasetDict({
            "train": split_1["train"],
            "validation": split_2["train"],
            "test": split_2["test"],
        })

    elif choice == "massive":
        raw = load_dataset(DATASET_CONFIGS["massive"]["hf_name"])
        unique_intents = sorted(list(set(raw["train"]["label_text"])))
        intent2id = {name: i for i, name in enumerate(unique_intents)}

        def remap_massive(batch):
            batch["label"] = [intent2id[name] for name in batch["label_text"]]
            return batch

        dataset = raw.map(remap_massive, batched=True)
        intent_names = unique_intents

    elif choice == "banking77":
        raw = load_dataset(DATASET_CONFIGS["banking77"]["hf_name"])
        intent_names = raw["train"].features["label"].names
        split = raw["train"].train_test_split(test_size=0.10, seed=seed, stratify_by_column="label")
        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            "test": raw["test"],
        })

    elif choice == "merged":
        print("[Dataset Loader] Merging Amazon MASSIVE + Banking77 into Unified Mega-Corpus...")
        d_massive = load_dataset(DATASET_CONFIGS["massive"]["hf_name"])
        d_bank = load_dataset(DATASET_CONFIGS["banking77"]["hf_name"])
        bank_labels = d_bank["train"].features["label"].names

        # Namespace labels to prevent collision
        massive_train_texts = d_massive["train"]["text"]
        massive_train_labels = [f"massive_{name}" for name in d_massive["train"]["label_text"]]

        massive_val_texts = d_massive["validation"]["text"]
        massive_val_labels = [f"massive_{name}" for name in d_massive["validation"]["label_text"]]

        massive_test_texts = d_massive["test"]["text"]
        massive_test_labels = [f"massive_{name}" for name in d_massive["test"]["label_text"]]

        bank_train_texts = d_bank["train"]["text"]
        bank_train_labels = [f"bank_{bank_labels[i]}" for i in d_bank["train"]["label"]]

        bank_test_texts = d_bank["test"]["text"]
        bank_test_labels = [f"bank_{bank_labels[i]}" for i in d_bank["test"]["label"]]

        all_unique = sorted(list(set(massive_train_labels + bank_train_labels)))
        intent2id = {name: i for i, name in enumerate(all_unique)}
        intent_names = all_unique

        # Combine datasets
        dataset = DatasetDict({
            "train": Dataset.from_dict({
                "text": massive_train_texts + bank_train_texts[:8000],
                "label": [intent2id[l] for l in massive_train_labels + bank_train_labels[:8000]],
            }).shuffle(seed=seed),
            "validation": Dataset.from_dict({
                "text": massive_val_texts + bank_train_texts[8000:],
                "label": [intent2id[l] for l in massive_val_labels + bank_train_labels[8000:]],
            }),
            "test": Dataset.from_dict({
                "text": massive_test_texts + bank_test_texts,
                "label": [intent2id[l] for l in massive_test_labels + bank_test_labels],
            }),
        })

    else:
        raise ValueError(f"Unknown dataset choice: {choice}")

    print(
        f"[Dataset Loader] Ready! Total classes: {len(intent_names)} | "
        f"Train: {len(dataset['train'])} | "
        f"Validation: {len(dataset['validation'])} | "
        f"Test: {len(dataset['test'])}"
    )
    return dataset, intent_names


def train_and_evaluate(
    dataset_choice: str = DATASET_CHOICE,
    epochs: int = NUM_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    eval_only: bool = False,
    checkpoint_to_eval: str = None,
):
    """
    Main training and evaluation function.
    Executes training on RTX 4050 GPU, computes extensive metrics, and generates 5 visual plots.
    """
    set_seed(SEED)
    print("=" * 75)
    print(f"GenAI Jury 2: Intent Classification & Dialogue Optimization Engine")
    print(f"Hardware: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB | Precision: FP16={USE_FP16}")
    print("=" * 75)

    dataset, intent_names = load_and_prepare_dataset(dataset_choice, seed=SEED)
    num_labels = len(intent_names)

    id2label = {i: name for i, name in enumerate(intent_names)}
    label2id = {name: i for i, name in id2label.items()}

    # Save intent mappings to disk for inference & Streamlit
    mappings_path = REPORTS_DIR / "intent_mapping.json"
    with open(mappings_path, "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    # Tokenizer setup
    model_name = CLASSIFIER_BACKBONE
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    print("[Preprocessing] Tokenizing splits with dynamic padding support...")
    remove_cols = [c for c in dataset["train"].column_names if c != "label"]
    tokenized_train = dataset["train"].map(tokenize_function, batched=True, remove_columns=remove_cols, load_from_cache_file=True)
    tokenized_val = dataset["validation"].map(tokenize_function, batched=True, remove_columns=remove_cols, load_from_cache_file=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True, remove_columns=remove_cols, load_from_cache_file=True)

    # Model initialization
    load_path = checkpoint_to_eval if (eval_only and checkpoint_to_eval) else model_name
    print(f"[Model Init] Loading sequence classification head from: {load_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    # Compute metrics hook for Trainer
    def compute_metrics_eval(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=1)
        m = compute_comprehensive_metrics(labels, preds, probs, class_names=intent_names)
        return {
            "accuracy": m["accuracy"],
            "top3_accuracy": m["top3_accuracy"],
            "top5_accuracy": m["top5_accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "mcc": m["matthews_corrcoef"],
            "ece": m["expected_calibration_error"],
        }

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=USE_FP16,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,  # Keep disk lean on laptop
        logging_steps=50,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics_eval,
    )

    if not eval_only:
        print("\n[Trainer] Commencing training on RTX 4050 Laptop GPU...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_result = trainer.train()
        print(f"[Trainer] Training completed successfully. Train loss: {train_result.training_loss:.4f}")

        # Save model and tokenizer
        final_save_dir = CHECKPOINTS_DIR / "best_model"
        trainer.save_model(str(final_save_dir))
        tokenizer.save_pretrained(str(final_save_dir))
        print(f"[Trainer] Best model saved to: {final_save_dir}")

        # Plot training curves
        plot_training_curves(trainer.state.log_history, REPORTS_DIR / "training_validation_curves.png")

    # Final Comprehensive Evaluation on Held-Out Test Set
    print("\n[Evaluation] Running comprehensive inference on held-out test set...")
    test_output = trainer.predict(tokenized_test)
    test_logits = test_output.predictions
    test_labels = test_output.label_ids
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_preds = np.argmax(test_logits, axis=1)

    # Save validation logits for Conformal Prediction calibration
    val_output = trainer.predict(tokenized_val)
    np.save(str(REPORTS_DIR / "val_logits.npy"), val_output.predictions)
    np.save(str(REPORTS_DIR / "val_labels.npy"), val_output.label_ids)

    # Compute all detailed metrics
    test_metrics = compute_comprehensive_metrics(
        test_labels, test_preds, test_probs, class_names=intent_names
    )

    # Display Executive Results Table
    print("\n" + "=" * 65)
    print("           COMPREHENSIVE TEST SET EVALUATION METRICS")
    print("=" * 65)
    print(f"  Accuracy (Top-1)           : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']:.1%})")
    print(f"  Top-3 Accuracy             : {test_metrics['top3_accuracy']:.4f} ({test_metrics['top3_accuracy']:.1%})")
    print(f"  Top-5 Accuracy             : {test_metrics['top5_accuracy']:.4f} ({test_metrics['top5_accuracy']:.1%})")
    print(f"  Macro Precision            : {test_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall               : {test_metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score             : {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score          : {test_metrics['weighted_f1']:.4f}")
    print(f"  Matthews Corrcoef (MCC)    : {test_metrics['matthews_corrcoef']:.4f}")
    print(f"  Expected Calibration Error : {test_metrics['expected_calibration_error']:.4f}")
    print(f"  Multi-Class Brier Score    : {test_metrics['brier_score']:.4f}")
    print(f"  Cross Entropy Log Loss     : {test_metrics['cross_entropy']:.4f}")
    print("=" * 65)

    print("\nTop 5 Hardest Intents:")
    for h in test_metrics["hardest_classes"][:5]:
        print(f"  - {h['name']:<35} | F1: {h['f1']:.3f} | Support: {h['support']}")

    print("\nTop 5 Most Frequent Confusions (Actual -> Predicted):")
    for c in test_metrics["top_confusions"][:5]:
        print(f"  - {c['actual']}  -->  {c['predicted']}: {c['count']} errors")

    # Generate all visual plots
    print("\n[Visualizer] Generating high-resolution metric visualization plots...")
    plot_confusion_matrix(test_metrics["confusion_matrix"], intent_names, REPORTS_DIR / "confusion_matrix_heatmap.png")
    plot_calibration_curve(test_labels, test_probs, test_metrics["expected_calibration_error"], REPORTS_DIR / "calibration_reliability_diagram.png")
    plot_class_performance(test_metrics["class_performance"], REPORTS_DIR / "class_performance_distribution.png")
    plot_metrics_summary(test_metrics, REPORTS_DIR / "metrics_summary.png")

    # Save metrics JSON to disk
    json_metrics = {k: v for k, v in test_metrics.items() if k not in ["confusion_matrix"]}
    with open(REPORTS_DIR / "test_metrics_summary.json", "w") as f:
        json.dump(json_metrics, f, indent=2)

    print(f"[Visualizer] All 5 visualization plots saved to directory: {REPORTS_DIR}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Classifier Training & Comprehensive Metric Suite")
    parser.add_argument("--dataset", choices=["banking200k", "massive", "banking77", "merged"], default=DATASET_CHOICE, help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE, help="Per-device batch size")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate an existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path to evaluate")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_choice=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_only=args.eval_only,
        checkpoint_to_eval=args.checkpoint,
    )
