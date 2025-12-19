
import torch
import numpy as np
import os
import sys
import subprocess
import pandas as pd
import zipfile
import requests
import io
import inspect


# --- АВТОМАТИЧЕСКАЯ УСТАНОВКА ЗАВИСИМОСТЕЙ ---
def install_and_upgrade():
    print("⏳ Обновляем библиотеки до SOTA версий... Это займет минуту.")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "datasets", "transformers", "accelerate>=0.21.0",
         "scikit-learn", "pandas"])
    print("✅ Библиотеки обновлены! Если возникнут ошибки импорта, перезапустите среду (Runtime -> Restart Session).")


# Вызываем установку принудительно, чтобы избежать конфликтов версий
try:
    import transformers
    # Простая проверка версии, если она слишком старая - обновляем
    from packaging import version

    if version.parse(transformers.__version__) < version.parse("4.42.0"):
        install_and_upgrade()
except ImportError:
    install_and_upgrade()

# Импорты после установки
from datasets import load_dataset, Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Отключаем WandB
os.environ["WANDB_DISABLED"] = "true"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def load_data_fallback():
    print("⚠️ HuggingFace load failed. Trying direct download from UCI Repository...")
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['label', 'sms'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    print("✅ Данные успешно загружены через Fallback!")
    return dataset


def main():
    model_name = "distilbert-base-uncased"

    # Проверка GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU найден: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️ GPU НЕ НАЙДЕН! Обучение будет очень медленным.")
        print("Включи GPU: Runtime -> Change runtime type -> T4 GPU")

    # --- ЗАГРУЗКА ДАННЫХ ---
    try:
        dataset = load_dataset("sms_spam", split="train")
        dataset = dataset.train_test_split(test_size=0.2)
        print("✅ Данные загружены через HuggingFace.")
    except Exception as e:
        print(f"Ошибка HF: {e}")
        dataset = load_data_fallback()

    # --- ТОКЕНИЗАЦИЯ ---
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["sms"], truncation=True, padding=False)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- МОДЕЛЬ ---
    id2label = {0: "HAM (Норм)", 1: "SPAM (Спам)"}
    label2id = {"HAM (Норм)": 0, "SPAM (Спам)": 1}

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # --- ПАРАМЕТРЫ ---
    init_args = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "eval_strategy" if "eval_strategy" in init_args else "evaluation_strategy"
    print(f"🔧 Используем аргумент стратегии: {eval_strategy_key}")

    # Собираем аргументы в словарь
    args_dict = {
        "output_dir": "./results",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 2,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "push_to_hub": False,
        "report_to": "none",
        eval_strategy_key: "epoch"  # Подставляем правильный ключ динамически
    }

    training_args = TrainingArguments(**args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 Начинаем обучение...")
    trainer.train()

    save_path = "./my_spam_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"🏁 Готово! Модель сохранена в папку {save_path}.")


if __name__ == "__main__":
    main()


