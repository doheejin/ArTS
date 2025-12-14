from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from eval_qwk import Evaluator
import argparse
import wandb

# Load the ASAP dataset into a pandas DataFrame
def main():
    parser = argparse.ArgumentParser(description="Train T5 for AES")
    parser.add_argument('--max_tgt_len', type=int, default=162, help='max target length')
    parser.add_argument('--train_batch', type=int, default=2, help='train_batch')
    parser.add_argument('--valid_batch', type=int, default=2, help='valid_batch')
    parser.add_argument('--test_batch', type=int, default=2, help='test_batch')
    parser.add_argument('--output_path', type=str, default='./results', help='output_dir name')
    parser.add_argument('--data_path', type=str, default='data', help='data_dir name')
    parser.add_argument('--epoch', type=int, default=3, help='train epoch')
    parser.add_argument('--steps', type=int, default=3000, help='train and eval steps')
    parser.add_argument('--model', type=str, default='t5-base', help='pre-trained model')
    
    args = parser.parse_args()
    
    id = wandb.util.generate_id()
    print(id)
    wandb.init(project='asap', entity='heejindo', id=id, resume=True)
    
    dataset = load_dataset("csv", data_files=args.data_path+"/total_train.csv")
    val_set = load_dataset("csv", data_files=args.data_path+"/total_dev.csv")
    dataset['valid'] = val_set['train']

    # Initialize the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    #tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        inputs, targets, prompts = [], [], []
        for i in range(len(examples['content_text'])):
            if examples['content_text'][i] and examples['target'][i]:
                inputs.append(examples['content_text'][i])
                targets.append(str(examples['target'][i]))
                prompts.append(examples['prompt_id'][i])

        inputs = ["score the essay of the prompt " + str(prompts[i]) + ": " + inp for (i,inp) in enumerate(inputs)]
        model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_tgt_len, padding='max_length', truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    training_set = dataset['train'].map(preprocess_function, batched=True)
    validation_set = dataset['valid'].map(preprocess_function, batched=True)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.valid_batch,
        warmup_steps=3, # default=5 # EACL - 3
        #weight_decay=0.01,
        logging_dir='./logs',
        #logging_steps=1,
        eval_steps=args.steps,
        save_steps=args.steps,
        evaluation_strategy='steps', # steps
        save_strategy = "steps", # 'steps' 
        load_best_model_at_end=True,
        generation_max_length=args.max_tgt_len,
        predict_with_generate=True,
        save_total_limit = 2
    )
    
    # Define the trainer for training the T5 model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=validation_set,
        tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        #compute_metrics=compute_metrics(tokenizer=tokenizer),
    )
    
    # Train the T5 model
    trainer.train()

    trainer.evaluate()
    
    wandb.finish()

if __name__=='__main__':
    main()
