import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from ArTS_backup.utils.eval_qwk import Evaluator
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 for AES")
    parser.add_argument('--max_tgt_len', type=int, default=162, help='max target length')
    parser.add_argument('--test_batch', type=int, default=2, help='test_batch')
    parser.add_argument('--model_path', type=str, default='results/checkpoint-5190/', help='model checkpoint directory name')
    parser.add_argument('--data_path', type=str, default='data', help='data directory name')
    parser.add_argument('--output_path', type=str, default='output', help='output directory name')
    
    
    args = parser.parse_args()
    
    test_set_list = load_dataset("csv", data_files=args.data_path+"/total_test.csv", split=[f'train[{k}%:{k+20}%]' for k in range(0, 100, 20)])

    # Initialize the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
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

    test_set_list = [test_set.map(preprocess_function, batched=True) for test_set in test_set_list]

    model = model.cuda()

    def test(tokenizer, device, test_set):
        #predictions = np.argmax(preds.predictions[0], axis=2)
        preds = model.generate(
                input_ids=torch.tensor(test_set["input_ids"]).to(device),
                attention_mask=torch.tensor(test_set["attention_mask"]).to(device),
                max_length=args.max_tgt_len
            )
        
        predictions = preds
        label_ids = test_set['labels']
        assert len(predictions)==len(test_set)
        assert len(predictions)==len(label_ids)

        predictions = torch.tensor(predictions).to(device, dtype = torch.long)
        label_ids = torch.tensor(label_ids).to(device, dtype = torch.long)

        results = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in predictions]
        actuals = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in label_ids]
        return results, actuals
    
    predictions = []
    targets = []
    for list in test_set_list:
        pred, target = test(tokenizer, 'cuda', list)
        predictions.extend(pred)
        targets.extend(target)
        
    outputs = pd.DataFrame({'predictions':predictions, 'targets':targets})
    outputs.to_csv(args.output_path + '.csv')
    
    qwk_results = [] 
    over_range = []
    trait_1 = ['overall','content','word choice','organization','sentence fluency','conventions']
    trait_2 = ['overall','content','prompt adherence','language','narrativity']
    trait_3 = ['overall','content','organization','conventions','style']
    trait_4 = ['overall','content','word choice','organization','sentence fluency','conventions', 'voice']
    
    trait_sets = [trait_1,trait_1,trait_2,trait_2,trait_2,trait_2,trait_3,trait_4]
    test_target = pd.read_csv(args.data_path+"/total_test.csv")
    total_data = pd.DataFrame({'pred': predictions, 'target':targets, 'prompt':test_target['prompt_id']})
    
    for i in range(8):
        print('Prompt ', i+1, ' results!')
        traits = trait_sets[i]
        QWK_EVAL = Evaluator(traits)
        min_max = QWK_EVAL.get_min_max_scores()
        preds = total_data[total_data['prompt']==(i+1)]['pred'].reset_index()['pred']
        tars = total_data[total_data['prompt']==(i+1)]['target'].reset_index()['target']
        result = QWK_EVAL.evaluate_notnull(preds, tars)
        pred_df = QWK_EVAL.read_results(preds)
        over = {}
        for t in traits:
            df = pred_df[(pred_df[t]>min_max[i+1][t][1]) | (pred_df[t]<min_max[i+1][t][0])]
            over[t]=df
        qwk_results.append(result)
        over_range.append(over)
        #over_range[i+1]=over
        print(result)
        
    pd.DataFrame(over_range).to_csv(args.model_path + '/over_range.csv')
    pd.DataFrame(qwk_results).to_csv(args.model_path + '/qwk_results.csv')

if __name__=='__main__':
    main()
