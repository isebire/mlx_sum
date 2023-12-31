import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os, sys
from data_utils import readData, flat_accuracy, save_plots_models
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.abspath('..'))
os.environ["PYTHONIOENCODING"] = "utf-8"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='If set, use the GPU')
    parser.add_argument('--modelFile', type=str, default='',
                        help='Folder path to get model')
    parser.add_argument('--testFile', type=str, default='',
                        help='Folder path to read testing input')
    parser.add_argument('--outputDir', type=str, default='',
                        help='Folder path to write output to')
    args = parser.parse_args()

outDir = args.outputDir
modelPath = args.modelFile
testFile = args.testFile

print('outDir: {0}, modelPath: {1}, testFile: {2}'.format(outDir, modelPath, testFile))

if os.path.exists(outDir):
    filelist = [f for f in os.listdir(outDir)]
    for f in filelist:
        os.remove(os.path.join(outDir, f))
else:
    os.makedirs(outDir)

model_state_dict = torch.load(modelPath)
bert_model = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=2)
print("Fine-tuned model loaded with labels = ", model.num_labels)

model.cuda()
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
print('Device', device)

# testing
input_ids, labels, attention_masks, dataTypeId, fileId, context_id = readData(tokenizer, args, mode="test")

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)
prediction_dataTypeId = torch.tensor(dataTypeId)
prediction_fileId = torch.tensor(fileId)
prediction_context_id = torch.tensor(context_id)

batch_size = 32
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels, prediction_dataTypeId, prediction_fileId, prediction_context_id)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model.eval()

# Tracking variables
predictions, true_labels = [], []
nb_eval_steps = 0
eval_accuracy = 0
csv_output = []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels, b_dataTypeId, b_fileId, b_context_id = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    # shape (batch_size, config.num_labels)
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    count = 0
    for i in range(pred_flat.shape[0]):
        # iterate over the batch
        csv_output.append((b_input_ids[i], pred_flat[i], labels_flat[i], b_dataTypeId[i], b_fileId[i], b_context_id[i]))

print('Test Accuracy Accuracy: {0:0.4f}'.format((float(eval_accuracy) / float(nb_eval_steps))))

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

precision = precision_score(flat_true_labels, flat_predictions)
recall = recall_score(flat_true_labels, flat_predictions)
f1 = f1_score(flat_true_labels, flat_predictions)
print('Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(recall, precision, f1))

testFile = args.testFile.split('/')
testFile = testFile[len(testFile) - 1]  # take the last part

outFile = outDir + "/" + testFile[:len(testFile) - 4] + '_output.txt'
file = open(outFile, 'w')
print('Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(recall, precision, f1), file=file)
print('Saving scores to: ', outFile)

headings = ['context', 'predicted', 'label', 'dataTypeId', 'fileId', 'context_id']
df = pd.DataFrame(columns=headings)
for ids, pred, label, dataTypeId, fileId, context_id in csv_output:
    ids = np.trim_zeros(ids.cpu().numpy())
    sentence = tokenizer.convert_ids_to_tokens(ids)[1:-1]
    dataTypeId = dataTypeId.cpu().numpy()
    fileId = fileId.cpu().numpy()
    context_id = context_id.cpu().numpy()
    # data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
    data = [{'context': ' '.join(sentence), 'predicted': str(pred), 'label': str(label), 'dataTypeId': str(dataTypeId),
             'fileId': str(fileId), 'context_id': str(context_id)}]
    df = df.append(pd.DataFrame(data, columns=headings))


outFile = outDir + "/" + testFile[:len(testFile) - 4] + '_output.csv'
df.to_csv(outFile, index=False)
print('Saving output file to: ', outFile)

print('Process Completed')
