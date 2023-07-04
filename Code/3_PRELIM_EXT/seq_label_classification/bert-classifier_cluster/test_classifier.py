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
# labels return None

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_dataTypeId = torch.tensor(dataTypeId)
prediction_fileId = torch.tensor(fileId)
prediction_context_id = torch.tensor(context_id)

batch_size = 32
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_dataTypeId, prediction_fileId, prediction_context_id)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model.eval()

# Tracking variables
predictions = []
csv_output = []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_dataTypeId, b_fileId, b_context_id = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    # shape (batch_size, config.num_labels)
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Store predictions and true labels
    predictions.append(logits)
    pred_flat = np.argmax(logits, axis=1).flatten()
    count = 0
    for i in range(pred_flat.shape[0]):
        # iterate over the batch
        csv_output.append((b_input_ids[i], pred_flat[i], b_dataTypeId[i], b_fileId[i], b_context_id[i]))

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

testFile = args.testFile.split('/')
testFile = testFile[len(testFile) - 1]  # take the last part

testdf = pd.read_csv(args.testFile)
headings = ['context', 'predicted', 'dataTypeId', 'fileId', 'context_id']
df = pd.DataFrame(columns=headings)
for ids, pred, dataTypeId, fileId, context_id in csv_output:
    ids = np.trim_zeros(ids.cpu().numpy())
    sentence = tokenizer.convert_ids_to_tokens(ids)[1:-1]  # tokenized sentence
    fileId = fileId.cpu().numpy()
    context_id = context_id.cpu().numpy()

    # Get un-tokenzied original sentence
    testdf['fileId'] = testdf['fileId'].astype(str)
    testdf['context_id'] = testdf['context_id'].astype(str)
    this_df = testdf.loc[(testdf['fileId'] == str(fileId)) & (testdf['context_id'] == str(context_id))]
    assert (this_df.shape[0] == 1)
    sentence = this_df['context'].values[0]

    dataTypeId = dataTypeId.cpu().numpy()

    # data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
    data = [{'context': sentence, 'predicted': str(pred), 'dataTypeId': str(dataTypeId), 'fileId': str(fileId), 'context_id': str(context_id)}]
    df = df.append(pd.DataFrame(data, columns=headings))


outFile = outDir + "/" + testFile[:len(testFile) - 4] + '_output.csv'
df.to_csv(outFile, index=False)
print('Saving output file to: ', outFile)

print('Process Completed')
