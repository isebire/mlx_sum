from sklearn.metrics import roc_auc_score
import pandas
import scikitplot as skplt
import matplotlib.pyplot as plt

df = pandas.read_csv('bert_labels_test_output.csv')
auroc = roc_auc_score(df['label'], df['+ve confidence'])
print(auroc)
roc_confidences = []
for i in list(df['+ve confidence']):
    roc_confidences.append([1-i, i])
skplt.metrics.plot_roc_curve(list(df['label']), roc_confidences)
plt.show()
