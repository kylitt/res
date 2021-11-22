import torch
from tqdm import tqdm
import numpy as np
import scipy
from scipy.stats import t
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Helper Funcitons
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def test(model,meta_test_data):

    # Shift into evaluation mode
    model.eval()
    acc = []

    # No gradient within this context
    with torch.no_grad():

        # tqdm just displays a convenient progress bar
        for idx, data in tqdm(enumerate(meta_test_data)):

            # Retrieve data and assign to default GPU
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.to('cuda')
            query_xs = query_xs.to('cuda')
            
            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
            query_xs = query_xs.view(-1, channel, height, width)
            
            support_features = model(support_xs).view(support_xs.size(0), -1)
            query_features = model(query_xs).view(query_xs.size(0), -1)
            
            support_features = normalize(support_features)
            query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()
            clf = LogisticRegression(penalty='l2',
                                        random_state=0,
                                        C=1.0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        multi_class='multinomial')
            clf.fit(support_features, support_ys)
            query_ys_pred = clf.predict(query_features)

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    acc, std = mean_confidence_interval(acc)
    print('test_acc: {:.4f}, test_std: {:.4f}'.format(acc, std))