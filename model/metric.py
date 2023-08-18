import torch

def load_metric(type, predictions, label):
    if type == 'accuracy':
        return accuracy(predictions, label)
    elif type == 'bce_acc':
        return bce_accuracy(predictions, label)
    else:
        return top_k_acc(predictions, label)

def bce_accuracy(final_logit, label):
    with torch.no_grad():
        pred = torch.sigmoid(final_logit) >= 0.5
        assert pred.shape[0] == len(label)
        correct = 0
        correct += torch.sum(pred == label).item()
    return correct / len(label)


def accuracy(final_logit, label):
    with torch.no_grad():
        pred = torch.argmax(torch.softmax(final_logit, -1), dim=1)
        assert pred.shape[0] == len(label)
        correct = 0
        correct += torch.sum(pred == label).item()
    return correct / len(label)


def top_k_acc(output, label, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(label)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == label).item()
    return correct / len(label)
