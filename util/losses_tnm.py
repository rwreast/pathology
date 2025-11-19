import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        # print(target.shape)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / (self.n_classes)


class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self, config=None):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = 0
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        mask = mask.cuda()
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / (torch.sum(mask, dim=0)+ 0.000001)

        log_loss = torch.log(log_loss).reshape(-1, 1)

        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / (torch.sum(e) + 0.000001)


        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='elementwise_mean'): ###
        
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):

        # pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt + 1e-5) ** self.gamma * target * torch.log(pt + 1e-5) - (1 - alpha) * (pt + 1e-5) ** self.gamma * (1 - target) * torch.log(1 - pt + 1e-5)
        # loss = - alpha ** (1 - pt)  * target * torch.log(pt) -  (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
    
import numpy as np
import torch
import pandas as pd

class SurvivalAnalysis(object):
    """ This class contains methods used in survival analysis.
    """

    def c_index(self, risk, T, C):
        """Calculate concordance index to evaluate model prediction.

        C-index calulates the fraction of all pairs of subjects whose predicted
        survival times are correctly ordered among all subjects that can actually
		be ordered, i.e. both of them are uncensored or the uncensored time of
		one is smaller than the censored survival time of the other.

        Parameters
        ----------
        risk: numpy.ndarray
           m sized array of predicted risk (do not confuse with predicted survival time)
        T: numpy.ndarray
           m sized vector of time of death or last follow up
        C: numpy.ndarray
           m sized vector of censored status (do not confuse with observed status)

        Returns
        -------
        A value between 0 and 1 indicating concordance index.
        """
        n_orderable = 0.0 +0.000001
        score = 0.0
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                if (C[i] == 0 and C[j] == 0):
                    n_orderable = n_orderable + 1
                    if (T[i] > T[j]):
                        if (risk[j] > risk[i]):
                            score = score + 1
                    elif (T[j] > T[i]):
                        if (risk[i] > risk[j]):
                            score = score + 1
                    else:
                        if (risk[i] == risk[j]):
                            score = score + 1
                elif (C[i] == 1 and C[j] == 0):
                    if (T[i] >= T[j]):
                        n_orderable = n_orderable + 1
                        if (T[i] > T[j]):
                            if (risk[j] > risk[i]):
                                score = score + 1
                elif (C[j] == 1 and C[i] == 0):
                    if (T[j] >= T[i]):
                        n_orderable = n_orderable + 1
                        if (T[j] > T[i]):
                            if (risk[i] > risk[j]):
                                score = score + 1

        # print score to screen
        return score / n_orderable

    def neg_partial_loglik(self, preds, events, times):
        batch_size = len(preds)
        risk_set = np.zeros([batch_size, batch_size], dtype=int)
        for i in range(batch_size):
            for j in range(batch_size):
                risk_set[i, j] = times[j] >= times[i]
        risk_set = torch.FloatTensor(risk_set).cuda()
        events = torch.FloatTensor(events).cuda()

        theta = preds.reshape(-1)
        exp_theta = torch.exp(theta)

        loss = - torch.mean((theta - torch.log(torch.sum(exp_theta * risk_set, dim=1))) * events)
        return loss

    def calc_at_risk(self, X, T, O, img_names=[]):
        """Calculate the at risk group of all patients.

		For every patient i, this function returns the index of the first
		patient who died after i, after sorting the patients w.r.t. time of death.
        Refer to the definition of Cox proportional hazards log likelihood for
		details: https://goo.gl/k4TsEM

        Parameters
        ----------
        X: numpy.ndarray
           m*n matrix of input data
        T: numpy.ndarray
           m sized vector of time of death
        O: numpy.ndarray
           m sized vector of observed status (1 - censoring status)

        Returns
        -------
        X: numpy.ndarray
           m*n matrix of input data sorted w.r.t time of death
        T: numpy.ndarray
           m sized sorted vector of time of death
        O: numpy.ndarray
           m sized vector of observed status sorted w.r.t time of death
        at_risk: numpy.ndarray
           m sized vector of starting index of risk groups
        """
        # T = torch.stack(T).detach().cpu().numpy()
        # O = torch.stack(O).detach().cpu().numpy()
        # X = torch.stack(X)
        T = torch.squeeze(T, dim=1).detach().cpu().numpy()
        O = torch.squeeze(O, dim=1).detach().cpu().numpy()
        df1 = pd.DataFrame({'T': T, 'O': O})
        df1.sort_values(['T', 'O'], ascending=[False, True], inplace=True)
        sort_idx = list(df1.index)
        X = X[sort_idx]
        O = O[sort_idx]
        T = T[sort_idx]
        if len(img_names) != 0:
            img_names = img_names[sort_idx]
        failures = {}
        at_risk = {}
        n, cnt = 0, 0

        for i in range(len(O)):
            if O[i]:
                if T[i] not in failures:
                    failures[T[i].item()] = [i]
                    n += 1
                else:
                    # ties occured
                    cnt += 1
                    failures[T[i].item()].append(i)

                if T[i] not in at_risk:
                    at_risk[T[i].item()] = []
                    for j in range(0, i + 1):
                        at_risk[T[i].item()].append(j)
                else:
                    at_risk[T[i].item()].append(i)
        # when ties occured frequently
        if cnt >= n / 2 and n!=0:
            ties = 'efron'
        elif cnt > 0:
            ties = 'breslow'
        else:
            ties = 'noties'


        return X, T, O, at_risk, failures, ties, img_names
    



def cox_cost(logits, at_risk, observed, failures,ties):

    logL = 0
    # pre-calculate cumsum
    cumsum_logits = torch.cumsum(logits, dim=0).cuda()
    hazard_ratio = torch.exp(logits).cuda()
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, dim=0).cuda()
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio).cuda()

        likelihood = logits - log_risk

        uncensored_likelihood = likelihood * observed.float()

        logL = -1 * uncensored_likelihood.sum()
    else:
        # Loop for death times
            # print(failures)

            for t in failures:
                tfail = failures[t]
                trisk = at_risk[t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_logits[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_logits[tfail[0] - 1])

                if ties == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += torch.log(s) * d
                elif ties == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0] - 1])

                    for j in range(d):

                        logL += torch.log(s - j * r / d)

                else:
                    raise NotImplementedError('tie breaking method not recognized')

    return logL