import torch
import random
import numpy as np
import torch.nn.functional as F
from data_loader import get_data_loader_BERT

class Moment:
    def __init__(self, config) -> None:
        self.config = config
        self.features = None
        self.labels = None
        self.mem_samples = None
        self.mem_features = None
        self.mem_labels = None
        self.sample_k = config.sample_k
        self.temperature = config.contrastive_temp
        self.m = config.margin
    
    def init_moment(self, encoder, dataset, is_memory=False):
        encoder.eval()
        datalen = len(dataset)
        if not is_memory:
            self.features = torch.zeros(datalen, self.config.encoder_output_size)

            data_loader = get_data_loader_BERT(self.config, dataset) # shuffle=False
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                hidden  = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea)
                lbs.append(labels) # shuffle=False
            lbs = torch.cat(lbs)
            self.labels = lbs
        else:
            self.mem_samples = dataset
            self.mem_features = torch.zeros(datalen, self.config.encoder_output_size)

            data_loader = get_data_loader_BERT(self.config, dataset) # shuffle=False
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                hidden  = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea, is_memory)
                lbs.append(labels) # shuffle=False
            lbs = torch.cat(lbs)
            self.mem_labels = lbs            

    def update(self, ind, feature, is_memory=False):
        if not is_memory:
            self.features[ind] = feature
        else:
            self.mem_features[ind] = feature
    
    def update_allmem(self, encoder):
            data_loader = get_data_loader_BERT(self.config, self.mem_samples, batch_size=64) # shuffle=False
            for step, (instance, labels, ind) in enumerate(data_loader):
                hidden  = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea, is_memory=True)
        

    def get_mem_proto(self):
        cinds = []
        for x in self.mem_labels:
            if x.item() not in cinds:
                cinds.append(x.item())

        num = len(cinds)
        feats = self.mem_features
        centroids = torch.zeros((num, feats.size(1)), dtype=torch.float32, device=feats.device)
        for i, c in enumerate(cinds):
            ind = np.where(self.mem_labels.cpu().numpy() == c)[0]
            centroids[i, :] = feats[ind, :].mean(dim=0)
        return centroids

    def contrastive_loss(self, x, labels, is_memory=False):
        '''
        x (B, H)
        '''
        if is_memory:
            ct_x = self.mem_features.to(self.config.device)
            ct_y = self.mem_labels
        else:
            idx = list(range(len(self.features)))
            if len(idx) > self.sample_k:
                sample_id = random.sample(idx, self.sample_k)
            else:  # sample number > total feature
                sample_id = idx
            ct_x = self.features[sample_id].to(self.config.device)  # (N, H)
            ct_y = self.labels[sample_id]  # (N)

        # l2 normalize
        x = F.normalize(x, p=2, dim=1) # (B, H)
        ct_x = F.normalize(ct_x, p=2, dim=1) # (N, H)
        t1 = torch.mm(x, ct_x.T) + 1  # 0 <= cos + 1 <= 2


        mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
        mask_combined_neg = ~mask_combined_pos

        # Calculate beta for negative samples
        exp_t1_neg = torch.exp(t1 / self.temperature) * mask_combined_neg
        sum_exp_t1_neg = torch.sum(exp_t1_neg, dim=1, keepdim=True) + 1e-6

        exp_t1_pos = torch.exp(t1 / self.temperature) * mask_combined_pos
        sum_exp_t1_pos = torch.sum(exp_t1_pos, dim=1, keepdim=True) + 1e-6

        cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)

        sum_temp = sum_exp_t1_pos + sum_exp_t1_neg
        log_prob = -torch.log(sum_exp_t1_pos / sum_temp)
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos,
                                                           dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
