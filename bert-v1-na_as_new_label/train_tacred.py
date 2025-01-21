import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment
from encoder import EncodingModel
# import wandb

from transformers import BertTokenizer
from losses import MutualInformationLoss, HardSoftMarginTripletLoss, HardMarginLoss
from sklearn.metrics import f1_score

class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(1536, 256).to(config.device)
        self.fc2 = nn.Linear(256, 1).to(config.device)
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist
    def _cosine_similarity(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        cos = nn.CosineSimilarity(dim=1)
        sim = []
        for i in range(b):
            sim_i = cos(x2, x1[i])
            sim.append(torch.unsqueeze(sim_i, 0))
        sim = torch.cat(sim, 0)
        return sim
    

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        

    def train_model(self, encoder, training_data, seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True, training=True if not is_memory else False)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        # softmax = nn.Softmax(dim=0)
        soft_margin_loss = HardSoftMarginTripletLoss()

        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)

                batch_instance = {'ids': [], 'mask': []} 

                batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)

                # calculate loss factors per label
                label_weights = torch.ones(len(labels)).to(self.config.device)
                unique_labels, label_counts = torch.unique(labels, return_counts=True)
                label_weights = 1.0 / label_counts[torch.searchsorted(unique_labels, labels)]
                label_weights = label_weights / label_weights.sum() * len(labels)
                label_weights = label_weights.to(self.config.device) # (b)


                n = len(labels)
                new_matrix_labels = np.zeros((n, n), dtype=float)

                # Fill the matrix according to the label comparison
                for i1 in range(n):
                    for j in range(n):
                        if labels[i1] == labels[j]:
                            new_matrix_labels[i1][j] = 1.0

                new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)

                # labels tensor shape b*b
                
                
                hidden = encoder(instance) # b, dim
                loss1 = self.moment.contrastive_loss(hidden, labels, is_memory)
                labels_des = encoder(batch_instance, is_des = True) # b, dim
                rd = encoder(instance, is_rd=True) # b, dim

                # compute hard margin contrastive loss
                hard_margin_loss = HardMarginLoss()
                loss3 = []
                for idx in range(len(labels)):
                    rep_des = labels_des[idx]
                    label_for_loss = labels == labels[idx]
                    loss3.append(hard_margin_loss(rep_des, hidden, label_for_loss))
                loss3 = torch.stack(loss3).mean()

                # compute hard margin constrastive loss for relation description: rd vs hidden
                hard_margin_loss_1 = HardMarginLoss()
                loss3_1 = []
                for idx in range(len(labels)):
                    rep_rd = rd[idx]
                    label_for_loss = labels == labels[idx]
                    loss3_1.append(hard_margin_loss_1(rep_rd, hidden, label_for_loss))
                loss3_1 = torch.stack(loss3_1).mean()

                # compute hard margin constrastive loss for relation description: rd vs labels_des
                hard_margin_loss_2 = HardMarginLoss()
                loss3_2 = []
                for idx in range(len(labels)):
                    rep_des = labels_des[idx]
                    label_for_loss = labels == labels[idx]
                    loss3_2.append(hard_margin_loss_2(rep_des, rd, label_for_loss))
                loss3_2 = torch.stack(loss3_2).mean()


                loss_retrieval = MutualInformationLoss(weights=label_weights)
                loss2 = loss_retrieval(hidden, labels_des, new_matrix_labels_tensor)

                loss_retrieval_1 = MutualInformationLoss(weights=label_weights)
                loss2_1 = loss_retrieval_1(hidden, rd, new_matrix_labels_tensor)

                loss_retrieval_2 = MutualInformationLoss(weights=label_weights)
                loss2_2 = loss_retrieval_2(rd, labels_des, new_matrix_labels_tensor)


                # compute soft margin triplet loss
                uniquie_labels = labels.unique()
                if len(uniquie_labels) > 1:
                    loss4 = soft_margin_loss(hidden, labels.to(self.config.device))
                else:
                    loss4 = 0.0

                # print(f"loss1: {loss1}, loss2: {loss2}, loss3: {loss3}, loss4: {loss4}")

                # loss = 1*loss1 + 2*loss2 + 0.5*loss3 + 1*loss4
                loss = 1*loss1 + 2*loss2 + 0.5*loss3 + 1*loss4 + 0.5*loss3_1 + 0.5*loss3_2 + 0.5*loss2_1 + 0.5*loss2_2
            
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')             

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total

    def f1(self, preds, labels):
        unique_labels = set(preds + labels)
        if self.config.na_id in unique_labels:
            unique_labels.remove(self.config.na_id)
        
        unique_labels = list(unique_labels)

        # Calculate F1 score for each class separately
        f1_per_class = f1_score(preds, labels, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(preds, labels, average='micro', labels=unique_labels)

        # Calculate macro-average F1 score
        # f1_macro = f1_score(preds, labels, average='macro', labels=unique_labels)

        # Calculate weighted-average F1 score
        f1_weighted = f1_score(preds, labels, average='weighted', labels=unique_labels)

        print("F1 score per class:", dict(zip(unique_labels, f1_per_class)))
        print("Micro-average F1 score:", f1_micro)
        # print("Macro-average F1 score:", f1_macro)
        print("Weighted-average F1 score:", f1_weighted)

        return f1_micro

    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        """
        Args:
            encoder: Encoder
            seen_proto: seen prototypes. NxH tensor
            seen_relid: relation id of protoytpes
            test_data: test data
            rep_des: representation of seen relation description. N x H tensor

        Returns:

        """
        batch_size = 48
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        preds = []
        labels = []
        corrects = 0.0

        preds1 = []
        corrects1 = 0.0

        preds2 = []
        corrects2 = 0.0

        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations
            logits_des = self._cosine_similarity(fea, rep_des)  # (B, N)
            # combine using rrf
            rrf_k = 60
            
            logits_ranks = torch.argsort(torch.argsort(-logits, dim=1), dim=1) + 1
            logits_des_ranks = torch.argsort(torch.argsort(-logits_des, dim=1), dim=1) + 1
            rrf_logits = 0.4 / (rrf_k + logits_ranks)
            rrf_logits_des = 0.6 / (rrf_k + logits_des_ranks)
            logits_rrf = rrf_logits + rrf_logits_des
           
            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            preds.extend(pred)
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size

            # by logits_des
            cur_index1 = torch.argmax(logits_des,dim=1)
            pred1 = []
            for i in range(cur_index1.size()[0]):
                pred1.append(seen_relid[int(cur_index1[i])])
            preds1.extend(pred1)
            pred1 = torch.tensor(pred1)
            correct1 = torch.eq(pred1, label).sum().item()
            acc1 = correct1/ batch_size
            corrects1 += correct1

            # by rrf
            cur_index2 = torch.argmax(logits_rrf,dim=1)
            pred2 = []
            for i in range(cur_index2.size()[0]):
                pred2.append(seen_relid[int(cur_index2[i])])
            preds2.extend(pred2)
            pred2 = torch.tensor(pred2)
            correct2 = torch.eq(pred2, label).sum().item()
            acc2 = correct2/ batch_size
            corrects2 += correct2

            labels.extend(label.cpu().tolist())

            # sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
            #                  .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            # sys.stdout.write('[EVAL DES] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
            #                  .format(batch_num, 100 * acc1, 100 * (corrects1 / total)) + '\r')
            # sys.stdout.write('[EVAL RRF] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
            #                  .format(batch_num, 100 * acc2, 100 * (corrects2 / total)) + '\r')
            # sys.stdout.flush()
        print('')
        # return corrects / total, corrects1 / total, corrects2 / total
        return self.f1(preds, labels), self.f1(preds1, labels), self.f1(preds2, labels)

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number wo na
        cur_acc_wo_na, total_acc_wo_na = [], []
        cur_acc1_wo_na, total_acc1_wo_na = [], []
        cur_acc2_wo_na, total_acc2_wo_na = [], []


        cur_acc_num_wo_na, total_acc_num_wo_na = [], []
        cur_acc_num1_wo_na, total_acc_num1_wo_na = [], []
        cur_acc_num2_wo_na, total_acc_num2_wo_na = [], []

        # step is continual task number w na
        cur_acc_w_na, total_acc_w_na = [], []
        cur_acc1_w_na, total_acc1_w_na = [], []
        cur_acc2_w_na, total_acc2_w_na = [], []


        cur_acc_num_w_na, total_acc_num_w_na = [], []
        cur_acc_num1_w_na, total_acc_num1_w_na = [], []
        cur_acc_num2_w_na, total_acc_num2_w_na = [], []


        memory_samples = {}
        data_generation = []
        seen_des = {}


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])


        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

            for rel in current_relations:
                ids = self.tokenizer.encode(seen_descriptions[rel][0],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.config.max_length)     
                # mask
                mask = np.zeros(self.config.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                mask[:end_index + 1] = 1 
                if rel not in seen_des:
                    seen_des[rel] = {}
                    seen_des[rel]['ids'] = ids
                    seen_des[rel]['mask'] = mask

            print(f"seen_des: {seen_des.keys()}")

            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize, seen_des)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True) 
                self.train_model(encoder, memory_data_initialize, seen_des, is_memory=True)

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])


            # get representation of seen description
            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]
            list_seen_des = []
            for i in range(len(seen_proto)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            rep_des = []
            for i in range(len(list_seen_des)):
                sample = {
                    'ids' : torch.tensor([list_seen_des[i]['ids']]).to(self.config.device),
                    'mask' : torch.tensor([list_seen_des[i]['mask']]).to(self.config.device)
                }
                hidden = encoder(sample, is_des=True)
                hidden = hidden.detach().cpu().data
                rep_des.append(hidden)
            rep_des = torch.cat(rep_des, dim=0)

            # Eval current task and history task wo na
            test_data_initialize_cur_wo_na, test_data_initialize_seen_wo_na = [], []
            for rel in current_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_cur_wo_na += test_data[rel]
                
            for rel in seen_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_seen_wo_na += historic_test_data[rel]
            
            ac1_wo_na, ac1_des_wo_na, ac1_rrf_wo_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_wo_na,rep_des)
            ac2_wo_na, ac2_des_wo_na, ac2_rrf_wo_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_wo_na,rep_des)

            # Eval current task and history task w na
            test_data_initialize_cur_w_na, test_data_initialize_seen_w_na = [], []
            for rel in current_relations:
                test_data_initialize_cur_w_na += test_data[rel]
                
            for rel in seen_relations:
                test_data_initialize_seen_w_na += historic_test_data[rel]
            
            ac1_w_na, ac1_des_w_na, ac1_rrf_w_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_w_na,rep_des)
            ac2_w_na, ac2_des_w_na, ac2_rrf_w_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_w_na,rep_des)
            
            # wo na
            cur_acc_num_wo_na.append(ac1_wo_na)
            total_acc_num_wo_na.append(ac2_wo_na)
            cur_acc_wo_na.append('{:.4f}'.format(ac1_wo_na))
            total_acc_wo_na.append('{:.4f}'.format(ac2_wo_na))
            print('cur_acc_wo_na: ', cur_acc_wo_na)
            print('his_acc_wo_na: ', total_acc_wo_na)

            cur_acc_num1_wo_na.append(ac1_des_wo_na)
            total_acc_num1_wo_na.append(ac2_des_wo_na)
            cur_acc1_wo_na.append('{:.4f}'.format(ac1_des_wo_na))
            total_acc1_wo_na.append('{:.4f}'.format(ac2_des_wo_na))
            print('cur_acc des_wo_na: ', cur_acc1_wo_na)
            print('his_acc des_wo_na: ', total_acc1_wo_na)

            cur_acc_num2_wo_na.append(ac1_rrf_wo_na)
            total_acc_num2_wo_na.append(ac2_rrf_wo_na)
            cur_acc2_wo_na.append('{:.4f}'.format(ac1_rrf_wo_na))
            total_acc2_wo_na.append('{:.4f}'.format(ac2_rrf_wo_na))
            print('cur_acc rrf_wo_na: ', cur_acc2_wo_na)
            print('his_acc rrf_wo_na: ', total_acc2_wo_na)

            # w na
            cur_acc_num_w_na.append(ac1_w_na)
            total_acc_num_w_na.append(ac2_w_na)
            cur_acc_w_na.append('{:.4f}'.format(ac1_w_na))
            total_acc_w_na.append('{:.4f}'.format(ac2_w_na))
            print('cur_acc_w_na: ', cur_acc_w_na)
            print('his_acc_w_na: ', total_acc_w_na)

            cur_acc_num1_w_na.append(ac1_des_w_na)
            total_acc_num1_w_na.append(ac2_des_w_na)
            cur_acc1_w_na.append('{:.4f}'.format(ac1_des_w_na))
            total_acc1_w_na.append('{:.4f}'.format(ac2_des_w_na))
            print('cur_acc des_w_na: ', cur_acc1_w_na)
            print('his_acc des_w_na: ', total_acc1_w_na)

            cur_acc_num2_w_na.append(ac1_rrf_w_na)
            total_acc_num2_w_na.append(ac2_rrf_w_na)
            cur_acc2_w_na.append('{:.4f}'.format(ac1_rrf_w_na))
            total_acc2_w_na.append('{:.4f}'.format(ac2_rrf_w_na))
            print('cur_acc rrf_w_na: ', cur_acc2_w_na)
            print('his_acc rrf_w_na: ', total_acc2_w_na)


        torch.cuda.empty_cache()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        return (total_acc_num_wo_na, total_acc_num1_wo_na, total_acc_num2_wo_na), (total_acc_num_w_na, total_acc_num1_w_na, total_acc_num2_w_na)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.na_id = 80
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description_detail_3.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.na_id = 41
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description_detail_3.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'

    config.majority_label = config.na_id

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    # wo na
    acc_list_wo_na = []
    acc_list1_wo_na = []
    aac_list2_wo_na = []

    # w na
    acc_list_w_na = []
    acc_list1_w_na = []
    aac_list2_w_na = []

    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        (acc_wo_na, acc1_wo_na, aac2_wo_na), (acc_w_na, acc1_w_na, aac2_w_na) = manager.train()
        
        # wo na
        acc_list_wo_na.append(acc_wo_na)
        acc_list1_wo_na.append(acc1_wo_na)
        aac_list2_wo_na.append(aac2_wo_na)

        # w na
        acc_list_w_na.append(acc_w_na)
        acc_list1_w_na.append(acc1_w_na)
        aac_list2_w_na.append(aac2_w_na)

        torch.cuda.empty_cache()
    
    # wo na
    accs_wo_na = np.array(acc_list_wo_na)
    ave_wo_na = np.mean(accs_wo_na, axis=0)
    print('----------END')
    print('his_acc mean_wo_na: ', np.around(ave_wo_na, 4))
    accs1_wo_na = np.array(acc_list1_wo_na)
    ave1_wo_na = np.mean(accs1_wo_na, axis=0)
    print('his_acc des mean_wo_na: ', np.around(ave1_wo_na, 4))
    accs2_wo_na = np.array(aac_list2_wo_na)
    ave2_wo_na = np.mean(accs2_wo_na, axis=0)
    print('his_acc rrf mean_wo_na: ', np.around(ave2_wo_na, 4))

    # w na
    accs_w_na = np.array(acc_list_w_na)
    ave_w_na = np.mean(accs_w_na, axis=0)
    print('his_acc mean_w_na: ', np.around(ave_w_na, 4))
    accs1_w_na = np.array(acc_list1_w_na)
    ave1_w_na = np.mean(accs1_w_na, axis=0)
    print('his_acc des mean_w_na: ', np.around(ave1_w_na, 4))
    accs2_w_na = np.array(aac_list2_w_na)
    ave2_w_na = np.mean(accs2_w_na, axis=0)
    print('his_acc rrf mean_w_na: ', np.around(ave2_w_na, 4))
    
    