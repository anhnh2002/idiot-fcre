import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers import RobertaModel
from transformers import BertForMaskedLM
class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
            self.lm_head = BertForMaskedLM.from_pretrained(config.bert_path).to(config.device).cls
        elif config.model == 'roberta':
            self.encoder = RobertaModel.from_pretrained(config.roberta_path).to(config.device)
            self.encoder.resize_token_embeddings(config.vocab_size)
        if config.tune == 'prompt':
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bert_word_embedding = self.encoder.get_input_embeddings()
        self.embedding_dim = self.bert_word_embedding.embedding_dim
        self.prompt_lens = config.prompt_len * config.prompt_num
        self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(config.device)
        # initialize prompt embedding
        self._init_prompt()
        self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.config.device)

        self.info_nce_fc = nn.Linear(self.embedding_dim , self.embedding_dim).to(config.device)

        self.trigger_fc = nn.Linear(self.embedding_dim, 1).to(config.device)
        self.trigger_loss = nn.BCEWithLogitsLoss(reduction='none')

        # self.merge_fc = nn.Linear(2*self.embedding_dim, self.embedding_dim).to(config.device)


    def infoNCE_f(self, V, C):
        """
        V : B x vocab_size
        C : B x embedding_dim
        """
        try:
            out = self.info_nce_fc(V) # B x embedding_dim
            out = torch.matmul(out , C.t()) # B x B

        except:
            print("V.shape: ", V.shape)
            print("C.shape: ", C.shape)
            print("info_nce_fc: ", self.info_nce_fc)
        return out
    def _init_prompt(self):
        # is is is [e1] is is is [MASK] is is is [e2] is is is
        if self.config.prompt_init == 1:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            token_embedding = self.bert_word_embedding.weight[2003]
            prompt_embedding[list(range(self.prompt_lens)), :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data
       
        # ! @ # [e1] he is as [MASK] * & % [e2] just do it  
        elif self.config.prompt_init == 2:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            ids = [999, 1030, 1001, 2002, 2003, 2004, 1008, 1004, 1003, 2074, 2079,  2009]
            for i in range(self.prompt_lens):
                token_embedding = self.bert_word_embedding.weight[ids[i]]
                prompt_embedding[i, :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data


    def embedding_input(self, input_ids): # (b, max_len)
        input_embedding = self.bert_word_embedding(input_ids) # (b, max_len, h)
        prompt_embedding = self.softprompt_encoder(self.prompt_ids) # (prompt_len, h)

        for i in range(input_ids.size()[0]):
            p = 0
            for j in range(input_ids.size()[1]):
                if input_ids[i][j] == self.config.prompt_token_ids:
                    input_embedding[i][j] = prompt_embedding[p]
                    p += 1

        return input_embedding


    def forward(self, inputs, is_des=False, is_rd: bool=False): # (b, max_length)
        batch_size = inputs['ids'].size()[0]
        tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        
        trigger_mask: torch.Tensor = inputs.get('trigger_mask', None)
        # print(f"trigger_mask: {trigger_mask.sum()}")
        trigger_loss = None

        if not is_rd:
            if pattern == 'softprompt' or pattern == 'hybridprompt':
                input_embedding = self.embedding_input(inputs['ids'])
                outputs_words = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0] # (b, max_length, h)

                if not is_des:
                    trigger_logits = self.trigger_fc(outputs_words).squeeze(-1) # (b, max_length)

                    if trigger_mask is not None:
                        trigger_loss = self.trigger_loss(trigger_logits, trigger_mask.float())
                        trigger_loss = torch.masked_select(trigger_loss, inputs['mask'].bool()).mean()
                    # else:
                    #     trigger_mask = (trigger_logits > 0).to(inputs['mask'].dtype)
                    #     trigger_mask = trigger_mask * inputs['mask']

                    # outputs_words (b, max_length, h) and trigger_mask (b, max_length)
                    # trigger_embedding = (outputs_words*trigger_mask.unsqueeze(-1)).sum(dim=1) / trigger_mask.sum(-1, keepdim=True)
                    
            else:
                outputs_words = self.encoder(inputs['ids'], attention_mask=inputs['mask'])[0] # (b, max_length, h)
        else:
            if pattern == 'softprompt' or pattern == 'hybridprompt':
                input_embedding = self.embedding_input(inputs['rd_ids'])
                outputs_words = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['rd_mask'])[0]
            else:
                outputs_words = self.encoder(inputs['rd_ids'], attention_mask=inputs['rd_mask'])[0] # (b, max_length, h)

        # return [CLS] hidden
        if pattern == 'cls' or pattern == 'softprompt':
            clss = torch.zeros(batch_size, dtype=torch.long)
            return outputs_words[tensor_range ,clss] # (b, h)

        # return [MASK] hidden
        elif pattern == 'hardprompt' or pattern == 'hybridprompt':
            masks = []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                try:
                    mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
                except:
                    mask = 0
                
                masks.append(mask)
            if is_des or is_rd:
                average_outputs_words = torch.mean(outputs_words, dim=1)
                return average_outputs_words
            else:
                mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
                
                # concatenated_embedding = torch.cat((mask_hidden, trigger_embedding), dim=1) # (b, 2h)

                # mask_hidden = self.merge_fc(concatenated_embedding) # (b, h)

                if inputs.get('trigger_mask', None) is not None:
                    return mask_hidden, trigger_loss
                
                return mask_hidden
            # lm_head_output = self.lm_head(mask_hidden) # (b, max_length, vocab_size)
            # return mask_hidden , average_outputs_words

        # return e1:e2 hidden
        elif pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
                h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
                t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

            h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
            t_state = outputs_words[tensor_range, torch.tensor(t1)]

            concerate_h_t = (h_state + t_state) / 2 # (b, h)
            return concerate_h_t

