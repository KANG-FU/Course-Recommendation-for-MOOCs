import os
import time
import torch

from model import Recommendation
from utils import *

# hyper parameters
dataset = 'courses'
train_dir = 'default'
batch_size = 256  # 128
maxlen = 10  # 50 too long
device = 'cuda' # cpu or cuda
hidden_units = 50
num_blocks = 1
num_epochs = 150  
num_heads = 1
dropout_rate = 0.1
l2_emb = 0.0
lr = 0.0001
cutoff = 10



dataset = data_partition(dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) // batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
# compute the average length of sequences：
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join('log.txt'), 'w')  # 'w' open for writing

# Preapare for output
columns_output = ['cutoff','epoch','validation_NDCG', 'validation_hit', "test_NDCG", "test_hit"]
res = pd.DataFrame(columns=columns_output)
# num_epochs_group = [10, 50,100, 200]
cutoffs = [5,6,7,8,9,10,11,12,13,14,15]
training_curve = pd.DataFrame(columns=['epoch', 'step', 'loss'])
for cutoff in cutoffs:
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)
    model = Recommendation(usernum, itemnum, hidden_units, num_heads, num_blocks, maxlen, dropout_rate, device).to(device) 
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
            print('success', name)
        except:
            print('failed init layer', name)
            pass # just ignore those failed init layers
    
    
    model.train()  # enable model training
    
    epoch_start_idx = 1
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    
    
    print('Working on the parameter num_heads: ', num_heads)
    for epoch in range(epoch_start_idx, num_epochs + 1):
    
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_embedding.parameters(): loss += l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 10 ==0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
            if step == num_batch - 1:
                To_output = [[epoch, step, loss.item()]]
                training_curve = training_curve.append(pd.DataFrame(To_output, columns=['epoch', 'step', 'loss']))
        
        if epoch % 5 == 0 or epoch == num_epochs:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, maxlen, cutoff)
            t_valid = evaluate_valid(model, dataset, maxlen, cutoff)
            print('\n epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
            if epoch == num_epochs:
                To_output = [[cutoff, epoch, t_valid[0], t_valid[1], t_test[0], t_test[1]]]
                res = res.append(pd.DataFrame(To_output, columns=columns_output))
    
        if epoch == num_epochs:
            fname = 'Recommendation.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(num_epochs, lr, num_blocks, num_heads, hidden_units, maxlen)
            torch.save(model.state_dict(), os.path.join(fname))
            
        
f.close()
sampler.close()
res.to_csv("Experiment with cutoff.csv")
#training_curve.to_csv("training_curve(200epochs).csv")
print("Done")
