import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
import numpy as np
from tensorboardX import SummaryWriter
from modules.min_norm_solvers import MinNormSolver, gradient_normalizers
writer_tsne = SummaryWriter('runs/tsne')

def binary_cross_entropy_with_logits(input, target, mean=False):
    """
    Function that measures Binary Cross Entropy between target and output logits:
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = loss.sum(dim=1)
    return loss.mean() if mean else loss

def compute_supcon_loss(feats, gt, scale=None):
    if scale is not None:
        scale = torch.tensor([scale[i][gt[i].item()].item() for i in range(gt.shape[0])]).cuda()
        scale = scale.repeat(scale.shape[0], 1)
    tau = 1.0
    feats_filt = F.normalize(feats, dim=1)
    targets_r = gt.reshape(-1, 1)
    targets_c = gt.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    if scale is not None:
        positives *= scale
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim = 1)
    pred = pred.detach().cpu().numpy()
    score = (pred == np.array(labels))
    tot_correct = score.sum()
    return tot_correct


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores
    
def compute_loss(output, labels):

    #Function for calculating loss
    
    ce_loss = nn.CrossEntropyLoss(reduction='mean')(output, labels.squeeze(-1).long())
    
    return ce_loss


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results


def train(model, m_model, optim, train_loader, loss_fn, tracker, writer, tb_count, epoch, args):

    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))
    for v, q, a, mg, scale, q_id, f1, type, a_type in loader:
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        mg = mg.cuda()
        scale = scale.cuda()
        f1 = f1.cuda()
        dict_args = {'margin': mg, 'epoch': epoch, 'per': f1}
        gt = torch.argmax(a, 1)   

        hidden_, ce_logits, att = model(v, q)

        hidden, pred = m_model(hidden_, ce_logits, mg, epoch, a)


        ce_loss = - F.log_softmax(ce_logits, dim=-1) * a
        ce_loss = ce_loss * f1
        ce_loss = ce_loss.sum(dim=-1).mean()
        margin_loss = loss_fn(hidden, a, **dict_args)
        loss_t = margin_loss + ce_loss
            
        if args.DLR:
            supcon_loss = compute_supcon_loss(hidden_, gt, scale)
            loss_t = loss_t + supcon_loss
        else:
            supcon_loss = compute_supcon_loss(hidden_, gt)
            loss_t = loss_t + supcon_loss
        

        writer.add_scalars('data/losses', {
        }, tb_count)
        tb_count += 1
        
        if args.GMS:
            q_mask = torch.tensor([train_loader.dataset.dictionary.padding_idx] * (config.max_question_len)).cuda()
            q_mask = q_mask.unsqueeze(0).repeat(q.size(0), 1)
            _, logit_q, _ = model(None, q)
            _, logit_v, _ = model(v, q_mask)
            loss_q = 1 * F.cross_entropy(logit_q, gt)
            loss_v = 1 * F.cross_entropy(logit_v, gt)
            
            grads_t = {}
            grad_qscale = {}
            grad_vscale = {}
            grad_tscale = {}
            losses = {'q': loss_q, 'v':loss_v, 't':loss_t}
            all_lt = ['q', 'v', 't']
            grads_q = {}
            grads_v = {}
            for idx, lt in enumerate(all_lt):
                loss = losses[lt]
                loss.backward(retain_graph=True)
                grads_t[lt] = {}
                grads_q[lt] = {}
                grads_v[lt] = {}
                grad_qscale[lt] = 0.
                grad_vscale[lt] = 0.
                grad_tscale[lt] = 0.
                ls_q = []
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if (lt in ['q', 'v', 't']) and ('q_emb' in name or 'q_net' in name):
                        ls_q.append(param.grad.data.clone().flatten())
                        grads_q[lt][name] = param.grad.data.clone()
                
                if lt in ['q', 'v', 't']:
                    grads_q[lt]["concat"] = torch.cat(ls_q)
                optim.zero_grad()

            gn_q = gradient_normalizers(grads_q, losses, 'loss+')
            for lt in all_lt:
                for gr_i in grads_q[lt]:
                    grads_q[lt][gr_i] = grads_q[lt][gr_i] / gn_q[lt]
            
            # Frank-Wolfe iteration to compute scales.
            sol_q, min_norm = MinNormSolver.find_min_norm_element([list(grads_q[lt].values()) for lt in ['q', 'v', 't']])
            for i, lt in enumerate(['q', 'v', 't']):
                grad_qscale[lt] = float(sol_q[i])
            
            optim.pc_backward([grad_qscale['v'] * loss_v, grad_qscale['q'] * loss_q, grad_qscale['t'] * loss_t], model)
                
        else:
            loss = loss_t
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()
        
        #Ensemble the logit heads, as mentioned in Section 3 of the main paper, if bias-injection is enabled
        if config.bias_inject or config.learnable_margins:
            ce_logits = F.normalize(ce_logits)
            pred_l = F.normalize(pred)
            pred = (ce_logits + pred_l) / 2
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss_t.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                           acc=fmt(acc_trk.mean.value))

    return tb_count


#Evaluation code
def evaluate(model, m_model, dataloader, eval_dset, args, epoch=0, write=False):
    score = 0
    results = []  # saving for evaluation
    qat_score = {}
    qat_total = {}
    upper_bound = 0
    for v, q, a, mg, _, q_id, _, qtype, a_type in tqdm(dataloader, ncols=0, leave=True):
        v = v.cuda()
        q = q.cuda()
        mg = mg.cuda()
        a = a.cuda()
        hidden, ce_logits, att = model(v, q)
        hidden, pred = m_model(hidden, ce_logits, mg, epoch, a)
        
        #Ensemble the logit heads
        if config.learnable_margins or config.bias_inject:
            ce_logits = F.softmax(F.normalize(ce_logits) / config.temp, 1)
            pred_l = F.softmax(F.normalize(pred), 1)
            pred = config.alpha * pred_l + (1-config.alpha) * ce_logits


        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum(1)
        score += batch_score.sum()  
        upper_bound += a.max(1)[0].sum().item()
        for i in range(len(qtype)):
            if a[i].sum().item() == 0:
                continue
            qat_score[qtype[i]] = qat_score.get(qtype[i], 0) + batch_score[i]
            qat_total[qtype[i]] = qat_total.get(qtype[i], 0) + 1
            qat_score[a_type[i]] = qat_score.get(a_type[i], 0) + batch_score[i]
            qat_total[a_type[i]] = qat_total.get(a_type[i], 0) + 1
    
    score = score.item()
    print(score, len(dataloader.dataset))
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    
    if write:
        print("saving prediction results to disk...")
        result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
            config.task, config.test_split, config.version, epoch)
        with open(result_file, 'w') as fd:
            json.dump(results, fd)
    print('score:', score)
    print('upper_bound:', upper_bound)
    print('update_score:', score / upper_bound)
    for key in qat_score:
        print(key + ": " + str((qat_score[key]/qat_total[key]*100).item()))
    
    return score
