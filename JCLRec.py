import numpy as np
import pandas as pd
import math
import random
import argparse
import pickle
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import os
import time as Time
from utility import pad_history, calculate_hit, extract_axis_1
from Modules_ori import *
from logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='zhihu',
                        help='yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--w', type=float, default=2.0,
                        help='dropout ')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument('--CL_Loss', type=str, default='MSE_Loss',
                        help='')
    parser.add_argument('--lambda1', type=float, default='1',
                        help='')
    parser.add_argument('--lambda2', type=float, default='1',
                        help='')
    parser.add_argument('--threshold', type=int, default='20',
                        help='')
    parser.add_argument('--tsne', type=bool, default=False)
    parser.add_argument('--modelName', type=str, default='JCLRec',
                        help='')
    parser.add_argument('--item_crop', type=float, default='0.6',
                        help='')
    parser.add_argument('--item_mask', type=float, default='0.3',
                        help='')
    parser.add_argument('--item_reorder', type=float, default='0.6',
                        help='')

    return parser.parse_args()

topk = [5, 10, 20, 50]

args = parse_args()
logger = Logger(log_configs=True, args=args)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(args.random_seed)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(np.array(betas)).float()


def trunc_lin(num_diffusion_timesteps, beta_start, beta_end):
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * beta_start + 0.01
    beta_end = scale * beta_end + 0.01
    if beta_end > 1:
        beta_end = scale * 0.001 + 0.01
    return torch.linspace(beta_start, beta_end, num_diffusion_timesteps)


class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()
        elif args.beta_sche == 'trunc_lin':
            self.betas = trunc_lin(self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def D(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=1).mean()

    def negative_loss(self, predict_x_aug1, predict_x_aug2):
        nowPositiveTensors = predict_x_aug1
        random_indices = torch.randperm(predict_x_aug2.size()[0])
        nowNegtiveTensors = predict_x_aug2[random_indices]
        loss = F.mse_loss(nowPositiveTensors, nowNegtiveTensors)

        return loss

    def p_losses(self, denoise_model, x_start, target, h, t, seq_cup, seq, len_seq_numpy, device, noise=None, loss_type="l2"):
        #
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_x = denoise_model(x_noisy, h, t)

        aug_seq1, aug_len1, aug_seq2, aug_len2 = denoise_model.cl4srec_aug(seq, args)
        aug_seq3, aug_len3 = denoise_model.duorec_aug(seq_cup, len_seq_numpy, target, device)

        h_aug1 = denoise_model.cacu_h(aug_seq1, aug_len1, args.p)
        h_aug2 = denoise_model.cacu_h(aug_seq2, aug_len2, args.p)
        h_aug3 = denoise_model.cacu_h(aug_seq3, aug_len3, args.p)

        predicted_x_aug1 = denoise_model(x_noisy, h_aug1, t)
        predicted_x_aug2 = denoise_model(x_noisy, h_aug2, t)
        predicted_x_aug3 = denoise_model(x_noisy, h_aug3, t)

        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss_mse = F.mse_loss(x_start, predicted_x)

            if args.CL_Loss == 'MSE_Loss':
                loss_correct = args.lambda1 * F.mse_loss(x_start, predicted_x_aug3)
                loss_sparse = args.lambda2 * F.mse_loss(predicted_x_aug1, predicted_x_aug2)
                loss = loss_mse + loss_sparse + loss_correct
            else:
                loss_sparse = 0.1 * model.info_nce(
                    predicted_x_aug1, predicted_x_aug2, temp=1, batch_size=aug_seq1.shape[0])
                loss_correct = 0.1 * model.info_nce(x_start, predicted_x_aug3, temp=1, batch_size=aug_seq1.shape[0])
                loss = loss_mse + loss_sparse + loss_correct

        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, loss_mse, loss_correct, loss_sparse, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h, test_batch_size):
        totalRandom = []
        for i in range(h.shape[0] // test_batch_size):
            tempRandom = torch.randn_like(h[:test_batch_size, ])
            totalRandom.append(tempRandom)

        x = torch.cat(totalRandom, dim=0)

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h, torch.full((h.shape[0], ), n, device=device, dtype=torch.long), n)

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.train_seq_len_numpy = None
        self.train_seq_len = None
        self.train_seq = None
        self.same_target_index = None
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

        self.mask_default = self.mask_correlated_samples(
            batch_size=args.batch_size)
        self.cl_loss_func = nn.CrossEntropyLoss()

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
            )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size*2),
                nn.GELU(),
                nn.Linear(self.hidden_size*2, self.hidden_size),
            )

    # Contrastive Learning
    def cl4srec_aug(self, batch_seqs, args):
        def item_crop(seq, length, eta=args.item_crop):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.full(seq.shape, self.item_num)
            if crop_begin != 0:
                croped_item_seq[:num_left] = seq[crop_begin:(crop_begin + num_left)]
            else:
                croped_item_seq[:num_left] = seq[:(crop_begin + num_left)]
            return croped_item_seq.tolist(), num_left

        def item_mask(seq, length, gamma=args.item_mask):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = self.item_num
            return masked_item_seq.tolist(), length

        def item_reorder(seq, length, beta=args.item_reorder):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(
                range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            reordered_item_seq[reorder_begin:(reorder_begin + num_reorder)] = reordered_item_seq[shuffle_index]
            return reordered_item_seq.tolist(), length

        seqs = batch_seqs.tolist()
        mask = torch.ne(batch_seqs, self.item_num)
        lengths = mask.count_nonzero(dim=1).tolist()
        nonIndex = [index for index, val in enumerate(lengths) if val == 0]
        for index in nonIndex:
            lengths[index] = 1

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            seq = np.asarray(seq.copy(), dtype=np.int64)
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            if aug_len > 0:
                aug_seq1.append(aug_seq)
                aug_len1.append(aug_len)
            else:
                aug_seq1.append(seq.tolist())
                aug_len1.append(length)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            if aug_len > 0:
                aug_seq2.append(aug_seq)
                aug_len2.append(aug_len)
            else:
                aug_seq2.append(seq.tolist())
                aug_len2.append(length)

        aug_seq1 = torch.tensor(
            np.array(aug_seq1), dtype=torch.long, device=batch_seqs.device)
        aug_len1 = torch.tensor(
            np.array(aug_len1), dtype=torch.long, device=batch_seqs.device)
        aug_seq2 = torch.tensor(
            np.array(aug_seq2), dtype=torch.long, device=batch_seqs.device)
        aug_len2 = torch.tensor(
            np.array(aug_len2), dtype=torch.long, device=batch_seqs.device)
        return aug_seq1, aug_len1, aug_seq2, aug_len2

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != args.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)
        return info_nce_loss

    def semantic_augmentation(self, data_name, train_seq, train_seq_len, target,  train_seq_len_numpy, device):
        global args
        same_target_index_file = os.path.join('./data/', data_name, f'same_target_index_{args.threshold}.pkl')
        self.train_seq = train_seq
        self.train_seq_len = train_seq_len
        self.train_seq_len_numpy = train_seq_len_numpy
        if os.path.exists(same_target_index_file):
            with open(same_target_index_file, 'rb') as f:
                self.same_target_index = pickle.load(f)
            print("The candidate set has been loaded!!!!")
        else:
            same_target_index = {}
            train_last_items = np.asarray(target, dtype=np.int32)

            sorted_indices = np.argsort(train_last_items)
            train_last_items = train_last_items[sorted_indices]
            pre_item_id = train_last_items[0]
            pre_idx = 0
            for idx, item_id in enumerate(train_last_items):
                if item_id != pre_item_id:
                    last_group_indices = sorted_indices[pre_idx:idx]
                    same_target_seq_probility = {}
                    for seq_idx in last_group_indices:
                        source_seq = train_seq[seq_idx]
                        temp_probility = []
                        candidate_idx = []
                        for candidate_seq_idx in last_group_indices:
                            if seq_idx != candidate_seq_idx:
                                candidate_idx.append(candidate_seq_idx)
                                candidate_seq = train_seq[candidate_seq_idx]
                                intersection = set(source_seq) & set(candidate_seq)
                                union = set(source_seq) | set(candidate_seq)
                                probility = len(intersection) / len(union)
                                if probility == 0:
                                    probility = 1e-3
                                temp_probility.append(probility)
                        p = np.array(temp_probility) / sum(temp_probility)
                        if len(temp_probility) > args.threshold:
                            sampled_same_id = np.random.choice(candidate_idx, args.threshold, p=p,replace=False)
                        else:
                            sampled_same_id = [candidate_idx, p]
                        if len(candidate_idx) > 0:
                            seq_key = "-".join(map(str, source_seq))
                            same_target_seq_probility[seq_key] = sampled_same_id
                    if len(same_target_seq_probility) > 0:
                        same_target_index[pre_item_id.item()] = same_target_seq_probility
                    pre_item_id = item_id
                    pre_idx = idx
            with open(same_target_index_file, 'wb') as f:
                pickle.dump(same_target_index, f)
            self.same_target_index = same_target_index
            print("The candidate set has been constructed!!!!")

    def duorec_aug(self, batch_seqs, len_seq_numpy, batch_last_items, device):
        last_items = batch_last_items.tolist()
        train_seqs = self.train_seq
        train_seqs_len = self.train_seq_len
        sampled_pos_seqs = []
        sampled_pos_seqs_len = []
        for i, item in enumerate(last_items):
            if item in self.same_target_index:
                same_target_seq_probility = self.same_target_index[item]
                seq_key = "-".join(map(str, batch_seqs[i]))
                if seq_key in same_target_seq_probility:
                    if type(same_target_seq_probility[seq_key][0]) != type([]):
                        sampled_seq_idx = np.random.choice(same_target_seq_probility[seq_key])
                    else:
                        sampled_seq_idx = np.random.choice(a=same_target_seq_probility[seq_key][0], size=1, p=same_target_seq_probility[seq_key][1])[0]
                    sampled_seq_idx = sampled_seq_idx.astype(int)
                    sampled_pos_seqs.append(train_seqs[sampled_seq_idx])
                    sampled_pos_seqs_len.append(train_seqs_len[sampled_seq_idx])
            else:
                sampled_pos_seqs.append(batch_seqs[i])
                sampled_pos_seqs_len.append(len_seq_numpy[i].tolist())
        sampled_pos_seqs = torch.tensor(sampled_pos_seqs, dtype=torch.long, device=device)
        sampled_pos_seqs_len = torch.tensor(sampled_pos_seqs_len, device=device)
        return sampled_pos_seqs, sampled_pos_seqs_len

    def forward(self, x, h, step):
        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)
        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))

        return res

    def cacu_x(self, x):
        x = self.item_embeddings(x)
        return x

    def cacu_h(self, states, len_states, p):
        #hidden
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq, True)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)
        return h

    def predict(self, states, len_states, diff, test_batch_size):
        #hidden
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq, True)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        x = diff.sample(self.forward, self.forward_uncon, h, test_batch_size)

        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))

        return scores, x


def evaluate(model, test_data, diff, device, phaseText, epoch_index):
    eval_data=pd.read_pickle(os.path.join(data_directory, test_data))

    total_purchase = 0.0
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)
    batch_size = 100

    num_total = len(seq)

    bb= []
    cc = []
    dd = []
    ff = []
    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]

        bb.extend(seq_b)
        cc.extend(len_seq_b)
        dd.extend(target_b)

        total_purchase += batch_size

    states = np.array(bb)
    states = torch.LongTensor(states)
    states = states.to(device)
    temp = torch.LongTensor(np.array(cc)).to(device)

    prediction, testEmbeddings = model.predict(states, temp, diff, batch_size)
    ff.append(prediction)
    _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
    topK = topK.cpu().detach().numpy()
    sorted_list2=np.flip(topK,axis=1)
    sorted_list2 = sorted_list2
    calculate_hit(sorted_list2, topk, dd, hit_purchase, ndcg_purchase)

    hr_list = []
    ndcg_list = []

    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])

        if i == 1:
            hr_20 = hr_purchase

    eval_result = {"HR":[hr_list[0], hr_list[1], hr_list[2], hr_list[3]],
                       "NDCG": [ndcg_list[0], ndcg_list[1], ndcg_list[2], ndcg_list[3]]}

    logger.log_eval(eval_result, topk, data_type=phaseText, epoch_idx=epoch_index)

    return hr_20, testEmbeddings

def emTSNE(model, epoch):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    item_embeddings = model.item_embeddings.weight.data.cpu().numpy()
    item_embeddings = item_embeddings[:]
    item_embeddings = tsne.fit_transform(item_embeddings)
    plt.figure(figsize=(8, 6))
    # , cmap='viridis'
    plt.scatter(item_embeddings[:, 0], item_embeddings[:, 1], c=(213/255,38/255,39/255), s=3, alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig("./pic/{}_epoch_{}.pdf".format(args.data, epoch))


def save_model(model, model_name, testEmbeddings, epoch):
    model_state_dict = model.state_dict()
    save_dir_path = "./checkpoint/{}".format(args.data)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    save_path = "{}/{}_{}_epoch_{}.pth".format(save_dir_path, model_name, args.data, epoch)
    torch.save(model_state_dict, save_path)
    save_path_em = "{}/{}_{}_{}_epoch_{}.pth".format(save_dir_path, model_name, args.data, "Embedding", epoch)
    torch.save(testEmbeddings.data, save_path_em)
    logger.log("Save model parameters to {}".format(save_path), save_to_file=False)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5, 10, 20, 50]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timesteps = args.timesteps


    model = Tenc(args.hidden_factor,item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

    train_seq, train_seq_len, last_item = list(train_data['seq'].values), list(train_data['len_seq'].values), list(train_data['next'].values)

    train_seq_len_numpy = np.array(train_seq_len)

    model.semantic_augmentation(args.data, train_seq, train_seq_len, last_item, train_seq_len_numpy, device)


    total_step=0
    hr_max = 0
    best_epoch = 0

    All_Loss = {}
    Loss = []
    Loss_MSE = []
    Loss_Sparse = []
    Loss_Correct = []

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for i in range(args.epoch): # args.epoch
        start_time = Time.time()
        for j in range(num_batches):
            model.train()
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq_cpu = list(batch['seq'].values())
            len_seq_list = list(batch['len_seq'].values())
            target=list(batch['next'].values())

            optimizer.zero_grad()
            seq_cuda = torch.LongTensor(seq_cpu)
            len_seq = torch.LongTensor(len_seq_list)
            len_seq_numpy = np.array(len_seq_list)
            target = torch.LongTensor(target)

            seq_cuda = seq_cuda.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)

            x_start = model.cacu_x(target)

            h = model.cacu_h(seq_cuda, len_seq, args.p)

            n = torch.randint(0, args.timesteps, (args.batch_size, ), device=device).long()
            loss, loss_mse, loss_correct, loss_sparse, predicted_x = diff.p_losses(model, x_start, target, h, n, seq_cpu, seq_cuda, len_seq_numpy, device, loss_type='l2')

            loss.backward()
            optimizer.step()

        if args.report_epoch:
            if i % 1 == 0:
                Loss.append(round(loss.item(), 4))
                Loss_MSE.append(round(loss_mse.item(), 4))
                Loss_Sparse.append(round(loss_sparse.item(), 4))
                Loss_Correct.append(round(loss_correct.item(), 4))
                messages = "Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(loss) + 'MSE loss: {:.4f}; '.format(loss_mse) + 'Sparse loss: {:.4f}; '.format(loss_sparse) + 'Correct loss: {:.4f}; '.format(loss_correct) + "Time cost: " + Time.strftime(
                    "%H: %M: %S", Time.gmtime(Time.time()-start_time))
                logger.log_train(messages)

            if (i + 1) % 10 == 0:
                eval_start = Time.time()
                model.eval()
                # print('------------------------------------ VAL PHRASE ------------------------------------')
                _, _1 = evaluate(model, 'val_data.df', diff, device, "VAL PHRASE", i)
                # print('------------------------------------ TEST PHRASE -----------------------------------')
                _, testEmbeddings = evaluate(model, 'test_data.df', diff, device, "TEST PHRASE", i)

                messages = "Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)) + "\n" + "-" * 81
                logger.log_train(messages)

            if i == 999:
                All_Loss['Loss'] = Loss
                All_Loss['Loss_MSE'] = Loss_MSE
                All_Loss['Loss_Sparse'] = Loss_Sparse
                All_Loss['Loss_Correct'] = Loss_Correct
                loss_outputfile_path = os.path.join(f'./analyze/loss/{args.data}')
                loss_outputfile = os.path.join(loss_outputfile_path, f'{args.CL_Loss}_epoch_{i}.pkl')
                if not os.path.exists(loss_outputfile_path):
                    os.makedirs(loss_outputfile_path)
                with open(loss_outputfile, 'wb') as f:
                    pickle.dump(All_Loss, f)
                break

            # if i == 1:
            #     loss_outputfile_path = os.path.join(f'./analyze/loss/{args.data}')
            #     loss_outputfile = os.path.join(loss_outputfile_path, f'epoch_{10}.pkl')
            #     if os.path.exists(loss_outputfile):
            #         with open(loss_outputfile, 'rb') as f:
            #             All_Loss = pickle.load(f)
            #         print("The candidate set has been loaded!!!!")
            #         print(All_Loss)
            #         break




