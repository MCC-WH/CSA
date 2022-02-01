import os
import time
from utils import get_data_root, load_pickle, save_pickle
import torch
import torch.utils.data as data


def collate_tuples(batch):
    batch = list(filter(lambda x: x is not None, batch))
    sim_list, relative_pre = zip(*batch)
    return torch.stack(sim_list, dim=0), torch.stack(relative_pre, dim=0)


class FeatureFromList(data.Dataset):
    def __init__(self, features=None, topk_indices=None):
        super(FeatureFromList, self).__init__()
        self.features = features
        self.topk_indices = topk_indices
        self.len = topk_indices.size(0)

    def __getitem__(self, index):
        topk_indices = self.topk_indices[index]
        feature = self.features[topk_indices]
        return feature

    def __len__(self):
        return self.len


class RerankDataset_TopKSIM(data.Dataset):
    def __init__(self, names, mode, topk=512, sim_len=512):

        if not (mode == 'train' or mode == 'val'):
            raise (RuntimeError("MODE should be either train or val, passed as string"))

        db_fn = os.path.join(get_data_root(), "annotations", "retrieval-SfM-120k.pkl")
        db = load_pickle(db_fn)[mode]
        clusters = torch.Tensor(db['cluster'])
        print('>> dn info read done')

        # load for image features
        self.topk = topk
        self.sim_len = sim_len
        
        topk = max(topk, sim_len)

        if len(names) > 1:
            topk_prefix = os.path.join(get_data_root(), 'train', names + '_{}sim{}topk_AUG.pkl'.format(self.sim_len, self.topk))
        else:
            topk_prefix = os.path.join(get_data_root(), 'train', names + '_{}sim{}topk.pkl'.format(self.sim_len, self.topk))
        db_features = [os.path.join(get_data_root(), 'train', name + '.pkl') for name in names]

        if os.path.exists(topk_prefix):
            print('>> read topk indices from:{}'.format(topk_prefix))
            self.topk_indices = torch.Tensor(load_pickle(topk_prefix)['topk']).long()
            self.clusters = []
            self.features = []
            for path in db_features:
                print('>> read db feature from:{}'.format(path))
                features = torch.tensor(load_pickle(path)[mode]).float()
                self.features.append(features)
                self.clusters.append(clusters)
            self.clusters = torch.cat(self.clusters, dim=-1)
            self.features = torch.cat(self.features, dim=0)
        else:
            print('>> Generate topk indices and save to:{}'.format(topk_prefix))
            self.clusters = []
            self.features = []
            self.topk_indices = []
            total = 0
            for path in db_features:
                print('>> read db feature from:{}'.format(path))
                features = torch.tensor(load_pickle(path)[mode]).float()
                self.features.append(features)
                self.clusters.append(clusters)
                sim = torch.mm(features, features.t())
                topk_indices = torch.topk(sim, k=topk, dim=-1)[1]
                relative = []
                for k in topk_indices:
                    cluster_id = clusters[k]
                    relative.append(cluster_id == cluster_id[0])
                relative = torch.stack(relative, dim=0).float()
                print('>> average relative percentage: {:.2f}'.format(relative.mean(dim=-1).mean() * 100))
                self.topk_indices.append(topk_indices + total)
                total = total + features.size(0)
            self.clusters = torch.cat(self.clusters, dim=-1)
            self.features = torch.cat(self.features, dim=0)
            self.topk_indices = torch.cat(self.topk_indices, dim=0).long()
            save_pickle(topk_prefix, {'topk': self.topk_indices.numpy()})

        self.relative = None
        self.generate_relate()

    @torch.no_grad()
    def generate_relate(self):
        start_time = time.time()
        self.relative = []
        for topk in self.topk_indices:
            cluster_id = self.clusters[topk]
            self.relative.append(cluster_id == cluster_id[0])
        print('>> Generate top {} indices relate label for training in {:.2f} s'.format(self.topk, time.time() - start_time))
        self.relative = torch.stack(self.relative, dim=0).float()
        print('total training samples:{}'.format(self.topk_indices.size(0)))
        print('total training features:{}'.format(self.features.size(0)))
        print('>> average relative percentage: {:.2f}'.format(self.relative.float().mean(dim=-1).mean() * 100))

    def __getitem__(self, index):
        try:
            topk_indices = self.topk_indices[index][:self.topk]
            sim_indices = self.topk_indices[index][:self.sim_lem]
            feature = self.features[topk_indices]
            sim_feature = self.features[sim_indices]
            sim_list = torch.mm(feature, sim_feature.t())
            # random permute
            permute_index = torch.cat((torch.zeros(1).long(), torch.randperm(sim_list.size(0) - 1) + 1))
            relative_pre = self.relative[index]
            return sim_list[permute_index], relative_pre[permute_index]
        except:
            return None

    def __len__(self):
        return self.topk_indices.size(0)
