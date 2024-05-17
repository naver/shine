import torch


class Themer:
    def __init__(self, method, thresh=1, alpha=0.5):
        if method not in ['mean', 'peigen', 'mixed', 'all_eigens']:
            raise NameError(f"{method} is not supported")

        self.method = method
        self.T = thresh
        self.alpha = alpha

    def _get_principal_eigenvector(self, stacked_feats):
        # Single child case
        if stacked_feats.shape[0] == 1:
            return stacked_feats[0]
        # print("========== used Peigen when shape={}".format(stacked_feats.shape))

        # Convert to float32 (clip feat is in float16)
        stacked_feats32 = stacked_feats.to(torch.float32)

        # Compute principal eigenvector
        U, S, V = torch.svd(stacked_feats32)            # SVD decomposition
        peigen_v = V[:, 0]                              # principal eigenvector
        peigen_v = peigen_v
        peigen_v = peigen_v.to(torch.float16)           # convert it back to float16 for CLIP
        return peigen_v

    def _get_all_eigenvector(self, stacked_feats):
        # Single child case
        if stacked_feats.shape[0] == 1:
            return stacked_feats[0]
        # print("========== used ALL eigens when shape={}".format(stacked_feats.shape))

        # Convert to float32 (clip feat is in float16)
        stacked_feats32 = stacked_feats.to(torch.float32)

        # Compute principal eigenvector
        U, S, V = torch.svd(stacked_feats32)            # SVD decomposition
        normalized_weights_s = S / torch.sum(S)
        weighted_avg_v = torch.zeros_like(V[:, 0])

        for i in range(V.size(1)):
            weighted_avg_v += normalized_weights_s[i] * V[:, i]

        weighted_avg_v = weighted_avg_v
        weighted_avg_v = weighted_avg_v.to(torch.float16)
        return weighted_avg_v                       # weighted average
        # avg_v = V.mean(dim=1)
        # avg_v = avg_v / torch.norm(avg_v)
        # return avg_v.to(torch.float16)    # normal average

    def _get_mean_vector(self, stacked_feats):
        # print("========== used Mean when shape={}".format(stacked_feats.shape))

        mean_theme = torch.mean(stacked_feats, dim=0)   # mean vector
        return mean_theme

    def _get_mixed_vector(self, stacked_feats):
        # print("========== used MIXED when shape={}".format(stacked_feats.shape))
        mean_v = self._get_mean_vector(stacked_feats)
        peigen_v = self._get_principal_eigenvector(stacked_feats)
        mixed_v = self.alpha * mean_v + (1 - self.alpha) * peigen_v
        return mixed_v

    def get_theme(self, stacked_feats):
        if self.method == 'peigen' and stacked_feats.shape[0] > self.T:
            return self._get_principal_eigenvector(stacked_feats)
        elif self.method == 'all_eigens' and stacked_feats.shape[0] > self.T:
            return self._get_all_eigenvector(stacked_feats)
        elif self.method == 'mixed' and stacked_feats.shape[0] > self.T:
            return self._get_mixed_vector(stacked_feats)
        else:
            return self._get_mean_vector(stacked_feats)


if __name__ == '__main__':
    analyzer = Themer(method='mean', thresh=1)
    stacked_feats = torch.rand((5, 10))  # Replace this with your actual data

    result = analyzer.get_theme(stacked_feats)
    print(result)

