import torch

# 计算输入数据 X 和聚类中心 centroids 之间的 欧氏距离平方矩阵
# 欧氏距离的平方是 K-Means 损失函数中常用的度量
def default_target_func(X, centroids):
    return torch.cdist(X, centroids, p=2)**2

class KMeansPlusPlus:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, device='cuda', logging=False):
        """
        Initialize the KMeans++ class.

        Parameters:
            n_clusters (int): Number of clusters. 聚类簇数
            max_iter (int): Maximum number of iterations. 最大迭代次数
            tol (float): Convergence tolerance. 收敛阈值，若迭代中质心移动量小于此值则停止
            device (str): Device type, 'cuda' or 'cpu'. 运行设备，可以是 'cuda' 或 'cpu'
            logging (bool): Whether to log convergence information. 是否打印调试信息
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        # 聚类中心初始化为空
        self.centroids = None
        self.logging = logging
        # 当前损失初始化为空
        self.loss = None
        self.empty_cluster_warning = 0
        self.vq_ids = None   # 新增：保存每个样本的最终cluster编号

    # 用 k-means++ 方法初始化聚类中心，比随机初始化更稳定。
    def initialize_centroids(self, X, target_func, sample_weight=None):
        """
        Initialize centroids using the k-means++ method with sample weights.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).
            sample_weight (Tensor, optional): Sample weights, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        # 初始化质心
        centroids = torch.empty((self.n_clusters, n_features), device=self.device)
        
        # Randomly select the first centroid 随机选取第一个中心点。
        indices = torch.randint(0, n_samples, (1,), device=self.device)
        centroids[0] = X[indices]
        
        # Initialize distances 初始化距离
        distances = torch.full((n_samples,), float('inf'), device=self.device)
        
        # 重复直到选满 n_clusters 个中心
        for i in range(1, self.n_clusters):
            # Compute distances from each point to each centroid 计算每个样本到已选中心的最小距离。
            dist = target_func(X, centroids[None, i - 1]).squeeze(-1)
            
            distances = torch.min(distances, dist)
            
            # 如果 sample_weight 不为空，会将样本权重纳入距离计算
            if sample_weight is not None:
                # Incorporate sample weights into probabilities
                weighted_distances = distances * sample_weight
            else:
                weighted_distances = distances
            
            # Select the next centroid with probability proportional to the weighted distance
            # 以“距离越大，被选中概率越高”的策略，按概率抽样选下一个中心点
            probabilities = weighted_distances / torch.sum(weighted_distances)
            categorical = torch.distributions.Categorical(probs=probabilities)
            index = categorical.sample().item()
            centroids[i] = X[index]
        
        self.centroids = centroids

    # 核心的 K-Means 训练过程
    def fit(self, X, target_func=default_target_func, sample_weight=None):
        """
        Train the KMeans model with optional sample weights.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).
            sample_weight (Tensor, optional): Sample weights, shape (n_samples,).
        """

        assert len(X.shape) == 2

        input_device = X.device

        # 把输入数据 X 和可选的 sample_weight 转到指定 device，并转换成 float32
        X = X.to(self.device).to(torch.float32)
        
        if sample_weight is not None:
            sample_weight = sample_weight.to(self.device).to(torch.float32)
            # Normalize sample weights to sum to 1 如果有样本权重，先归一化（总和为1）
            sample_weight = sample_weight / torch.sum(sample_weight)
        
        # 增加：ensure vq_ids reset at start
        self.vq_ids = None

        # 调用 initialize_centroids 用 k-means++ 选择初始中心
        self.initialize_centroids(X, target_func, sample_weight)
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # Compute distances from each point to each centroid
            # 分配标签：计算所有样本到每个中心的距离平方矩阵，取最小值的索引作为类别 labels
            distances = target_func(X, self.centroids)
            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)
            # 计算损失：记录每个样本的最小距离平方和
            self.loss = distances.min(dim=-1).values.sum()
            if self.logging:
                print(f"[INFO] clustering loss {self.loss}")
            
            # Compute new centroids with sample weights
            # 更新中心点
            new_centroids = torch.zeros_like(self.centroids)
            # 按簇对样本求均值，如果有 sample_weight 就是加权均值
            if sample_weight is not None:
                # Multiply X by sample weights
                weighted_X = X * sample_weight.unsqueeze(1)
                new_centroids.index_add_(0, labels, weighted_X)
                counts = torch.zeros(self.n_clusters, device=self.device)
                counts.scatter_add_(0, labels, sample_weight)
            else:
                new_centroids.index_add_(0, labels, X)
                counts = torch.zeros(self.n_clusters, device=self.device)
                counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float, device=self.device))
            # 处理空簇：如果某个簇没有样本，避免除0，将该簇计数设为1，并给出 warning
            if not torch.all(counts > 0):
                self.empty_cluster_warning = (counts == 0).sum().item()
                counts = torch.where(counts == 0, 1, counts)
            
            new_centroids /= counts.unsqueeze(1)
            
            # Compute the shift in centroids
            # 检查收敛：计算新旧质心的 欧氏距离总变化量
            centroid_shift = torch.sum((self.centroids - new_centroids) ** 2).sqrt().item()
            self.centroids = new_centroids
            
            # Check for convergence 如果变化量小于 tol，认为已收敛并提前结束
            if centroid_shift < self.tol:
                if self.logging:
                    print(f"[INFO] converged at iteration {iteration}")
                break
        else: # 如果循环结束还没收敛，打印 warning
            print("[WARNING] kmeans clustering did not converge")

        # 将 centroids 转回原始输入设备
        self.centroids = self.centroids.to(input_device)
        # 如果存在空簇，打印警告
        if self.empty_cluster_warning:
            print(f"[WARNING] {self.empty_cluster_warning} empty clusters")
            self.empty_cluster_warning = 0
        
        # 新增：在收敛后直接计算最终的样本->cluster 标签
        # vq_ids 即为每个输入样本对应的簇编号
        # 最终计算并保存 vq_ids（返回到原始输入设备）
        try:
            self.vq_ids = self.predict(X.to(input_device), target_func).detach().clone()
        except Exception as e:
            # 保底：如果 predict 失败，仍保持 vq_ids 为 None，并打印警告
            print(f"[ERROR] failed to compute vq_ids: {e}", flush=True)
            self.vq_ids = None
        

    # 输入数据 X，计算其到训练好 centroids 的距离平方，并返回最近的质心索引作为预测标签
    def predict(self, X, target_func=default_target_func):
        """
        Predict cluster labels for new data points.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).

        Returns:
            labels (Tensor): Cluster labels for each point.
        """
        input_device = X.device
        X = X.to(self.device).to(torch.float32)
        distances = target_func(X, self.centroids.to(self.device))
        labels = torch.argmin(distances, dim=1)
        return labels.to(input_device)

    # 一站式接口：先 fit 训练，再 predict 返回标签
    def fit_predict(self, X, target_func=default_target_func, sample_weight=None):
        """
        Train the model and predict cluster labels with optional sample weights.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).
            sample_weight (Tensor, optional): Sample weights, shape (n_samples,).

        Returns:
            labels (Tensor): Cluster labels for each point.
        """
        self.fit(X, target_func, sample_weight)
        return self.predict(X, target_func)
    