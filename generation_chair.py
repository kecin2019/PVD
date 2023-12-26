import torch
from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base

from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds

"""
models
"""


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个正态分布之间的KL散度。
    正态分布的参数为均值和对数方差。

    参数:
    mean1, logvar1: 第一个正态分布的均值和对数方差。
    mean2, logvar2: 第二个正态分布的均值和对数方差。

    返回:
    KL散度的值。
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # 假设数据是整数，范围在[0, 1]之间
    assert x.shape == means.shape == log_scales.shape

    # 定义一个标准正态分布作为基准分布
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    # 中心化数据
    centered_x = x - means
    # 计算每个正态分布的逆标准差
    inv_stdv = torch.exp(-log_scales)

    # 计算两个相邻离散点处的逆标准差
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)  # 计算这些点处的累积分布函数

    min_in = inv_stdv * (centered_x - 0.5)
    cdf_min = px0.cdf(min_in)  # 计算这些点处的累积分布函数

    # 计算log cdf和log(1-cdf)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(
        torch.max(1.0 - cdf_min, torch.ones_like(cdf_min) * 1e-12)
    )

    cdf_delta = cdf_plus - cdf_min

    # 根据x的值，选择适当的log概率
    log_probs = torch.where(
        x < 0.001,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12)),
        ),
    )

    # 确保log_probs的形状与x的形状相同
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        # 设置损失类型、模型均值类型和模型方差类型
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        # 转换betas为numpy数组，并确保所有值都在0到1之间
        betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()

        # 获取时间步数
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # 计算alphas并转换为torch tensor
        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(
            np.append(1.0, alphas_cumprod[:-1])
        ).float()

        # 存储betas, alphas_cumprod, alphas_cumprod_prev为torch tensor
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # 计算用于扩散q(x_t | x_{t-1})和其他的数学量
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float()

        # 计算用于后验q(x_{t-1} | x_t, x_0)的数学量
        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()

        # 计算后验的方差
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        # 对后验的方差进行裁剪，避免过小的值
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
        )
        # 计算后验均值的系数1
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # 计算后验均值的系数2
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        从指定的时间步中提取一些系数，然后重塑为 [batch_size, 1, 1, 1, 1, ...] 以便于广播。
        """
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        # 将输出重塑为与x_shape相同的batch_size，其余维度均为1
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        """
        计算指定时间步的均值和方差。
        """
        # 使用_extract方法从sqrt_alphas_cumprod中提取系数，并乘以x_start得到均值
        mean = (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
        )
        # 使用_extract方法从1.0 - alphas_cumprod中提取系数得到方差
        variance = self._extract(
            1.0 - self.alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        # 使用_extract方法从log_one_minus_alphas_cumprod中提取系数得到对数方差
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        对数据进行扩散（t == 0表示扩散了1步）。
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        # 使用_extract方法从sqrt_alphas_cumprod和sqrt_one_minus_alphas_cumprod中提取系数
        # 然后与x_start和noise相乘，最后相加得到扩散后的样本
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算扩散后某一时间步的后验均值和方差。
        """
        # 确保x_start和x_t的形状相同
        assert x_start.shape == x_t.shape
        # 使用_extract方法从posterior_mean_coef1和posterior_mean_coef2中提取系数，
        # 然后与x_start和x_t相乘并相加，得到后验均值
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape)
            * x_start
            + self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape)
            * x_t
        )
        # 使用_extract方法从posterior_variance中提取系数，得到后验方差
        posterior_variance = self._extract(
            self.posterior_variance.to(x_start.device), t, x_t.shape
        )
        # 使用_extract方法从posterior_log_variance_clipped中提取系数，得到对数后验方差
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape
        )
        # 确保后验均值、方差和对数方差的形状与x_start相同
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool
    ):
        """
        通过解噪声函数计算数据在特定时间步的预测均值、方差和对数方差。
        """
        # 使用解噪声函数处理数据，得到模型输出
        model_output = denoise_fn(data, t)

        # 根据模型方差类型选择对应的方差和对数方差
        if self.model_var_type in ["fixedsmall", "fixedlarge"]:
            # 对于fixedlarge类型，我们这样设置初始（对数）方差以获得更好的解码对数似然
            model_variance, model_log_variance = {
                "fixedlarge": (
                    self.betas.to(data.device),
                    torch.log(
                        torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                    ).to(data.device),
                ),
                "fixedsmall": (
                    self.posterior_variance.to(data.device),
                    self.posterior_log_variance_clipped.to(data.device),
                ),
            }[self.model_var_type]
            # 使用_extract方法从model_variance中提取系数，得到模型方差
            model_variance = self._extract(
                model_variance, t, data.shape
            ) * torch.ones_like(data)
            # 使用_extract方法从model_log_variance中提取系数，得到模型对数方差
            model_log_variance = self._extract(
                model_log_variance, t, data.shape
            ) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        # 根据模型均值类型选择相应的计算方法
        if self.model_mean_type == "eps":
            # 使用解噪声函数处理数据，得到模型输出
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                # 如果需要裁剪，则对重构值进行裁剪
                x_recon = torch.clamp(x_recon, -0.5, 0.5)

            # 使用q_posterior_mean_variance方法计算模型的均值、方差和对数方差
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x_recon, x_t=data, t=t
            )
        else:
            raise NotImplementedError(self.loss_type)

        # 确保所有输出的形状与输入数据相同
        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        # 根据return_pred_xstart参数决定是否返回重构值x_recon
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        # 确保输入数据的形状与噪声形状相同
        assert x_t.shape == eps.shape
        # 使用_extract方法从sqrt_recip_alphas_cumprod中提取系数，得到sqrt_recip_alphas_cumprod_t
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape
        )
        # 使用_extract方法从sqrt_recipm1_alphas_cumprod中提取系数，得到sqrt_recipm1_alphas_cumprod_t
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape
        )
        # 使用提取的系数计算重构值x_start
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps

    """ samples """

    def p_sample(
        self,
        denoise_fn,
        data,
        t,
        noise_fn,
        clip_denoised=False,
        return_pred_xstart=False,
        use_var=True,
    ):
        """
        从模型中采样
        """
        # 计算模型的均值、方差和重构值x_start
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            data=data,
            t=t,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        # 根据给定的噪声函数生成噪声数据
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        # 确保噪声数据与输入数据的形状相同
        assert noise.shape == data.shape
        # 当t为0时，不进行噪声添加
        nonzero_mask = torch.reshape(
            1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1)
        )

        # 初始化采样为模型均值
        sample = model_mean
        # 如果使用方差，将噪声添加到采样中
        if use_var:
            sample = sample + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        # 确保采样形状与重构值x_start的形状相同
        assert sample.shape == pred_xstart.shape
        # 如果返回pred_xstart，则返回采样和x_start，否则只返回采样
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(
        self,
        denoise_fn,
        shape,
        device,
        noise_fn=torch.randn,
        constrain_fn=lambda x, t: x,
        clip_denoised=True,
        max_timestep=None,
        keep_running=False,
    ):
        """
        生成样本
        keep_running: 如果我们运行2 x num_timesteps，则为True，如果只运行num_timesteps，则为False
        """
        # 如果没有指定最大时间步，则将最终时间设置为模型的时间步数
        if max_timestep is None:
            final_time = self.num_timesteps
        else:
            final_time = max_timestep

        # 检查shape是否为元组或列表
        assert isinstance(shape, (tuple, list))
        # 使用给定的噪声函数生成初始噪声图像
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)

        # 对于每一个时间步，从最终时间到0（不包括0）
        for t in reversed(
            range(0, final_time if not keep_running else len(self.betas))
        ):
            # 对图像应用约束函数
            img_t = constrain_fn(img_t, t)
            # 创建一个张量来存储当前的时间步
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            # 使用p_sample方法进行采样，但不返回x_start
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            ).detach()

        # 确保生成的图像与给定的形状相同
        assert img_t.shape == shape
        # 返回生成的图像
        return img_t

    def reconstruct(
        self, x0, t, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t: x
    ):
        # 断言t大于等于1，以确保有足够的时间步来执行重构
        assert t >= 1

        # 创建一个张量来表示t-1的时间步
        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(
            t - 1
        )
        # 使用q_sample方法对输入x0进行编码
        encoding = self.q_sample(x0, t_vec)

        # 初始化img_t为编码后的图像
        img_t = encoding

        # 对于每一个时间步，从t-1到0（不包括0）
        for k in reversed(range(0, t)):
            # 对图像应用约束函数
            img_t = constrain_fn(img_t, k)
            # 创建一个张量来表示当前的时间步k
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            # 使用p_sample方法进行采样，但不返回x_start
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                clip_denoised=False,
                return_pred_xstart=False,
                use_var=True,
            ).detach()

        # 返回重构后的图像
        return img_t


class PVCNN2(PVCNN2Base):
    # 定义SA块的参数，包括各个阶段的空间分解卷积块和空间注意力卷积块的参数
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    # 定义FP块的参数，包括各个阶段的空间分解卷积块和空间注意力卷积块的参数
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
        self,
        num_classes,
        embed_dim,
        use_att,
        dropout,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        # 初始化父类的构造函数，并传入相应的参数
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )


class Model(nn.Module):
    def __init__(
        self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str
    ):
        # 初始化模型类
        super(Model, self).__init__()

        # 创建GaussianDiffusion实例，用于处理噪声扩散
        self.diffusion = GaussianDiffusion(
            betas, loss_type, model_mean_type, model_var_type
        )

        # 创建PVCNN2实例，用于空间分类
        self.model = PVCNN2(
            num_classes=args.nc,  # 类别数量
            embed_dim=args.embed_dim,  # 嵌入维度
            use_att=args.attention,  # 是否使用注意力机制
            dropout=args.dropout,  # 丢弃率
            extra_feature_channels=0,  # 额外特征通道数
        )

    def prior_kl(self, x0):
        # 计算先验分布与给定数据的KL散度
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        # 计算所有与KL散度相关的值，包括总KL散度、各个时间步的KL散度、先验KL散度和MSE
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
            self._denoise, x0, clip_denoised
        )

        return {
            "total_bpd_b": total_bpd_b,
            "terms_bpd": vals_bt,
            "prior_bpd_b": prior_bpd_b,
            "mse_bt": mse_bt,
        }

    def _denoise(self, data, t):
        # 对给定的数据进行去噪
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        # 使用PVCNN2模型进行去噪
        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        # 计算损失迭代
        B, D, N = data.shape
        t = torch.randint(
            0, self.diffusion.num_timesteps, size=(B,), device=data.device
        )

        # 如果提供了噪声，则对其进行更新
        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        # 计算损失
        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises
        )
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(
        self,
        shape,
        device,
        noise_fn=torch.randn,
        constrain_fn=lambda x, t: x,
        clip_denoised=False,
        max_timestep=None,
        keep_running=False,
    ):
        # 生成样本
        return self.diffusion.p_sample_loop(
            self._denoise,
            shape=shape,
            device=device,
            noise_fn=noise_fn,
            constrain_fn=constrain_fn,
            clip_denoised=clip_denoised,
            max_timestep=max_timestep,
            keep_running=keep_running,
        )

    def reconstruct(self, x0, t, constrain_fn=lambda x, t: x):
        # 重构给定数据
        return self.diffusion.reconstruct(
            x0, t, self._denoise, constrain_fn=constrain_fn
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == "linear":
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == "warm0.1":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.2":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.5":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    """
    生成一个约束函数，用于在优化过程中约束形状
    :param target_shape_constraint: target voxels
    :return: constrained x
    """
    # 生成一系列的eps值，从大到小，用于逐步约束形状
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000) ** 2))

    def constrain_fn(x, t):
        # 根据当前时间步长选择合适的eps值
        eps_ = eps_all[t] if (t < 1000) else 0
        # 进行num_steps次约束操作
        for _ in range(num_steps):
            x = x - eps_ * ((x - ground_truth) * mask)

        return x

    return constrain_fn


#############################################################################


def get_dataset(dataroot, npoints, category, use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=[category],
        split="train",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True,
        use_mask=use_mask,
    )
    te_dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=[category],
        split="val",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask,
    )
    return tr_dataset, te_dataset


def evaluate_gen(opt, ref_pcs, logger):
    # 如果没有提供参考点云数据，则从测试数据集中生成它
    if ref_pcs is None:
        # 从测试数据集中加载数据
        _, test_dataset = get_dataset(
            opt.dataroot, opt.npoints, opt.category, use_mask=False
        )
        # 创建一个数据加载器来批量处理测试数据
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=False,
        )
        ref = []
        # 遍历测试数据的每个批次
        for data in tqdm(
            test_dataloader, total=len(test_dataloader), desc="Generating Samples"
        ):
            x = data["test_points"]
            m, s = data["mean"].float(), data["std"].float()

            # 对测试数据进行归一化，并将其添加到参考点云列表中
            ref.append(x * s + m)

        # 将参考点云列表合并为一个张量
        ref_pcs = torch.cat(ref, dim=0).contiguous()

    # 加载生成的点云样本
    logger.info("Loading sample path: %s" % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    # 打印生成的样本和参考点云的大小
    logger.info(
        "Generation sample size:%s reference size: %s"
        % (sample_pcs.size(), ref_pcs.size())
    )

    # 计算各种评估指标
    results = compute_all_metrics(sample_pcs, ref_pcs, opt.batch_size)
    # 将评估指标转换为字典，并将张量转换为标量值
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }

    # 打印评估指标
    pprint(results)
    logger.info(results)

    # 计算JSD距离
    jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
    pprint("JSD: {}".format(jsd))
    logger.info("JSD: {}".format(jsd))


def generate(model, opt):
    # 从测试数据集中加载数据
    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    # 创建一个数据加载器来批量处理测试数据
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        drop_last=False,
    )

    # 使用模型生成样本
    with torch.no_grad():
        samples = []  # 存储生成的样本
        ref = []  # 存储原始参考数据

        # 遍历测试数据的每个批次
        for i, data in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            desc="Generating Samples",
        ):
            x = data["test_points"].transpose(1, 2)  # 获取当前批次的测试数据
            m, s = data["mean"].float(), data["std"].float()  # 获取数据的均值和标准差
            print(x.shape)

            # 使用模型生成样本
            gen = model.gen_samples(x.shape, "cuda", clip_denoised=False).detach().cpu()

            # 对生成样本和测试数据进行归一化
            gen = gen.transpose(1, 2).contiguous()
            x = x.transpose(1, 2).contiguous()
            gen = gen * s + m
            x = x * s + m
            samples.append(gen)  # 将生成的样本添加到列表中
            ref.append(x)  # 将原始参考数据添加到列表中

            # 可视化生成的样本
            visualize_pointcloud_batch(
                os.path.join(str(Path(opt.eval_path).parent), "x.png"),
                gen[:64],
                None,
                None,
                None,
            )
        # 将生成的样本和参考数据合并为单个张量
        samples = torch.cat(samples, dim=0)
        ref = torch.cat(ref, dim=0)

        # 保存生成的样本到指定的路径
        torch.save(samples, opt.eval_path)

    # 返回原始参考数据
    return ref


def main(opt):
    # 如果模型类别为“飞机”，则设置相关参数
    if opt.category == "airplane":
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = "warm0.1"

    # 获取当前脚本的文件名，并去除扩展名
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    # 获取当前脚本所在的目录
    dir_id = os.path.dirname(__file__)
    # 设置输出目录，基于实验ID的子目录
    output_dir = get_output_dir(dir_id, exp_id)
    # 复制当前脚本到输出目录
    copy_source(__file__, output_dir)
    # 设置日志记录
    logger = setup_logging(output_dir)

    # 设置合成数据的输出子目录
    (outf_syn,) = setup_output_subdirs(output_dir, "syn")

    # 根据调度类型、beta起始和结束值以及时间数量来获取beta值列表
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    # 根据命令行参数和beta值列表来实例化模型
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    # 如果使用GPU，则将模型移至GPU
    if opt.cuda:
        model.cuda()

    # 定义一个内部函数来将模型包装为DataParallel，以支持多GPU训练
    def _transform_(m):
        return nn.parallel.DataParallel(m)

    # 使用上述函数来包装模型
    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    # 设置模型为评估模式
    model.eval()

    # 在评估过程中，加载之前保存的模型参数
    with torch.no_grad():
        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param["model_state"])

        # 如果命令行参数指定生成合成数据，则生成数据
        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, "samples.pth")
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref = generate(model, opt)

        # 如果命令行参数指定评估生成数据，则评估
        if opt.eval_gen:
            # 调用相应的函数来评估生成数据
            evaluate_gen(opt, ref, logger)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", default="/workspace/dataset/ShapeNet/ShapeNetCore.v2.PC15k"
    )
    parser.add_argument("--category", default="chair")

    parser.add_argument("--batch_size", type=int, default=50, help="input batch size")
    parser.add_argument("--workers", type=int, default=16, help="workers")
    parser.add_argument(
        "--niter", type=int, default=10000, help="number of epochs to train for"
    )

    parser.add_argument("--generate", default=True)
    parser.add_argument("--eval_gen", default=False)

    parser.add_argument("--nc", default=3)
    parser.add_argument("--npoints", default=5000)
    """model"""
    parser.add_argument("--beta_start", default=0.0001)
    parser.add_argument("--beta_end", default=0.02)
    parser.add_argument("--schedule_type", default="linear")
    parser.add_argument("--time_num", default=1000)

    # params
    parser.add_argument("--attention", default=True)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--loss_type", default="mse")
    parser.add_argument("--model_mean_type", default="eps")
    parser.add_argument("--model_var_type", default="fixedsmall")

    parser.add_argument(
        "--model",
        default="saved_model/chair_1799.pth",
        # required=True,
        help="path to model (to continue training)",
    )

    """eval"""

    parser.add_argument("--eval_path", default="")

    parser.add_argument("--manualSeed", default=42, type=int, help="random seed")

    parser.add_argument(
        "--gpu", type=int, default=0, metavar="S", help="gpu id (default: 0)"
    )

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt


if __name__ == "__main__":
    opt = parse_args()
    set_seed(opt)

    main(opt)
