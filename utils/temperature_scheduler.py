"""
温度退火调度器
==============

实现温度退火策略，用于控制聚类软分配的锐度

温度参数的作用:
    - 高温 (τ大): 软分配更平滑，梯度流畅
    - 低温 (τ小): 近似硬分配，聚类清晰

退火策略:
    训练初期使用高温，允许模型探索不同的聚类分配
    训练后期逐渐降温，使聚类边界更加清晰

退火曲线示例 (tau_init=1.0, tau_min=0.05, decay_rate=0.95):
    | Epoch | τ    | 阶段说明           |
    |-------|------|-------------------|
    | 0     | 1.00 | 初始，完全软分配   |
    | 10    | 0.60 | 逐渐锐化          |
    | 20    | 0.36 | 中期，聚类开始清晰 |
    | 30    | 0.21 | 聚类明确          |
    | 40    | 0.13 | 接近硬分配        |
    | 50    | 0.08 | 近似硬分配        |
    | 60+   | 0.05 | 最小值锁定        |
"""


class TemperatureScheduler:
    """
    温度退火调度器

    功能:
        管理温度参数的退火过程，从初始温度逐渐降到最小温度

    参数:
        tau_init: 初始温度，默认1.0
        tau_min: 最小温度，默认0.05
        decay_rate: 衰减率，默认0.95
        decay_strategy: 衰减策略，可选 'exponential' 或 'cosine'

    使用示例:
        scheduler = TemperatureScheduler(tau_init=1.0, tau_min=0.05)

        for epoch in range(epochs):
            tau = scheduler.get_tau(epoch)
            # 使用 tau 进行训练
            output = model(V_patch, tau=tau)
            ...
    """

    def __init__(self, tau_init=1.0, tau_min=0.05, decay_rate=0.95, decay_strategy='exponential'):
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.decay_rate = decay_rate
        self.decay_strategy = decay_strategy

        # 当前温度
        self.tau = tau_init

    def step(self):
        """
        更新温度（每个epoch调用一次）

        返回:
            tau: 更新后的温度
        """
        if self.decay_strategy == 'exponential':
            self.tau = max(self.tau_min, self.tau * self.decay_rate)
        else:
            # 保持当前值
            pass
        return self.tau

    def get_tau(self, epoch=None):
        """
        获取当前温度

        参数:
            epoch: 可选，直接计算指定epoch的温度

        返回:
            tau: 温度值
        """
        if epoch is None:
            return self.tau

        if self.decay_strategy == 'exponential':
            tau = self.tau_init * (self.decay_rate ** epoch)
            return max(tau, self.tau_min)
        elif self.decay_strategy == 'cosine':
            return self._cosine_decay(epoch)

        return self.tau

    def _cosine_decay(self, epoch, total_epochs=100):
        """
        余弦退火策略

        参数:
            epoch: 当前轮次
            total_epochs: 总轮次

        返回:
            tau: 温度值
        """
        import math
        progress = epoch / total_epochs
        tau = self.tau_min + 0.5 * (self.tau_init - self.tau_min) * (1 + math.cos(math.pi * progress))
        return max(tau, self.tau_min)

    def reset(self):
        """重置温度到初始值"""
        self.tau = self.tau_init

    def state_dict(self):
        """获取调度器状态"""
        return {
            'tau': self.tau,
            'tau_init': self.tau_init,
            'tau_min': self.tau_min,
            'decay_rate': self.decay_rate,
        }

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.tau = state_dict['tau']
        self.tau_init = state_dict['tau_init']
        self.tau_min = state_dict['tau_min']
        self.decay_rate = state_dict['decay_rate']
