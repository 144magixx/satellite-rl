"""离散联合动作的双 Q 期望；不依赖训练入口或环境。"""
import torch


def expected_min_joint_q(probabilities, q1, q2, frequency_chunk=25):
    """先比较完整动作 Q，再按三个动作概率求期望，保留概率梯度。"""
    pf, ps, pp = probabilities
    result = torch.zeros_like(pf[:, :1])
    for start in range(0, pf.shape[1], frequency_chunk):
        stop = min(start + frequency_chunk, pf.shape[1])
        joint1 = (q1[0][:, start:stop, None, None]
                  + q1[1][:, None, :, None] + q1[2][:, None, None, :])
        joint2 = (q2[0][:, start:stop, None, None]
                  + q2[1][:, None, :, None] + q2[2][:, None, None, :])
        values = (torch.minimum(joint1, joint2) * pp[:, None, None, :]).sum(-1)
        values = (values * ps[:, None, :]).sum(-1)
        result = result + (values * pf[:, start:stop]).sum(-1, keepdim=True)
    return result
