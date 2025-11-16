import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯용 (일부 버전은 안 써도 되지만 호환 위해 추가)

# =======================
# 1. 하이퍼파라미터 / 설정
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_MAX = 5.0          # 학습할 시간 구간: [0, T_MAX]
N_COLLOCATION = 4000 # 물리 방정식(ODE) 잔차를 계산할 시간 샘플 개수
N_IC = 64            # 초기조건에 사용할 t=0 샘플 개수
EPOCHS = 20000
LR = 1e-3

G = 1.0  # 중력 상수 (스케일 맞추기 용; 편의상 1로 둠)

# 질량 (3개의 질량)
masses = torch.tensor([1.0, 1.0, 1.0], device=device)  # [3]


# =======================
# 2. 삼체 문제 초기 조건 설정 (3D)
# =======================
# r_i = (x, y, z), v_i = (vx, vy, vz)
r1_0 = torch.tensor([-1.0, 0.0,  0.0], device=device)
r2_0 = torch.tensor([ 1.0, 0.0,  0.0], device=device)
r3_0 = torch.tensor([ 0.0, 0.5, 0.5], device=device)

v1_0 = torch.tensor([0.0,  0.6,  0.0], device=device)
v2_0 = torch.tensor([0.0, -0.6,  0.0], device=device)
v3_0 = torch.tensor([0.0,  0.0,  0.0], device=device)

# (3, 3) 형태로 모으기
R0 = torch.stack([r1_0, r2_0, r3_0], dim=0)  # [3, 3]
V0 = torch.stack([v1_0, v2_0, v3_0], dim=0)  # [3, 3]


# =======================
# 3. Residual Block + PINN 모델 정의
#    입력: t (스칼라)
#    출력: 3개 물체의 3D 위치 (총 9차원)
# =======================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        # x: [N, dim]
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + x)  # skip connection + Tanh


class ThreeBodyPINN(nn.Module):
    def __init__(self, hidden_dim=128, num_blocks=4):
        super().__init__()
        in_dim = 1
        out_dim = 9  # (x1,y1,z1,x2,y2,z2,x3,y3,z3)

        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, t):
        """
        t: [N, 1]
        return: [N, 9]
        """
        # time normalization: [0, T_MAX] -> [-1, 1]
        t_norm = 2.0 * (t / T_MAX) - 1.0
        x = self.act(self.input(t_norm))
        for blk in self.blocks:
            x = blk(x)
        y = self.output(x)
        return y


model = ThreeBodyPINN().to(device)


# =======================
# 4. autograd로 1, 2차 시간 미분 구하기
# =======================
def time_derivatives(t, y):
    """
    t: [N, 1] (requires_grad=True)
    y: [N, D]  (D=9)

    반환:
        dy_dt: [N, D]
        d2y_dt2: [N, D]
    """
    dy_dt_list = []
    d2y_dt2_list = []

    for i in range(y.shape[1]):
        yi = y[:, i:i+1]  # [N,1]

        dyi_dt = torch.autograd.grad(
            outputs=yi,
            inputs=t,
            grad_outputs=torch.ones_like(yi),
            create_graph=True,
            retain_graph=True
        )[0]  # [N,1]

        d2yi_dt2 = torch.autograd.grad(
            outputs=dyi_dt,
            inputs=t,
            grad_outputs=torch.ones_like(dyi_dt),
            create_graph=True,
            retain_graph=True
        )[0]  # [N,1]

        dy_dt_list.append(dyi_dt)
        d2y_dt2_list.append(d2yi_dt2)

    dy_dt = torch.cat(dy_dt_list, dim=1)      # [N,D]
    d2y_dt2 = torch.cat(d2y_dt2_list, dim=1)  # [N,D]

    return dy_dt, d2y_dt2


# =======================
# 5. 중력 가속도 계산 (3D)
# =======================
def gravitational_accelerations(R, masses, G=1.0):
    """
    R: [N, 3, 3]  - 각 시점마다 3개의 물체 위치 (3D)
    masses: [3]
    반환: a: [N, 3, 3]
    """
    N = R.shape[0]
    a = torch.zeros_like(R)

    for i in range(3):
        ri = R[:, i, :]  # [N,3]
        acc_i = torch.zeros_like(ri)

        for j in range(3):
            if i == j:
                continue
            rj = R[:, j, :]  # [N,3]
            diff = rj - ri   # [N,3]
            dist_sq = torch.sum(diff**2, dim=1, keepdim=True) + 1e-6  # [N,1]
            dist_three = dist_sq * torch.sqrt(dist_sq)               # |r|^3

            acc_i = acc_i + G * masses[j] * diff / dist_three  # [N,3]

        a[:, i, :] = acc_i

    return a  # [N,3,3]


# =======================
# 6. 에너지 계산 (K + V)
# =======================
def total_energy(R, V, masses, G=1.0):
    """
    R: [N, 3, 3] 위치
    V: [N, 3, 3] 속도
    masses: [3]
    반환: E: [N]
    """
    # kinetic
    # V^2 sum over dim=-1 -> [N,3]
    v_sq = torch.sum(V**2, dim=-1)
    m = masses.unsqueeze(0)  # [1,3]
    T = 0.5 * torch.sum(m * v_sq, dim=1)  # [N]

    # potential
    N = R.shape[0]
    U = torch.zeros(N, device=R.device)
    for i in range(3):
        for j in range(i+1, 3):
            ri = R[:, i, :]  # [N,3]
            rj = R[:, j, :]
            diff = ri - rj
            dist = torch.sqrt(torch.sum(diff**2, dim=1) + 1e-6)  # [N]
            U_ij = -G * masses[i] * masses[j] / dist
            U += U_ij

    E = T + U
    return E  # [N]


# 초기 에너지 (reference, scalar)
E0 = total_energy(R0.unsqueeze(0), V0.unsqueeze(0), masses, G=G)[0].detach()


# =======================
# 7. 손실 함수 정의 (커리큘럼 + 에너지 보존 + 가중치 스케줄링)
# =======================
def pinn_loss(model, epoch, total_epochs):
    # ---- 커리큘럼: 초반에는 짧은 시간 구간만 ----
    # 예: 전체의 30%까지는 t_max를 선형 증가
    frac = min(1.0, epoch / (0.3 * total_epochs))
    t_max_curr = T_MAX * frac

    # ---- Loss 가중치 스케줄링 ----
    # 초반엔 IC를 강하게, 후반엔 physics/energy를 더 강조
    lambda_ic = 10.0 * (1.0 - 0.5 * frac)   # 10 -> 5 정도로 감소
    lambda_phys = 1.0 + 4.0 * frac          # 1 -> 5로 증가
    lambda_energy = 1.0                     # 에너지 보존은 계속 일정하게

    # ---- (1) collocation points: 물리 방정식 손실 ----
    t_coll = torch.rand(N_COLLOCATION, 1, device=device) * t_max_curr
    t_coll.requires_grad_(True)

    y_coll = model(t_coll)  # [N,9]
    dy_dt, d2y_dt2 = time_derivatives(t_coll, y_coll)  # [N,9], [N,9]

    R = y_coll.view(-1, 3, 3)        # [N,3,3]
    d2R_dt2 = d2y_dt2.view(-1, 3, 3) # [N,3,3]

    a_grav = gravitational_accelerations(R, masses, G=G)  # [N,3,3]

    physics_residual = d2R_dt2 - a_grav  # [N,3,3]
    loss_physics = torch.mean(physics_residual**2)

    # 속도도 구해서 에너지 보존에 활용
    V_coll = dy_dt.view(-1, 3, 3)  # [N,3,3]
    E_coll = total_energy(R, V_coll, masses, G=G)  # [N]
    loss_energy = torch.mean((E_coll - E0)**2)

    # ---- (2) 초기 조건 손실 ----
    t_ic = torch.zeros(N_IC, 1, device=device)
    t_ic.requires_grad_(True)

    y_ic = model(t_ic)                    # [N_IC, 9]
    dy_ic_dt, _ = time_derivatives(t_ic, y_ic)

    R_ic = y_ic.view(-1, 3, 3)            # [N_IC,3,3]
    V_ic = dy_ic_dt.view(-1, 3, 3)        # [N_IC,3,3]

    R0_expanded = R0.unsqueeze(0).expand_as(R_ic)  # [N_IC,3,3]
    V0_expanded = V0.unsqueeze(0).expand_as(V_ic)  # [N_IC,3,3]

    loss_ic_pos = torch.mean((R_ic - R0_expanded)**2)
    loss_ic_vel = torch.mean((V_ic - V0_expanded)**2)

    loss_ic = loss_ic_pos + loss_ic_vel

    # ---- (3) 총 손실 ----
    loss = (
        lambda_phys * loss_physics +
        lambda_ic * loss_ic +
        lambda_energy * loss_energy
    )

    return loss, loss_physics, loss_ic, loss_energy, lambda_phys, lambda_ic


# =======================
# 8. 학습 루프
# =======================
optimizer = optim.Adam(model.parameters(), lr=LR)
print("Using device:", device)

for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    loss, loss_phys, loss_ic, loss_energy, lambda_phys, lambda_ic = pinn_loss(model, epoch, EPOCHS)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"[{epoch}/{EPOCHS}] "
            f"loss = {loss.item():.4e} | "
            f"phys = {loss_phys.item():.4e} (λ={lambda_phys:.2f}) | "
            f"IC = {loss_ic.item():.4e} (λ={lambda_ic:.2f}) | "
            f"energy = {loss_energy.item():.4e}"
        )

print("학습 종료!")

# ------ 학습된 모델 저장 ------
save_path = "three_body_pinn.pt"
torch.save(model.state_dict(), save_path)
print(f"모델 파라미터를 '{save_path}' 파일로 저장했습니다.")


# =======================
# 9. 학습된 모델로 궤적 샘플링
# =======================
@torch.no_grad()
def sample_trajectory(model, num_steps=400):
    t = torch.linspace(0.0, T_MAX, num_steps, device=device).view(-1, 1)
    y = model(t)              # [num_steps, 9]
    R = y.view(-1, 3, 3)      # [num_steps, 3, 3]
    return t.cpu(), R.cpu()

t_sample, R_sample = sample_trajectory(model)  # R_sample[k, i, :] = k번째 시각에서 i번째 물체 위치 (x,y,z)


# =======================
# 10. matplotlib으로 3D 궤적 플롯
# =======================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 각 물체에 대해 궤적 그리기
colors = ['tab:red', 'tab:blue', 'tab:green']
labels = ['Body 1', 'Body 2', 'Body 3']

for i in range(3):
    xi = R_sample[:, i, 0]
    yi = R_sample[:, i, 1]
    zi = R_sample[:, i, 2]

    ax.plot(xi, yi, zi, color=colors[i], label=labels[i])
    # 시작점 표시
    ax.scatter(xi[0], yi[0], zi[0], color=colors[i], marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Three-Body Problem Trajectories (PINN + Residual + Energy)')
ax.legend()
ax.grid(True)

# 축 비율을 동일하게 맞추기 (3D에서 중요)
x_all = R_sample[:, :, 0].numpy().flatten()
y_all = R_sample[:, :, 1].numpy().flatten()
z_all = R_sample[:, :, 2].numpy().flatten()
max_range = max(x_all.max() - x_all.min(),
                y_all.max() - y_all.min(),
                z_all.max() - z_all.min()) / 2.0

mid_x = (x_all.max() + x_all.min()) * 0.5
mid_y = (y_all.max() + y_all.min()) * 0.5
mid_z = (z_all.max() + z_all.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()
