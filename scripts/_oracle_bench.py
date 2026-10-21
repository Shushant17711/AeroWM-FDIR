import torch, sys
from agents.pid_baseline import CascadePIDController
from envs.rocket_vec_env import VectorizedRocketEnv
dev=torch.device('cuda')
n=int(sys.argv[1]) if len(sys.argv)>1 else 20
onset=int(sys.argv[2]) if len(sys.argv)>2 else 150
oracle=(sys.argv[3] if len(sys.argv)>3 else '1')=='1'
env=VectorizedRocketEnv(num_envs=8,device=dev,seed=0)
tot=[0,0]
for mode in range(5):
    c=CascadePIDController(device=dev,oracle_fault_compensation=oracle)
    env.fixed_fault_mode=mode; s=t=0; crash=to=0; attempts=0
    while t<n and attempts<200:
        attempts+=1
        env.reset(mode=mode,probability=1.0 if mode else 0.0,onset=onset,severity=0.35 if mode==1 else 0.5)
        c.reset(8,dev); obs=env._observation()
        for step in range(env.env_cfg.max_episode_steps):
            out=env.step(c.act(obs,fault_state=env.fault if oracle else None)); obs=out.obs
            d=out.terminated|out.truncated
            if d.any():
                land=out.info['landing_success']&out.terminated
                s+=int(land.sum()); crash+=int((out.terminated&~land).sum()); to+=int(out.truncated.sum())
                t+=int(d.sum())
                if t>=n: break
    r=s/max(1,s+crash+to); tot[0]+=s; tot[1]+=s+crash+to
    print(f"{['NOMINAL','LOE','BIAS','STUCK','COM'][mode]:8s} {s}/{s+crash+to} = {r:.3f}  (crash {crash}, timeout {to})")
print(f"MEAN {tot[0]/max(1,tot[1]):.3f}")
