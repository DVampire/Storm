import sys
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from mmengine import DictAction
from copy import deepcopy
import torch
from torch import nn
import gym
import pathlib
from dotenv import load_dotenv
import torch.optim as optim
import time

load_dotenv(verbose=True)

root = str(pathlib.Path(__file__).resolve().parents[1])
current = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(current)
sys.path.append(root)

from storm.config import build_config
from storm.utils import assemble_project_path
from storm.utils import to_torch_dtype
from storm.log import Logger
from storm.log import TensorBoardLogger
from storm.log import WandbLogger
from storm.registry import DATASET
from storm.registry import ENVIRONMENT
from storm.registry import AGENT
from storm.environment import make_env
from storm.metrics import ARR, SR, MDD, CR, SOR, VOL
from storm.utils import save_json

def build_storage(shape, type, device):
    if type.startswith("int32"):
        type = torch.int32
    elif type.startswith("float32"):
        type = torch.float32
    elif type.startswith("int64"):
        type = torch.int64
    elif type.startswith("bool"):
        type = torch.bool
    else:
        type = torch.float32
    return torch.zeros(shape, dtype=type, device=device)

def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "agent", "AAPL_day_dj30_dynamic_dual_vqvae.py"), help="config file path")

    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--tensorboard_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=False)

    parser.add_argument("--writer", action="store_true", default=True, help="enable tensorboard")
    parser.add_argument("--no_writer", action="store_false", dest="writer")
    parser.set_defaults(writer=True)

    parser.add_argument("--wandb", action="store_true", default=True, help="enable wandb")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb")
    parser.set_defaults(wandb=True)

    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. build config
    config = build_config(assemble_project_path(args.config), args)

    # 2. set dtype
    dtype = to_torch_dtype(config.dtype)

    # 3. get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. init logger
    logger = Logger(config.log_path, accelerator = None)
    writer = None
    if config.writer:
        writer = TensorBoardLogger(config.tensorboard_path, accelerator = None)
    wandb = None
    if config.wandb:
        wandb = WandbLogger(project="storm",
                            name=config.tag,
                            config=config.to_dict(),
                            dir=config.wandb_path,
                            accelerator=None)

    dataset = DATASET.build(config.dataset)

    train_environment_config = deepcopy(config.train_environment)
    train_environment_config.update({"dataset": dataset})
    train_environment = ENVIRONMENT.build(train_environment_config)

    valid_environment_config = deepcopy(config.valid_environment)
    valid_environment_config.update({"dataset": dataset})
    valid_environment = ENVIRONMENT.build(valid_environment_config)

    test_environment_config = deepcopy(config.test_environment)
    test_environment_config.update({"dataset": dataset})
    test_environment = ENVIRONMENT.build(test_environment_config)

    train_environments = gym.vector.SyncVectorEnv([
        make_env("Trading-v0",env_params=dict(env = deepcopy(train_environment),
                                     transition_shape = config.transition_shape, seed = config.seed + i)) for i in range(config.num_envs)
    ])

    valid_environments = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(valid_environment),
                                                 transition_shape=config.transition_shape, seed=config.seed + i)) for i in range(1)
    ])

    test_environments = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(test_environment),
                                                 transition_shape=config.transition_shape, seed=config.seed + i)) for i in range(1)
    ])

    agent = AGENT.build(config.agent).to(device)

    policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, list(agent.actor.parameters())), lr=config.policy_learning_rate, eps=1e-5, weight_decay=0)
    value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.critic.parameters())), lr=config.value_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    transition_shape = config.transition_shape
    obs = build_storage(shape = (config.num_steps, *transition_shape["states"]["shape"]),
                        type = transition_shape["states"]["type"], device = device)
    actions = build_storage(shape = (config.num_steps, *transition_shape["actions"]["shape"]),
                            type = transition_shape["actions"]["type"], device = device)
    logprobs = build_storage(shape = (config.num_steps, *transition_shape["logprobs"]["shape"]),
                                type = transition_shape["logprobs"]["type"], device = device)
    rewards = build_storage(shape = (config.num_steps, *transition_shape["rewards"]["shape"]),
                                type = transition_shape["rewards"]["type"], device = device)
    dones = build_storage(shape = (config.num_steps, *transition_shape["dones"]["shape"]),
                                type = transition_shape["dones"]["type"], device = device)
    values = build_storage(shape = (config.num_steps, *transition_shape["values"]["shape"]),
                                type = transition_shape["values"]["type"], device = device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()

    state, info = train_environments.reset()

    next_obs = torch.Tensor(state).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size
    num_critic_warm_up_updates = config.critic_warm_up_steps // config.batch_size

    is_warmup = True
    prefix = "train"
    for update in range(1, num_updates + 1 + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False

        # Annealing the rate if instructed to do so.
        if config.anneal_lr and not is_warmup:
            frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
            policy_optimizer.param_groups[0]["lr"] = frac * config.policy_learning_rate
            value_optimizer.param_groups[0]["lr"] = frac * config.value_learning_rate

        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, truncted, info = train_environments.step(action.cpu().numpy())
            # print(info)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            if "final_info" in info:
                print("final_info", info["final_info"])
                for info_item in info["final_info"]:
                    if info_item is not None:
                        logger.info(f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                        writer.log_scalar(f"{prefix}/total_return", info_item["total_return"], global_step)
                        writer.log_scalar(f"{prefix}/total_profit", info_item["total_profit"], global_step)

                        wandb_dict = {
                            f"{prefix}/total_return": info_item["total_return"],
                            f"{prefix}/total_profit": info_item["total_profit"],
                        }
                        wandb.log(wandb_dict)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + transition_shape["states"]["shape"][1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.view(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        pg_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        old_approx_kl = torch.tensor(0)
        approx_kl = torch.tensor(0)
        total_approx_kl = torch.tensor(0)

        for epoch in range(config.update_epochs):
            if kl_explode:
                break
            # update value
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.value_minibatch_size):
                end = start + config.value_minibatch_size
                mb_inds = b_inds[start:end]
                newvalue = agent.get_value(b_obs[mb_inds])

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * config.vf_coef

                value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                value_optimizer.step()

            if is_warmup:
                continue

            policy_optimizer.zero_grad()
            # update policy
            for start in range(0, config.batch_size, config.policy_minibatch_size):
                if policy_update_steps % config.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + config.policy_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / config.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss
                loss /= config.gradient_checkpointing_steps

                loss.backward()

                if policy_update_steps % config.gradient_checkpointing_steps == 0:
                    if config.target_kl is not None:
                        if total_approx_kl > config.target_kl:
                            policy_optimizer.zero_grad()
                            kl_explode = True
                            policy_update_steps -= config.gradient_checkpointing_steps
                            # print("break", policy_update_steps)
                            break

                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    policy_optimizer.step()
                    policy_optimizer.zero_grad()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.log_scalar(f"{prefix}/policy_learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
        writer.log_scalar(f"{prefix}/value_learning_rate", value_optimizer.param_groups[0]["lr"], global_step)
        writer.log_scalar(f"{prefix}/value_loss", v_loss.item(), global_step)
        writer.log_scalar(f"{prefix}/policy_loss", pg_loss.item(), global_step)
        writer.log_scalar(f"{prefix}/entropy", entropy_loss.item(), global_step)
        writer.log_scalar(f"{prefix}/old_approx_kl", old_approx_kl.item(), global_step)
        writer.log_scalar(f"{prefix}/approx_kl", approx_kl.item(), global_step)
        writer.log_scalar(f"{prefix}/total_approx_kl", total_approx_kl.item(), global_step)
        writer.log_scalar(f"{prefix}/policy_update_times", policy_update_steps // config.gradient_checkpointing_steps, global_step)
        writer.log_scalar(f"{prefix}/clipfrac", num_clipfracs, global_step)
        writer.log_scalar(f"{prefix}/explained_variance", explained_var, global_step)
        writer.log_scalar(f"{prefix}/SPS", global_step / (time.time() - start_time), global_step)

        wandb_dict = {
            f"{prefix}/policy_learning_rate": policy_optimizer.param_groups[0]["lr"],
            f"{prefix}/value_learning_rate": value_optimizer.param_groups[0]["lr"],
            f"{prefix}/value_loss": v_loss.item(),
            f"{prefix}/policy_loss": pg_loss.item(),
            f"{prefix}/entropy": entropy_loss.item(),
            f"{prefix}/old_approx_kl": old_approx_kl.item(),
            f"{prefix}/approx_kl": approx_kl.item(),
            f"{prefix}/total_approx_kl": total_approx_kl.item(),
            f"{prefix}/policy_update_times": policy_update_steps // config.gradient_checkpointing_steps,
            f"{prefix}/clipfrac": num_clipfracs,
            f"{prefix}/explained_variance": explained_var,
            f"{prefix}/SPS": global_step / (time.time() - start_time),
        }
        wandb.log(wandb_dict)

        logger.info(f"SPS: {global_step}, {(time.time() - start_time)}")

        if global_step // config.check_steps != pre_global_step // config.check_steps:
            validate_agent(config, agent, valid_environments, logger, writer, wandb, device, global_step, config.exp_path)
            torch.save(agent.state_dict(), os.path.join(config.checkpoint_path, "{}.pth".format(global_step // config.check_steps)))
        pre_global_step = global_step

    validate_agent(config, agent, valid_environments, logger, writer, wandb, device, global_step, config.exp_path)
    torch.save(agent.state_dict(), os.path.join(config.checkpoint_path, "{}.pth".format(global_step // config.check_steps + 1)))

    train_environments.close()
    valid_environments.close()
    test_environments.close()
    writer.close()
    wandb.finish()

def validate_agent(config, agent, envs, logger, writer, wandb,  device, global_step, exp_path):

    prefix = "valid"

    rets = []
    trading_records = {
        "timestamp": [],
        "value": [],
        "cash": [],
        "position": [],
        "ret": [],
        "price": [],
        "discount": [],
        "total_profit": [],
        "total_return": [],
        "action": [],
    }

    # TRY NOT TO MODIFY: start the game
    state, info = envs.reset()
    rets.append(info["ret"])

    next_obs = torch.Tensor(state).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    while True:
        obs = next_obs
        dones = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            logits = agent.actor(next_obs)
            action = torch.argmax(logits, dim=1)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, truncted, info = envs.step(action.cpu().numpy())
        rets.append(info["ret"])
        trading_records["timestamp"].append(info["timestamp"])
        trading_records["value"].append(info["value"])
        trading_records["cash"].append(info["cash"])
        trading_records["position"].append(info["position"])
        trading_records["ret"].append(info["ret"])
        trading_records["price"].append(info["price"])
        trading_records["discount"].append(info["discount"])
        trading_records["total_profit"].append(info["total_profit"])
        trading_records["total_return"].append(info["total_return"])
        # trading_records["action"].append(envs.actions[action.cpu().numpy()])
        trading_records["action"].append(action.cpu().numpy())

        if trading_records["action"][-1] != info["action"]:
            trading_records["action"][-1] = info["action"]

        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor(done).to(device)

        if "final_info" in info:
            print("val final_info", info["final_info"])
            for info_item in info["final_info"]:
                if info_item is not None:
                    logger.info(f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                    writer.log_scalar(f"{prefix}/total_return", info_item["total_return"], global_step)
                    writer.log_scalar(f"{prefix}/total_profit", info_item["total_profit"], global_step)

                    wandb_dict = {
                        f"{prefix}/total_return": info_item["total_return"],
                        f"{prefix}/total_profit": info_item["total_profit"],
                    }

                    wandb.log(wandb_dict)

            break

    rets = np.array(rets)
    arr = ARR(rets)       # take as reward
    sr = SR(rets)
    dd = MDD(rets)
    mdd = MDD(rets)
    cr = CR(rets, mdd=mdd)
    sor = SOR(rets, dd=dd)
    vol = VOL(rets)

    writer.log_scalar(f"{prefix}/ARR%", arr * 100, global_step)
    writer.log_scalar(f"{prefix}/SR", sr, global_step)
    writer.log_scalar(f"{prefix}/CR", cr, global_step)
    writer.log_scalar(f"{prefix}/SOR", sor, global_step)
    writer.log_scalar(f"{prefix}/DD", dd, global_step)
    writer.log_scalar(f"{prefix}/MDD%", mdd * 100, global_step)
    writer.log_scalar(f"{prefix}/VOL", vol, global_step)

    wandb_dict = {
        f"{prefix}/ARR%": arr * 100,
        f"{prefix}/SR": sr,
        f"{prefix}/CR": cr,
        f"{prefix}/SOR": sor,
        f"{prefix}/DD": dd,
        f"{prefix}/MDD%": mdd * 100,
        f"{prefix}/VOL": vol,
    }
    wandb.log(wandb_dict)

    logger.info(
        f"global_step={global_step}, ARR%={arr * 100}, SR={sr}, CR={cr}, SOR={sor}, DD={dd}, MDD%={mdd * 100}, VOL={vol}"
    )

    # print(f"trading_records is   {trading_records}")
    for key in trading_records.keys():
        trading_records[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in trading_records[key]]

    save_json(trading_records, os.path.join(exp_path, "valid_records.json"))

if __name__ == '__main__':
    main()