import torch
from torch import nn
import gym
import torch.optim as optim
import time
import numpy as np
import os
from copy import deepcopy
from tensordict import TensorDict
from einops import rearrange

from storm.registry import TRAINER
from storm.metrics import ARR, SR, MDD, CR, SOR, VOL
from storm.metrics import NumTrades, AvgHoldPeriod, TurnoverRate, ActivityRate, AvgTradeInterval, BuyToSellRatio, NumBuys, NumSells
from storm.utils import save_json
from storm.environment import make_env
from storm.utils import build_storage
from storm.utils import ReplayBuffer

@TRAINER.register_module(force=True)
class CQLTrainer():

    def __init__(self,
                 *args,
                 config = None,
                 train_environment = None,
                 valid_environment = None,
                 test_environment = None,
                 agent = None,
                 logger = None,
                 writer = None,
                 wandb = None,
                 device = None,
                 **kwargs):
        self.config = config
        
        self.train_environment = train_environment
        self.valid_environment = valid_environment
        self.test_environment = test_environment

        self.train_environments = gym.vector.AsyncVectorEnv([
            make_env("Trading-v2", env_params=dict(env=deepcopy(train_environment),
                                                   transition_shape=config.transition_shape, seed=config.seed + i)) for
            i in range(config.num_envs)
        ])

        self.valid_environments = gym.vector.AsyncVectorEnv([
            make_env("Trading-v2", env_params=dict(env=deepcopy(valid_environment),
                                                   transition_shape=config.transition_shape, seed=config.seed + i)) for
            i in range(1)
        ])

        self.test_environments = gym.vector.AsyncVectorEnv([
            make_env("Trading-v2", env_params=dict(env=deepcopy(test_environment),
                                                   transition_shape=config.transition_shape, seed=config.seed + i)) for
            i in range(1)
        ])

        self.agent = agent

        self.value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.critic.parameters())),
                                     lr=config.value_learning_rate, eps=1e-5)

        self.logger = logger
        self.writer = writer
        self.wandb = wandb

        self.device = device

        self.global_step = 0
        self.check_index = 0
        self.save_index = 0

        self.storage = self.set_storage()

        self.replay_buffer = ReplayBuffer(
            buffer_size = config.replay_buffer_size,
            transition=config.transition,
            transition_shape=config.transition_shape,
        )

        self.use_bc = self.config.use_bc

    def set_storage(self):

        transition = self.config.transition
        transition_shape = self.config.transition_shape

        storage = TensorDict({}, batch_size=self.config.num_steps).to(self.device)
        for name in transition:
            assert name in transition_shape
            shape = (self.config.num_steps, * transition_shape[name]["shape"])
            type = transition_shape[name]["type"]
            storage[name] = build_storage(shape, type, self.device)

        return storage

    def flatten_storage(self, storage):
        flat_storage = {}
        for key, value in storage.items():
            flat_storage[key] = rearrange(value, 'b n ... -> (b n) ...')
        flat_storage = TensorDict(flat_storage, batch_size=self.config.num_steps * self.config.num_envs).to(self.device)
        return flat_storage

    def explore_environment(self,
                            init_state = None,
                            init_info = None,
                            reset = False):
        prefix = "train"
        if reset:
            state, info = self.train_environments.reset()
        else:
            state, info = init_state, init_info

        # To TensorDict
        next_obs = TensorDict({key: torch.Tensor(value) for key, value in state.items()},
                              batch_size=self.config.num_envs).to(self.device)
        next_done = torch.zeros(self.config.num_envs).to(self.device)

        # Exploring the environment
        for step in range(0, self.config.num_steps):
            self.global_step += 1 * self.config.num_envs

            action = torch.tensor(info["expert_action"]).to(self.device) # get expert action

            for key, value in next_obs.items(): # current state
                self.storage[key][step] = value
            self.storage["training_dones"][step] = next_done
            self.storage["training_actions"][step] = action

            next_obs, reward, done, truncted, info = self.train_environments.step(action.cpu().numpy())
            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()},
                                  batch_size=self.config.num_envs).to(self.device)
            real_next_obs = TensorDict({
                "next_features": next_obs["features"],
                "next_policy_trading_cashes": next_obs["policy_trading_cashes"],
                "next_policy_trading_positions": next_obs["policy_trading_positions"],
                "next_policy_trading_actions": next_obs["policy_trading_actions"],
                "next_policy_trading_rets": next_obs["policy_trading_rets"],
            }, batch_size=self.config.num_envs).to(self.device)
            for key, value in real_next_obs.items():
                self.storage[key][step] = value

            reward = torch.tensor(reward).to(self.device).view(-1)
            self.storage["training_rewards"][step] = reward

            next_done = torch.Tensor(done).to(self.device)

            if "final_info" in info:
                for info_item in info["final_info"]:
                    if info_item is not None:
                        self.logger.info(
                            f"global_step={self.global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                        self.writer.log_scalar(f"{prefix}/total_return", info_item["total_return"], self.global_step)
                        self.writer.log_scalar(f"{prefix}/total_profit", info_item["total_profit"], self.global_step)

                        wandb_dict = {
                            f"{prefix}/total_return": info_item["total_return"],
                            f"{prefix}/total_profit": info_item["total_profit"],
                        }
                        self.wandb.log(wandb_dict)

        self.replay_buffer.update(self.storage)

    def update_value(self, flat_storage, b_inds, info):

        loss = info["v_loss"]

        # update value
        np.random.shuffle(b_inds)

        for start in range(0, self.config.batch_size, self.config.value_minibatch_size):
            end = start + self.config.value_minibatch_size
            mb_inds = b_inds[start:end]

            with torch.no_grad():
                input_tensor = TensorDict({
                    "features": flat_storage["next_features"][mb_inds],
                    "cashes": flat_storage["next_policy_trading_cashes"][mb_inds],
                    "positions": flat_storage["next_policy_trading_positions"][mb_inds],
                    "actions": flat_storage["next_policy_trading_actions"][mb_inds],
                    "rets": flat_storage["next_policy_trading_rets"][mb_inds],
                }, batch_size=self.config.value_minibatch_size).to(self.device)

                next_target_q = self.agent(input_tensor, use_target=True)
                next_target_q = torch.max(next_target_q, dim=-1)[0]
                next_target_q = flat_storage["training_rewards"][mb_inds] + \
                        (1.0 - flat_storage["training_dones"][mb_inds]) * self.config.gamma * next_target_q

            input_tensor = TensorDict({
                "features": flat_storage["features"][mb_inds],
                "cashes": flat_storage["policy_trading_cashes"][mb_inds],
                "positions": flat_storage["policy_trading_positions"][mb_inds],
                "actions": flat_storage["policy_trading_actions"][mb_inds],
                "rets": flat_storage["policy_trading_rets"][mb_inds],
            }, batch_size=self.config.value_minibatch_size).to(self.device)

            q = self.agent(input_tensor, use_target=False)

            q_value = q.gather(1, flat_storage["training_actions"][mb_inds].unsqueeze(-1).long()).view(-1)

            q_loss = nn.functional.mse_loss(q_value, next_target_q)

            conservative_loss = self.config.con_coef * (torch.logsumexp(q, dim=-1).mean() - q_value.mean())

            loss = q_loss + conservative_loss

            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()

        res_info = {
            "v_loss": loss,
        }

        return res_info

    def train(self):

        start_time = time.time()

        state, info = self.train_environments.reset()

        num_updates = self.config.total_timesteps // self.config.batch_size
        warm_up_updates = self.config.warm_up_steps // self.config.batch_size

        is_warmup = True
        prefix = "train"
        for update in range(1, num_updates + 1 + warm_up_updates):
            if is_warmup and update > warm_up_updates:
                is_warmup = False

            # Annealing the rate if instructed to do so.
            if self.config.anneal_lr and not is_warmup:
                frac = 1.0 - (update - 1.0 - warm_up_updates) / num_updates
                self.value_optimizer.param_groups[0]["lr"] = frac * self.config.value_learning_rate

            # Explore the environment and collect data
            self.explore_environment(init_state=state, init_info=info, reset=False)

            # Flatten the storage
            flat_storage = self.replay_buffer.sample(self.config.batch_size).to(self.device)
            b_inds = np.arange(self.config.batch_size)

            trading_records = {
                "policy_update_steps": 0,
                "v_loss": torch.tensor(0),
            }

            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)

                # Update value
                update_value_res_info = self.update_value(flat_storage,
                                                          b_inds,
                                                          trading_records)
                trading_records.update(update_value_res_info)

                if is_warmup:
                    continue

                # update the target network
                for param, target_param in zip(self.agent.critic.parameters(), self.agent.target_critic.parameters()):
                    target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.log_scalar(f"{prefix}/value_learning_rate", self.value_optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.log_scalar(f"{prefix}/value_loss", trading_records["v_loss"].item(), self.global_step)
            self.writer.log_scalar(f"{prefix}/SPS", self.global_step / (time.time() - start_time), self.global_step)

            wandb_dict = {
                f"{prefix}/value_learning_rate": self.value_optimizer.param_groups[0]["lr"],
                f"{prefix}/value_loss": trading_records["v_loss"].item(),
                f"{prefix}/SPS": self.global_step / (time.time() - start_time),
            }
            self.wandb.log(wandb_dict)

            self.logger.info(f"SPS: {self.global_step}, {(time.time() - start_time)}")

            if self.global_step % self.config.check_steps >= self.check_index:
                self.valid(self.global_step)
                self.check_index += 1

            if self.global_step % self.config.save_steps >= self.save_index:
                torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps)))
                self.save_index += 1

        self.valid(self.global_step)
        torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps + 1)))

        self.train_environments.close()
        self.valid_environments.close()
        self.test_environments.close()
        self.writer.close()
        self.wandb.finish()

    def valid(self, global_step):

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
            "action_label": [],
        }

        # TRY NOT TO MODIFY: start the game
        state, info = self.valid_environments.reset()
        rets.append(info["ret"])

        next_obs = TensorDict({key: torch.Tensor(value) for key, value in state.items()}, batch_size=1).to(self.device)

        while True:

            # ALGO LOGIC: action logic
            with torch.no_grad():

                input_tensor = TensorDict(
                    {
                        "features": next_obs["features"],
                        "cashes": next_obs["policy_trading_cashes"],
                        "positions": next_obs["policy_trading_positions"],
                        "actions": next_obs["policy_trading_actions"],
                        "rets": next_obs["policy_trading_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)
                logits = self.agent.critic(input_tensor)
                action = torch.argmax(logits, dim=1)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncted, info = self.valid_environments.step(action.cpu().numpy())

            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()}, batch_size=1).to(self.device)

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
            trading_records["action"].append(info["action"])
            trading_records["action_label"].append(info["action_label"])

            if "final_info" in info:
                for info_item in info["final_info"]:
                    if info_item is not None:
                        self.logger.info(
                            f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                        self.writer.log_scalar(f"{prefix}/total_return", info_item["total_return"], global_step)
                        self.writer.log_scalar(f"{prefix}/total_profit", info_item["total_profit"], global_step)

                        wandb_dict = {
                            f"{prefix}/total_return": info_item["total_return"],
                            f"{prefix}/total_profit": info_item["total_profit"],
                        }

                        self.wandb.log(wandb_dict)

                break

        rets = np.array(rets)
        arr = ARR(rets)  # take as reward
        sr = SR(rets)
        dd = MDD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)

        positions = np.array(trading_records["position"]).flatten()
        turnover_rate = TurnoverRate(positions)

        actions = np.array(trading_records["action"]).flatten()
        num_trades = NumTrades(actions)
        num_buys = NumBuys(actions)
        num_sells = NumSells(actions)
        avg_hold_period = AvgHoldPeriod(actions)
        activity_rate = ActivityRate(actions)
        avg_trade_interval = AvgTradeInterval(actions)
        buy_to_sell_ratio = BuyToSellRatio(actions)

        self.writer.log_scalar(f"{prefix}/ARR%", arr * 100, global_step)
        self.writer.log_scalar(f"{prefix}/SR", sr, global_step)
        self.writer.log_scalar(f"{prefix}/CR", cr, global_step)
        self.writer.log_scalar(f"{prefix}/SOR", sor, global_step)
        self.writer.log_scalar(f"{prefix}/DD", dd, global_step)
        self.writer.log_scalar(f"{prefix}/MDD%", mdd * 100, global_step)
        self.writer.log_scalar(f"{prefix}/VOL", vol, global_step)
        self.writer.log_scalar(f"{prefix}/TurnoverRate", turnover_rate, global_step)
        self.writer.log_scalar(f"{prefix}/NumTrades", num_trades, global_step)
        self.writer.log_scalar(f"{prefix}/NumBuys", num_buys, global_step)
        self.writer.log_scalar(f"{prefix}/NumSells", num_sells, global_step)
        self.writer.log_scalar(f"{prefix}/AvgHoldPeriod", avg_hold_period, global_step)
        self.writer.log_scalar(f"{prefix}/ActivityRate", activity_rate, global_step)
        self.writer.log_scalar(f"{prefix}/AvgTradeInterval", avg_trade_interval, global_step)
        self.writer.log_scalar(f"{prefix}/BuyToSellRatio", buy_to_sell_ratio, global_step)

        wandb_dict = {
            f"{prefix}/ARR%": arr * 100,
            f"{prefix}/SR": sr,
            f"{prefix}/CR": cr,
            f"{prefix}/SOR": sor,
            f"{prefix}/DD": dd,
            f"{prefix}/MDD%": mdd * 100,
            f"{prefix}/VOL": vol,
            f"{prefix}/TurnoverRate": turnover_rate,
            f"{prefix}/NumTrades": num_trades,
            f"{prefix}/NumBuys": num_buys,
            f"{prefix}/NumSells": num_sells,
            f"{prefix}/AvgHoldPeriod": avg_hold_period,
            f"{prefix}/ActivityRate%": activity_rate * 100,
            f"{prefix}/AvgTradeInterval": avg_trade_interval,
            f"{prefix}/BuyToSellRatio": buy_to_sell_ratio,
        }
        self.wandb.log(wandb_dict)

        self.logger.info(
            f"global_step={global_step}, "
            f"ARR%={arr * 100}, "
            f"SR={sr}, "
            f"CR={cr}, "
            f"SOR={sor}, "
            f"DD={dd}, "
            f"MDD%={mdd * 100}, "
            f"VOL={vol}, "
            f"TurnoverRate={turnover_rate}, "
            f"NumTrades={num_trades}, "
            f"NumBuys={num_buys}, "
            f"NumSells={num_sells}, "
            f"AvgHoldPeriod={avg_hold_period}, "
            f"ActivityRate%={activity_rate * 100}, "
            f"AvgTradeInterval={avg_trade_interval}, "
            f"BuyToSellRatio={buy_to_sell_ratio}"
        )

        for key in trading_records.keys():
            trading_records[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in trading_records[key]]

        save_json(trading_records, os.path.join(self.config.exp_path, "valid_records.json"))