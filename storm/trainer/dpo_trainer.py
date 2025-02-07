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

@TRAINER.register_module(force=True)
class DPOTrainer():

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

        self.policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, list(agent.actor.parameters())),
                                       lr=config.policy_learning_rate, eps=1e-5, weight_decay=0)

        self.logger = logger
        self.writer = writer
        self.wandb = wandb

        self.device = device

        self.global_step = 0
        self.check_index = 0
        self.save_index = 0

        self.storage = self.set_storage()

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

        # Exploring the environment
        for step in range(0, self.config.num_steps):
            self.global_step += 1 * self.config.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # Construct the input tensor
                input_tensor = TensorDict(
                    {
                        "features": next_obs["features"],
                        "cashes": next_obs["policy_trading_cashes"],
                        "positions": next_obs["policy_trading_positions"],
                        "actions": next_obs["policy_trading_actions"],
                        "rets": next_obs["policy_trading_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)
                actions, _ = self.agent.get_action(input_tensor)

            evaluate_rewards = info["evaluate_rewards"]
            evaluate_rewards = np.stack(evaluate_rewards, axis=0).astype(np.float32)  # (num_envs, num_actions)
            evaluate_rewards = torch.tensor(evaluate_rewards, dtype=torch.float32, device=self.device) # (num_envs, num_actions)

            for key, value in next_obs.items():
                self.storage[key][step] = value
            self.storage["training_evaluate_rewards"][step] = evaluate_rewards

            next_obs, reward, done, truncted, info = self.train_environments.step(actions.cpu().numpy())
            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()},
                                  batch_size=self.config.num_envs).to(self.device)

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

    def preference_loss(self, policy_chosen_logps,
                        policy_rejected_logps,
                        reference_chosen_logps,
                        reference_rejected_logps,
                        beta = 0.1):
        logits = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
        losses = - torch.nn.functional.logsigmoid(beta * logits)
        return losses.mean()

    def update_policy(self,
                      flat_storage,
                      b_inds,
                      info):

        policy_update_steps = info["policy_update_steps"]
        loss = info["loss"]

        # update policy
        for start in range(0, self.config.batch_size, self.config.policy_minibatch_size):

            policy_update_steps += 1
            end = start + self.config.policy_minibatch_size

            mb_inds = b_inds[start:end]

            input_tensor = TensorDict({
                "features": flat_storage["features"][mb_inds],
                "cashes": flat_storage["policy_trading_cashes"][mb_inds],
                "positions": flat_storage["policy_trading_positions"][mb_inds],
                "actions": flat_storage["policy_trading_actions"][mb_inds],
                "rets": flat_storage["policy_trading_rets"][mb_inds],
            }, batch_size=self.config.policy_minibatch_size).to(self.device)

            _, policy_probs = self.agent.get_action(input_tensor)

            evaluate_rewards = flat_storage["training_evaluate_rewards"][mb_inds]

            # Find the action with the maximum reward for each environment
            max_reward_action = torch.argmax(evaluate_rewards, dim=1)  # Shape: (num_envs,)

            # Reshape max_reward_action to (batch_size, 1) for broadcasting
            chosen_action = max_reward_action.unsqueeze(1)  # Shape: (num_envs, 1)
            batch_size, num_actions = evaluate_rewards.shape  # Get batch size and number of actions

            # Create a mask for rejected actions (all actions except chosen_action)
            rejected_mask = torch.ones((batch_size, num_actions), dtype=torch.bool, device=evaluate_rewards.device)
            rejected_mask.scatter_(1, chosen_action, False)  # Set chosen_action indices to False

            # Compute log probabilities for the policy
            policy_logps = torch.log(policy_probs + 1e-8)  # Shape: (num_envs, num_actions)
            policy_chosen_logps = policy_logps.gather(1, chosen_action)  # Shape: (num_envs, 1)
            policy_rejected_logps = policy_logps[rejected_mask].view(batch_size, -1)  # Shape: (num_envs, num_actions - 1)

            # Normalize rewards to create reference probabilities
            reference_rewards = evaluate_rewards  # Shape: (num_envs, num_actions)
            reference_probs = torch.softmax(reference_rewards, dim=1)  # Normalize along actions, shape: (num_envs, num_actions)
            reference_logps = torch.log(reference_probs + 1e-8)  # Convert probabilities to log probabilities

            # Extract reference log probabilities for chosen and rejected actions
            reference_chosen_logps = reference_logps.gather(1, chosen_action)  # Shape: (num_envs, 1)
            reference_rejected_logps = reference_logps[rejected_mask].view(batch_size, -1)  # Shape: (num_envs, num_actions - 1)

            # Compute preference loss
            loss = self.preference_loss(policy_chosen_logps,
                                        policy_rejected_logps,
                                        reference_chosen_logps,
                                        reference_rejected_logps,
                                        beta=self.config.beta)

            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()

        info = {
            "policy_update_steps": policy_update_steps,
            "loss": loss
        }

        return info

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
                self.policy_optimizer.param_groups[0]["lr"] = frac * self.config.policy_learning_rate

            self.explore_environment(init_state=state, init_info=info, reset=False)

            flat_storage = self.flatten_storage(self.storage)

            b_inds = np.arange(self.config.num_envs * self.config.num_steps)
            np.random.shuffle(b_inds)

            trading_records = {
                "policy_update_steps": 0,
                "loss": torch.tensor(0),
            }
            for epoch in range(self.config.update_epochs):

                update_policy_info = self.update_policy(flat_storage,
                                          b_inds,
                                          trading_records)
                trading_records.update(update_policy_info)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.log_scalar(f"{prefix}/policy_learning_rate", self.policy_optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.log_scalar(f"{prefix}/loss", trading_records["loss"].item(), self.global_step)
            self.writer.log_scalar(f"{prefix}/SPS", self.global_step / (time.time() - start_time), self.global_step)

            wandb_dict = {
                f"{prefix}/policy_learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                f"{prefix}/loss": trading_records["loss"].item(),
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
                logits = self.agent.actor(input_tensor)
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