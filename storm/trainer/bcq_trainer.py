import time
import torch
import gym
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
from copy import deepcopy


from storm.registry import TRAINER
from storm.utils import build_storage
from storm.metrics import ARR, SR, MDD, CR, SOR, VOL
from storm.metrics import NumTrades, AvgHoldPeriod, TurnoverRate, ActivityRate, AvgTradeInterval, BuyToSellRatio
from storm.utils import save_json
from storm.utils import ReplayBuffer
from storm.environment import make_env


@TRAINER.register_module(force=True)
class BCQTrainer():

    def __init__(self,
                 *args,
                 config=None,
                 train_environment=None,
                 valid_environment=None,
                 test_environment=None,
                 agent=None,
                 logger=None,
                 writer=None,
                 wandb=None,
                 device=None,
                 **kwargs
                 ):

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

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.qnet.parameters())),
                                    lr=config.learning_rate, eps=1e-5)

        self.logger = logger
        self.writer = writer
        self.wandb = wandb

        self.device = device

        self.replay_buffer_device = torch.device("cpu")
        self.replay_buffer = ReplayBuffer(buffer_size = config.replay_buffer_size,
                                          transition=config.transition,
                                          transition_shape=config.transition_shape,
                                          num_envs=config.num_envs,
                                          device=self.replay_buffer_device)

    def train(self):

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        check_index = 0
        save_index = 0
        start_time = time.time()

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
                self.optimizer.param_groups[0]["lr"] = frac * self.config.learning_rate

            # Sample a trajectory
            state, info = self.train_environments.reset()

            # Get the trajectory
            dp_trajectory = info["dp_datas"][0] # 0 is the first environment, which is the only one in this case

            update_arrays = {}
            for item in dp_trajectory:
                for key, value in item.items():
                    update_arrays.setdefault(key, []).append(value)

            # Convert the trajectory to tensors
            update_tensors = {}
            for key, value in update_arrays.items():
                update_tensors[key] = torch.from_numpy(np.stack(value, axis=0)).to(self.replay_buffer_device).unsqueeze(1)

            # Update the replay buffer
            self.replay_buffer.update(update_tensors)

            for epoch in range(self.config.update_epochs):

                for start in range(0, self.config.batch_size, self.config.mini_batch_size):

                    # Sample a batch
                    batch = self.replay_buffer.sample(self.config.mini_batch_size)

                    # To device
                    for key, value in batch.items():
                        batch[key] = value.to(self.device)

                    features = batch["features"]
                    reward = batch["rets"]
                    done = batch["dones"]
                    action = batch["actions"]

                    # Compute the target Q value
                    with torch.no_grad():
                        next_action = self.agent.get_action(batch)

                        target_value = self.agent.get_value(batch)

                        target_value = reward + done * self.config.discount * target_value.gather(1, next_action.to(torch.int64).unsqueeze(-1))

                    # Get current Q estimate
                    value, scaled_score, score= self.agent(batch)
                    value = value.gather(1, action.to(torch.int64))

                    # Compute Q loss
                    value_loss = F.smooth_l1_loss(value, target_value)

                    action_loss = F.nll_loss(scaled_score, action.squeeze(-1).long())

                    loss = value_loss + action_loss + 1e-2 * score.pow(2).mean()

                    # Optimize the Q
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    global_step += 1

            # Update the target Q network
            self.agent.soft_update()

            self.writer.log_scalar(f"{prefix}/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.log_scalar(f"{prefix}/loss", loss.item(), global_step)
            self.writer.log_scalar(f"{prefix}/SPS", global_step / (time.time() - start_time), global_step)

            wandb_dict = {
                f"{prefix}/learning_rate": self.optimizer.param_groups[0]["lr"],
                f"{prefix}/loss": loss.item(),
                f"{prefix}/SPS": global_step / (time.time() - start_time),
            }
            self.wandb.log(wandb_dict)

            self.logger.info(f"SPS: {global_step}, {(time.time() - start_time)}")

            if global_step // self.config.check_steps >= check_index:
                self.valid(global_step)
                check_index += 1

            if global_step % self.config.save_steps >= save_index:
                torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(
                    global_step // self.config.save_steps)))
                save_index += 1

        self.valid(global_step)
        torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(
            global_step // self.config.save_steps + 1)))

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

        tensors = {}
        for key, value in state.items():
            tensors[key] = torch.from_numpy(value).squeeze(1).to(self.device) # remove the env dimension, because it is 1 in this case

        while True:

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, _, _ = self.agent(tensors)
                action = action.argmax(dim=-1)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncted, info = self.valid_environments.step(action.cpu().numpy())

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

            tensors = {}
            for key, value in next_obs.items():
                tensors[key] = torch.from_numpy(value).squeeze(1).to(self.device)  # remove the env dimension, because it is 1 in this case

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
            f"AvgHoldPeriod={avg_hold_period}, "
            f"ActivityRate%={activity_rate * 100}, "
            f"AvgTradeInterval={avg_trade_interval}, "
            f"BuyToSellRatio={buy_to_sell_ratio}"
        )

        for key in trading_records.keys():
            trading_records[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in trading_records[key]]

        save_json(trading_records, os.path.join(self.config.exp_path, "valid_records.json"))