import dataclasses
from pathlib import Path

from learned_planner.common import (
    catch_different_env_types_warning,
)
from learned_planner.train import BaseCommandConfig, create_vec_env_and_eval_callbacks, make_model


@dataclasses.dataclass
class EvaluateConfig(BaseCommandConfig):
    def __post_init__(self):
        if self.eval_env is None:
            self.eval_env = self.env
        assert self.eval_env == self.env, "env and eval_env must be the same for evaluation."

    def run(self, run_dir: Path):
        """Loads a policy that uses algorithm `args.alg` and policy `args.policy` and makes a video of the policy in the
        environment `args.env`. Makes sure that the video is viewable in wandb.
        """
        vec_env, eval_callbacks = create_vec_env_and_eval_callbacks(self, run_dir, eval_freq=1, save_model=False)
        assert self.load_path is not None
        model = make_model(self, run_dir, vec_env, eval_callbacks)

        for eval_callback in eval_callbacks:
            with catch_different_env_types_warning(vec_env, [eval_callback]):
                eval_callback.init_callback(model)
            eval_callback.on_training_start(locals(), globals())
            eval_callback.on_step()

        for eval_callback in eval_callbacks:
            eval_callback.on_training_end()

        model.logger.close()
