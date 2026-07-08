from policy.algorithms import GoalConditionedDiffusionPolicyMLP
from policy.algorithms.networks import ResidualMLP


class GoalConditionedDiffusionPolicyResidualMLP(GoalConditionedDiffusionPolicyMLP):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_embedder = ResidualMLP(
            input_dim=self.task_dim,
            output_dim=self.state_embedding_dim,
            hidden_dims=self.hidden_dims,
        )
