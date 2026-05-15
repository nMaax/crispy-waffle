import torch

from policy.algorithms.state_translator import StateTranslator


class MultiTaskStateTranslator(StateTranslator):
    def forward(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            raise ValueError("Network is not configured.")

        x_norm = self.x_normalizer.normalize(x)
        y_norm_pred = self.network(x_norm, task_idx)
        y_pred = self.y_normalizer.unnormalize(y_norm_pred)
        return y_pred

    def _compute_loss(self, batch) -> torch.Tensor:
        if self.network is None:
            raise ValueError("Network is not configured.")

        x, y, task_idx = batch

        x_norm = self.x_normalizer.normalize(x)
        y_norm = self.y_normalizer.normalize(y)

        y_norm_hat = self.network(x_norm, task_idx)

        if getattr(self, "loss_mask", None) is not None:
            y_norm_hat = y_norm_hat[..., self.loss_mask]
            y_norm = y_norm[..., self.loss_mask]

        return torch.nn.functional.mse_loss(y_norm_hat, y_norm)

    def _configure_normalizers(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise ValueError("Datamodule is not available.")

        all_x, all_y = [], []

        for task_dataset in dm.train_set.datasets:
            base_dataset = task_dataset.base_translator_dataset.base_dataset

            for traj in base_dataset.trajectories:
                if base_dataset.lazy:
                    ep_id = traj["episode_id"]
                    h5_traj = base_dataset.h5_file[f"traj_{ep_id}"]
                    x_ep = torch.from_numpy(h5_traj["obs"][:])
                else:
                    x_ep = torch.from_numpy(traj["obs"])

                with torch.no_grad():
                    canonical_x = task_dataset.canonical_adapter.apply(x_ep)
                    target_y = task_dataset.base_translator_dataset.adapter.apply(x_ep)

                all_x.append(canonical_x)
                all_y.append(target_y)

        self.x_normalizer.fit(torch.cat(all_x, dim=0))
        self.y_normalizer.fit(torch.cat(all_y, dim=0))
