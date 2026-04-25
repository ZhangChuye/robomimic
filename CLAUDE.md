# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
pip install -e .
```

Core dependencies: PyTorch, h5py, numpy, tensorboard, imageio, huggingface_hub, transformers, diffusers.

## Common Commands

**Training:**
```bash
python robomimic/scripts/train.py --config /path/to/config.json
python robomimic/scripts/train.py --algo bc --dataset /path/to/dataset.hdf5 --debug
```

**Policy evaluation:**
```bash
python robomimic/scripts/run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400
python robomimic/scripts/run_trained_agent.py --agent model.pth --video_path out.mp4
```

**Dataset utilities:**
```bash
python robomimic/scripts/download_datasets.py
python robomimic/scripts/playback_dataset.py --dataset data.hdf5
python robomimic/scripts/dataset_states_to_obs.py --input states.hdf5 --output obs.hdf5
```

**Running tests:**
```bash
cd tests && bash test.sh          # run all tests
cd tests && python test_bc.py     # run a single algorithm test
```

Tests use a custom `robomimic.utils.test_utils` framework (not pytest). Each test file has a top-level `test_*()` function and supports a `--verbose` flag.

## Architecture

### Factory / Registry Pattern
Both algorithms and configs use a global registry keyed by algorithm name. To add a new algorithm:
1. Create `robomimic/algo/my_algo.py` subclassing `Algo` (or `PolicyAlgo`/`ValueAlgo`)
2. Decorate a factory function with `@register_algo_factory_func("my_algo")`
3. Create `robomimic/config/my_algo_config.py` with `ALGO_NAME = "my_algo"` — the `ConfigMeta` metaclass auto-registers it

### Config System (`robomimic/config/`)
`BaseConfig` (via `ConfigMeta`) auto-registers subclasses. Once `lock_keys()` is called, no new keys can be added — this catches typos at config load time. Config has five top-level sections:
- `experiment` — name, logging (tensorboard/wandb), checkpoint saving, rollout frequency
- `train` — dataset paths, batch size, optimizer, data augmentation
- `algo` — algorithm-specific hyperparameters
- `observation` — modality definitions (low_dim, rgb, depth, scan, point_cloud) and encoder specs
- `meta` — dataset metadata passthrough

### Algorithm Classes (`robomimic/algo/`)
Base hierarchy: `Algo → PolicyAlgo / ValueAlgo / PlannerAlgo → HierarchicalAlgo`. Key methods: `train_step()`, `eval_step()`, `get_action()`. `RolloutPolicy` wraps a trained policy for inference-time use (handles observation normalization, batching).

Implemented algorithms: BC (+ Gaussian/GMM/VAE/RNN variants), BCQ, CQL, IQL, IRIS, TD3-BC, HBC, GL, DiffusionPolicy (UNet).

### Models (`robomimic/models/`)
- `base_nets.py` — MLP, Conv backbones, ResNet, ViT, SpatialSoftmax pooling
- `obs_core.py` — per-modality encoders (VisualCore, ScanCore, PointCloudCore, etc.)
- `obs_nets.py` — combines modality encoders into a unified observation network
- `policy_nets.py` — policy heads (Gaussian, GMM, VAE, RNN-based)
- `value_nets.py` — Q/V networks
- `vae_nets.py`, `diffusion_policy_nets.py`, `transformers.py` — specialized architectures

### Observation System (`robomimic/utils/obs_utils.py`)
Encoders are registered globally via `ObsUtils`. Before any model is built, `ObsUtils.initialize_obs_utils_with_obs_specs()` must be called to set up modality groups and encoder mappings. Observation keys are grouped into modalities (e.g., `low_dim`, `rgb`) which determine how they are encoded.

### Data Pipeline (`robomimic/utils/dataset.py`)
`SequenceDataset` reads HDF5 files and returns fixed-length sequences with padding. Supports multiple cache modes (`all`, `low_dim`, `none`). `MetaDataset` wraps multiple `SequenceDataset` instances for multi-dataset training (v0.5.0+).

### Environment Wrappers (`robomimic/envs/`)
`EnvBase` defines a standard interface (`step`, `reset`, `reset_to`, `render`, `is_success`). Primary wrappers: `EnvRobosuite` (robosuite simulator), `EnvGym` (OpenAI Gym), `EnvIGMOMart` (MOMART). `EnvWrapper` provides composable middleware. Environments are created from HDF5 metadata via `EnvUtils.create_env_from_metadata()`.

### Training Loop (`robomimic/utils/train_utils.py`)
`run_epoch()` drives each training epoch. Validation rollouts are run periodically and videos saved to the experiment directory. Experiment directories are timestamped under `{experiment.logging.log_dir}/{experiment.name}/`.
