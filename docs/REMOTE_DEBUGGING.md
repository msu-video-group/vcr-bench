# vcr_bench Remote Execution

Remote commands are exposed through `vcr-bench-remote` or `python -m vcr_bench.remote`.

## Config

- Shared defaults: `configs/defaults.toml`
- Local overrides: `configs/local.toml`
- Curated suites: `configs/benchmarks.toml`

Each remote is defined under `[remote.remotes.<name>]`.

## Typical flow

1. Push local code:
   `vcr-bench-remote push-code --commit-message "update remote runner"`
2. Sync the remote checkout to the configured git ref:
   `vcr-bench-remote sync-remote --remote main`
3. Launch one focused job:
   `vcr-bench-remote launch-job --remote main --mode attack --model x3d --attack ifgsm --dataset kinetics400 --dataset-subset kinetics400_mini_val --attack-name debug_ifgsm --num-videos 8`
4. Launch the curated clean-accuracy suite:
   `vcr-bench-remote launch-accuracy-suite --remote main --suite default`
5. Launch the curated attack suite:
   `vcr-bench-remote launch-attack-suite --remote main --suite default`
6. Check job state:
   `vcr-bench-remote job-status --remote main --job-id 123456`
7. Fetch logs and results:
   `vcr-bench-remote fetch-job-artifacts --remote main --job-id 123456 --attack-name default_suite`

Use `--dry-run` to inspect the generated git, ssh, sbatch, and rsync commands without contacting the remote cluster.

## Notes

- Slurm container image and mount configuration are injected from TOML config into the shell scripts.
- Attack and accuracy suite definitions are committed in `configs/benchmarks.toml`.
- Remote artifacts are fetched into the configured `remote_results_dir` under a per-remote, per-job folder.
