# Reference results

These files are the saved evidence used to generate the final paper's figures and tables. They are inputs to the display generators, not training inputs.

- `main/`: raw initialization/fitting arrays for all 60 allocation runs, the Figure 2 trajectory, and summaries of those runs.
- `core/`: archived summaries for sampling, dataset-size, scale/rate, input-distribution, anisotropic, and noise experiments. Most original per-run NPZ files are not present in the archive. The training scripts generate new raw files and summaries.
- `initialization/`: individual-run and summary CSV records for the archived 20-run CPU sweep, including the retained-neuron diagnostics.
- `nonlinear/`: archived summaries for the two nonlinear odd teachers at all reported widths.
- `xavier/`: the new proper Xavier-normal control's raw data/checkpoints, individual runs, summary, and original verification report.

The full synthetic datasets and sampled-index sequences are retained for the new Xavier control; Figure 2 retains its fixed dataset. The allocation files retain initial/fitted parameters and monitored diagnostics. No external dataset is needed to regenerate any experiment.

`MANIFEST.sha256` records the exact reference files included in this release. The checksum manifest excludes itself and this descriptive README. These records are not a claim that every archived additional experiment was retrained during release preparation; `../validation/README.md` records the completed release checks.
