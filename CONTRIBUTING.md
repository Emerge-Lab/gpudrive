# Contributing to GPUDrive

Thank you for investing your time in contributing to GPUDrive! 🚗✨ We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We use [Github Flow](https://guides.github.com/introduction/flow/index.html), so all code changes happen through pull requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Report bugs 🐛 using Github's [issues](https://github.com/Emerge-Lab/gpudrive/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/Emerge-Lab/gpudrive/issues/new); it's that easy!

### Write bug reports with detail, background, and sample code

Here's [an example bug report](http://www.openradar.me/11905408), you can use as model and here is a useful [template](https://github.com/pygame/pygame/blob/main/.github/ISSUE_TEMPLATE/bug_report.md).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can. [This stackoverflow question](http://stackoverflow.com/q/12488905/180626) includes sample code that *anyone* with a base R setup can run to reproduce what I was seeing
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Learning benchmark 📈

Maybe you made some changes and want to make sure learning is working as intended. To do this, follow these steps:

- **Step 1**: Make sure you have a [wandb](https://wandb.ai/) account.
- **Step 2**: Run this out of the box, the only thing you might want to change is the "device" (if you encounter problems, please report the 🐛!):

```Python
python baselines/ippo/ippo_sb3.py
```

This should kick off a run that takes about **15-20 minutes to complete on a single gpu**. We’re using [Independent PPO](https://arxiv.org/abs/2103.01955) (IPPO) to train a number of agents distributed across 3 traffic scenarios. For an example of what a "healthy" run looks like, I ran the script above with these exact settings in  `baselines/ippo/config.py`  on  `08/19/2024` and created a wandb report with ***complete logs*** and ***videos:***

$$


$$

<details>
  <summary>🗂️ Running your test with more scenarios </summary>

---

Sometimes 3 scenarios is not enough to test your code. If you want to run your test with more scenarios:

1. Download the dataset (see README)
2. Update `selection_discipline = SelectionDiscipline.K_UNIQUE_N` in `baselines/ippo/config/ippo_ff_sb3.yaml`

For example, to use 10 different scenarios, we can run:

```bash
python baselines/ippo/ippo_sb3.py --data_dir='<your_data_path>' --render_n_worlds=10 --k_unique_scenes=10 --total_timesteps=15_000_000
```

This will kick off a run on 10 randomly sampled scenes and render all 10 of them.
---------------------------------------------------------------------------------

</details>

---

> ### 🔎 Checkout the wandb report [here](https://api.wandb.ai/links/emerge_/tax15h89)

---

If you have the suspicion that something might be broken, or are just looking for a good sanity check, compare your metrics with the runs in the report above. Do they all look similar? Then everything seems to be working fine. If a metric looks off, maybe give your code another look. Are your agents learning better/faster? That’s interesting - let us know why!

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook&#39;s Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md) and from the [Transcriptase adapted version](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62)
