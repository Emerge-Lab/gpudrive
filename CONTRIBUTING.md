# Contributing to GPUDrive

Thank you for investing your time in contributing to GPUDrive! 🚗✨ We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests

Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](https://github.com/Emerge-Lab/gpudrive/blob/main/LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs 🐛 using Github's [issues](https://github.com/Emerge-Lab/gpudrive/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/Emerge-Lab/gpudrive/issues/new); it's that easy!

### Write bug reports with detail, background, and sample code

[This is an example](http://stackoverflow.com/q/12488905/180626) of a bug report I wrote, and I think it's not a bad model. Here's [another example from Craig Hockenberry](http://www.openradar.me/11905408), an app developer whom I greatly respect.

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can. [My stackoverflow question](http://stackoverflow.com/q/12488905/180626) includes sample code that *anyone* with a base R setup can run to reproduce what I was seeing
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Learning benchmark 📈

Maybe you made some changes and want to make sure learning is works as before before you submit your PR. To do this, follow these steps:

- **Step 1**: Make sure you have a [wandb](https://wandb.ai/) account.
- **Step 2**: Run this out of the box, the only thing you might want to change is the "device" (if you encounter problems, please report the 🐛!):

```Python
python baselines/ippo/run_sb3_ppo.py
```

This should kick off a run that takes about **15 minutes to complete on a single gpu**. We’re using [Independent PPO](https://arxiv.org/abs/2103.01955) (IPPO) to train a number of agents distributed across 3 traffic scenarios. To show you what a healthy run looks like, I ran the script above with these exact settings on `08/16/2024` and created a wandb report with complete logs and videos of the agents:

---

> **🔎 Checkout the wandb report here**

---

If you have the suspicion that something might be broken, or are just looking for a good sanity check, compare your metrics with the one in the report above. Do they all look similar? Then everything seems to be working fine. If a metric seems off, maybe give your code another look. Are your agents learning faster? That’s really interesting - let us know why!

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook&#39;s Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)