# C-3: Does the PyTorch 2.13.0 PyPI wheel carry `sm_120`?

> Investigation-queue reference: `docs/investigation-queue.md` C-3.
> Touches, but does not edit (out of file-ownership scope for this pass):
> `pyproject.toml`'s `cuda` extras comment block (lines ~11-32) and
> `docs/roadmap.md` RK6 / T0-6 / H1.

## Verdict

**Yes.** A clean, from-scratch venv installing the plain `torch==2.13.0` PyPI wheel
(no index override, no nightly channel) reports:

```
$ python -m venv /tmp/torch-probe
$ /tmp/torch-probe/bin/pip install -q torch==2.13.0
$ /tmp/torch-probe/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
2.13.0+cu130 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

`sm_120` is in the list. This is the exact command the investigation queue specified;
ran in a genuinely clean venv (no reuse of any project venv), no GPU required to get
this answer (`torch.cuda.get_arch_list()` reads the static list of arches libtorch's
prebuilt kernels were compiled for — it doesn't need a GPU present, and none was used
for this check; a GPU was present on this box for other, unrelated work but this
specific check ran with `CUDA_VISIBLE_DEVICES` untouched and no device queried).

Reported build tag is `+cu130` (bundled against CUDA 13.0 runtime libs, installed
transitively as separate `nvidia-*-cu13` wheels — `nvidia_cudnn_cu13`,
`nvidia_cusparselt_cu13`, `nvidia_nccl_cu13`, etc. — this is PyTorch's normal PyPI
packaging split since the CUDA-12 era, not anything unusual).

## What this settles

`pyproject.toml`'s own comment on the `torch` pin (lines ~13-32) explains the
history precisely: the reference dev environment runs a **self-compiled
pre-release**, `2.13.0a0+gitcf30153` (editable install of `/home/bot/pytorch-build`),
built from source "because Blackwell/CC 12.0 support at verification time predated a
stock PyPI wheel that had it." That comment already anticipated this outcome and
says the plain wheel "also satisfies the same check" if installed — but it was
written before anyone had actually confirmed the **final** PyPI `2.13.0` release
(as opposed to some earlier PyPI `2.13.0` release candidate, or `2.12.x`) carries
`sm_120`. This probe closes that gap: as of 2026-08-01, the plain `pip install
torch==2.13.0` path is a real, working substitute for the self-compiled reference
build, at least with respect to Blackwell/SM120 kernel availability.

**Not settled by this probe** (out of scope / needs the actual reference env or a
GPU to check, flagged so it isn't silently assumed away):
- Whether the exact same *numerics* / kernel selection paths that
  `2.13.0a0+gitcf30153` exercises on this box are reproduced bit-for-bit by the
  stock `+cu130` wheel — `torch.cuda.get_arch_list()` says the kernels are compiled
  in, not that they're identical in tuning/algorithm choice to a from-source build
  against this box's exact CUDA 13.3/driver 610.47 stack (the reference env is
  CUDA 13.3; the PyPI wheel bundles CUDA 13.0 runtime libs via the `nvidia-*-cu13`
  wheels — a minor-version gap, and the kind of thing that belongs under RK6's
  "dependency chain drift" risk, not this ticket).
- Whether `nvidia-cutlass-dsl` and the rest of the `cuda` extras group resolve
  cleanly against a plain-wheel torch the same way they do against the
  self-compiled reference build. Only `torch` itself was probed here, per the
  investigation queue's specified command; the other pins were not re-verified
  together as a set.
- Nothing about this probe touches GPU execution — it did not run a kernel, launch
  anything on a device, or otherwise contend for GPU time with other work in
  progress on this box.

## Impact on roadmap items that reference this

- **RK6** ("依赖链漂移... 静默变慢或变错") — this finding *reduces* one specific
  facet of that risk (torch no longer strictly requires a from-source build to get
  SM120 support) but does not close RK6 generally; the CUDA 13.0-vs-13.3 gap above
  is itself a small instance of the same risk category RK6 exists to track.
- **H1 release gate** ("依赖可从公开源安装") — the roadmap's current text ties that
  specifically to sparkinfer upstreaming (RK2, still unresolved, still pinned to
  `origin/master @ 0844a4f`), not to torch. This finding does not change H1's
  blocking status — sparkinfer is still the long pole — but it does mean torch is
  no longer *also* a from-source dependency stacked on top of that gate. Concretely:
  if sparkinfer's RK2 gets resolved, `pip install -e '.[cuda]'` on a clean machine
  is closer to sufficient than it was when this project's `torch` pin was written,
  because the `torch==2.13.0` pin now resolves to a wheel that actually has the
  needed arch, not just a version number that happened to be declared.
- **`pyproject.toml`'s comment** (out of edit scope here, flagging for whoever owns
  that file next): the paragraph starting "If a fresh box installs the plain
  `torch==2.13.0` PyPI wheel instead, that also satisfies the same check" was
  written as a forward-looking claim; it can now be updated to state this was
  verified 2026-08-01, with the arch list above, rather than left as an unverified
  "also satisfies" assertion.

## Repro

```bash
python -m venv /tmp/torch-probe
/tmp/torch-probe/bin/pip install -q torch==2.13.0
/tmp/torch-probe/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# 2.13.0+cu130 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

Environment note: this box's network path to PyPI is slow (~290 KB/s observed during
this probe); the full `torch` + transitive `nvidia-*-cu13` wheel set is several
hundred MB to low GB, so this install took on the order of tens of minutes end to
end. Not a signal about anything architectural — just a heads-up for anyone
re-running this command expecting it to be instant.
