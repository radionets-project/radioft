(getting_started_users)=


# Getting Started for Users

```{warning}
The following guide is for *users*. If you want to contribute to
radioft as a developer, see [](getting_started_dev).
```


## Install `radioft`

``radioft`` is available on [PyPI](https://pypi.org/project/radioft/)
To install ``radioft`` into an existing virtual environment, use
one of the following installation methods.


::::{admonition} Should I use `pip` or ...?
:class: hint dropdown

With many so many package installers available, installing software can be
confusing. Here is a guide to help you make a sensible choice.

1. **Are you already using an environment manager?**

   Great, then you should use that tool to install `radioft`.

2. **Are you considering using an environment manager?**

   There are lots of environment managers to choose from.
   If you are unsure where to start, consider starting with
   [a Python virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
   [mamba](https://mamba.readthedocs.io/en/latest/) is also a great choice
   and comes in a lightweight bundle with [Miniforge](https://github.com/conda-forge/miniforge).

3. **If environment managers are not your thing...**

   ...you can also use `pip` to install packages directly to your Python path using

   ```console

      $ pip install -U radioft

   ```

:::{admonition} Ignoring environment management
:class: warning
:name: warning:env-management

While environment managers may sound complicated at first, they are strongly recommended.
Ignoring them may lead to confusion if something breaks later on.
:::
::::


::::{grid} 1 2 2 2

:::{grid-item-card} Install with `pip`

In a [virtual environment][venv]:

```shell-session
pip install radioft
```
:::

:::{grid-item-card} Install with [`mamba`][mamba] / `conda`

`radioft` is available through `conda-forge`.
```shell-session
$ mamba install -c conda-forge radioft
```
:::

:::{grid-item-card} Install with [`pipx`][pipx]

Never heard of `pipx`? See [the documentation][pipx] for more.

```shell-session
pipx install radioft
```
:::

:::{grid-item-card} Install with [`uv`][uv]

Never heard of `uv`? See [the documentation][uv] for more.

```shell-session
uv add radioft
```
Or, if you prefer the pip interface:
```shell-session
uv pip install radioft
```
:::

:::{grid-item-card} Install with [`pixi`][pixi]

Never heard of `pixi`? See [the documentation][pixi] for more.

`radioft` is available through `conda-forge` and can be installed using `pixi`.
```shell-session
pixi install radioft
```
:::

:::{grid-item-card} Recommendation

```{note}
We strongly recommend using uv to install
`radioft` for now.
```
:::

::::

[venv]: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
[mamba]: https://mamba.readthedocs.io/en/latest/
[pipx]: https://pipx.pypa.io/stable/
[uv]: https://docs.astral.sh/uv/
[pixi]: https://pixi.sh/
