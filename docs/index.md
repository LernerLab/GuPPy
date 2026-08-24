# GuPPy

```{image} ../assets/GuppyLogo.png
:alt: GuPPy logo
:width: 300px
:align: center
```

**Guided Photometry Analysis in Python** is an open-source tool for processing and analyzing fiber photometry recordings. It provides a GUI-based pipeline covering raw data ingestion, signal preprocessing, PSTH computation, transient detection, visualization, and export to NWB. Data formats supported include TDT, Doric, Neurophotometrics (NPM), and generic CSV.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Set up a conda environment and install GuPPy from PyPI or from source.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Learning-oriented guides that walk you through a complete workflow from start to finish. Start here if you are new to GuPPy.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` How-to Guides
:link: how-to/index
:link-type: doc

Task-oriented recipes for readers who already know what they want to accomplish.
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Explanation
:link: explanation/index
:link-type: doc

Background and context: the science behind fiber photometry, the design of the GuPPy pipeline, and the reasoning behind key parameter choices.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc

Lookup material: every input parameter the GUI exposes, every file GuPPy writes, and the vocabulary GuPPy uses for its data entities.
:::

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Contributor's Guide
:link: contributing/index
:link-type: doc

Get oriented in the codebase: pipeline architecture, the test suite, and how to add support for a new recording format.
:::

::::

## Getting help

GuPPy was initially developed with TDT recordings in mind. It now also supports Neurophotometrics, Doric, NWB and generic CSV inputs, but these are less extensively tested because of the limited sample data available for them. If you run into problems, get in touch on the [chat room](https://gitter.im/LernerLab/GuPPy?utm_source=share-link&utm_medium=link&utm_campaign=share-link) or by [raising an issue](https://github.com/LernerLab/GuPPy/issues), so that we can continue to improve the tool.

## Citing GuPPy

If you use GuPPy in your research, please cite:

> Venus N. Sherathiya, Michael D. Schaid, Jillian L. Seiler, Gabriela C. Lopez, and Talia N. Lerner. [GuPPy, a Python toolbox for the analysis of fiber photometry data](https://www.nature.com/articles/s41598-021-03626-9). Sci Rep 11, 24212 (2021). <https://doi.org/10.1038/s41598-021-03626-9>

## Links

- [Source code](https://github.com/LernerLab/GuPPy)
- [PyPI package](https://pypi.org/project/guppy-neuro/)
- [Issue tracker](https://github.com/LernerLab/GuPPy/issues)

```{toctree}
:maxdepth: 1
:hidden:

installation
tutorials/index
how-to/index
explanation/index
reference/index
contributing/index
```
