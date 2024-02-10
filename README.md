
This repository provides the codes for reproducing the experiments described in the paper [*Flooding with Absorption: An Efficient Protocol for Heterogeneous Bandits over Complex Networks*](https://arxiv.org/abs/2303.05445) (OPODIS 2023, ***best student paper***) by [Junghyun Lee](https://nick-jhlee.netlify.app/), Laura Schmid, and [Se-Young Yun](https://fbsqkd.github.io/).

If you plan to use this repository or refer to our work, please use the following bibtex format:

```latex
@InProceedings{lee2024absorption,
  author =	{Lee, Junghyun and Schmid, Laura and Yun, Se-Young},
  title =	{{Flooding with Absorption: An Efficient Protocol for Heterogeneous Bandits over Complex Networks}},
  booktitle =	{27th International Conference on Principles of Distributed Systems (OPODIS 2023)},
  pages =	{20:1--20:25},
  series =	{Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN =	{978-3-95977-308-9},
  ISSN =	{1868-8969},
  year =	{2024},
  volume =	{286},
  editor =	{Bessani, Alysson and D\'{e}fago, Xavier and Nakamura, Junya and Wada, Koichi and Yamauchi, Yukiko},
  publisher =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address =	{Dagstuhl, Germany},
  URL =		{https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.OPODIS.2023.20},
  URN =		{urn:nbn:de:0030-drops-195100},
  doi =		{10.4230/LIPIcs.OPODIS.2023.20},
  annote =	{Keywords: multi-armed bandits, multi-agent systems, collaborative learning, network protocol, flooding}
}
```


## Reproducing Figures

### Figure 3, 4, 5
Run
```python
python main.py
```
to obtain all the results. This will create Figure 4.

Then follow through the colab notebook *plotting.ipynb* to obtain Figures 3 and 5.



### Figure 6
Run
```python
python main_deltas.py
```

Then, go to the deltas folder and run
```python
python computing_deltas.py
```
and then follow through the colab notebook *plot_delta_regretgap.ipynb*
