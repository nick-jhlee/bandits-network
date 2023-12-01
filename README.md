
This repository provides the codes for reproducing the experiments described in the paper [*Flooding with Absorption: An Efficient Protocol for Heterogeneous Bandits over Complex Networks*](https://arxiv.org/abs/2303.05445) (OPODIS 2023, ***best student paper***) by [Junghyun Lee](https://nick-jhlee.netlify.app/), Laura Schmid, and [Se-Young Yun](https://fbsqkd.github.io/).

If you plan to use this repository or refer to our work, please use the following bibtex format:

```latex
@InProceedings{lee2023bandits,
	author =	{Junghyun Lee and Laura Schmid and Se-Young Yun},
	title =	{{Flooding with Absorption: An Efficient Protocol for Heterogeneous Bandits over Complex Networks}},
	booktitle =	{27th International Conference on Principles of Distributed Systems (OPODIS 2023)},
	pages =	{},
	series =	{Leibniz International Proceedings in Informatics (LIPIcs)},
	ISBN =	{},
	ISSN =	{},
	year =	{2023},
	volume =	{},
	editor =	{},
	publisher =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
	address =	{Dagstuhl, Germany},
	URL =		{https://arxiv.org/abs/2303.05445},
	URN =		{},
	doi =		{},
	annote =	{}
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