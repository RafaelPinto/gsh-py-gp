# Re-ranking seismic uncertainty analysis surfaces with Python

In this repository, we show how to use Xarray and SEGYSAK to create proxy seismic uncertainty surfaces and summarize them by coordinate pair.

We use the Groningen field seismic and horizons hosted by [Data underground 2020](https://dataunderground.org/dataset/groningen-open-fork), originally shared by [(NAM 2020)](https://public.yoda.uu.nl/geo/UU01/1QH0MW.html).

All the data exploration and discovery steps are documented with jupyter notebooks in the `/notebooks` directory.

To access the slides used in the GSH's 2023 Spring Symposium [click here](https://docs.google.com/presentation/d/1FcpRLheSzbDu1JakHIG9k5i2W6mAZHrlLe1dD3TaYDc/edit?usp=sharing).

## Development environment

To get all the dependencies into a single conda environment named `gsh_py_gp`, please run:

```shell
conda env create -f environment.yml
```

and then activate the environment with

```shell
conda activate gsh_py_gp
```

## Running the project tasks

We use the Python library `invoke` to keep track of the main commands needed to process the Groningen field data and produce the figures used in the conference slides. Below are the main tasks that can be run from a terminal having the `gsh_py_gp` conda environment activated:

1. Data download: To programatically download the Groningen field data from Data Underground's AWS buckets.

```shell
invoke data-download
```


2. Make surfaces: To create the target horizon SUA proxy surfaces.

```shell
invoke make-surfaces
```

3. Make quantiles: To summarize the SUA proxy surfaces by grouping by coordinate pairs and estimating the quantiles values (0.1, 0.25, 0.50, 0.75, 0.90).

```shell
invoke make-quantiles
```

4. Make figures: To create all the figures used in the conference slides.

```shell
invoke make-figures
```


## References

- DALL·E 2023-04-09 10.20.06 - A happy person in The Scream painting from Edvard Munch. Please use pastel colors and a peaceful atmosphere.

- Data Underground (2020). Groningen open fork, https://dataunderground.org/dataset/groningen-open-fork, accessed 15 January 2023.

- Edvard Munch, 1893, The Scream, oil, tempera and pastel on cardboard, 91 x 73 cm, National Gallery of Norway. National Gallery of Norway 8 January 2019 (upload date) by Coldcreation, Public Domain, https://commons.wikimedia.org/w/index.php?curid=69541493

- Hall, M. (2020). A big new almost-open dataset: Groningen, https://agilescientific.com/blog/2020/12/7/big-new-almost-open-data, accessed 15 January 2023.


- H.W. van Gent, S. Back, J.L. Urai, P.A. Kukla, K. Reicherter, (2009), Paleostresses of the Groningen area, The Netherlands—Results of a seismic based structural reconstruction. Tectonophysics, 470 (2009), pp. 147-161. https://doi.org/10.1016/j.tecto.2008.09.038

- Jan de Jager & Clemens Visser (2017). Geology of the Groningen field – an overview. Netherlands Journal of Geosciences — Geologie en Mijnbouw 96 (5), p 3–15, 2017 [DOI 10.1017/njg.2017.22.](https://www.cambridge.org/core/journals/netherlands-journal-of-geosciences/article/geology-of-the-groningen-field-an-overview/9947C006B646623624ADF30D3C6C8CC5)

- M Kortekaas & B Jaarsma (2017). Improved definition of faults in the Groningen field using seismic attributes. Netherlands Journal of Geosciences — Geologie en Mijnbouw 96 (5), p 71–85, 2017 [DOI 10.1017/njg.2017.24.](https://www.cambridge.org/core/journals/netherlands-journal-of-geosciences/article/improved-definition-of-faults-in-the-groningen-field-using-seismic-attributes/554FE576A50E25A8219D261D6BF270A1#article)

- NAM (2016). Production, Subsidence, Induced Earthquakes and Seismic Hazard and Risk Assessment in the Groningen Field. [Technical Addendum to the Winningsplan 2016](https://nam-feitenencijfers.data-app.nl/download/rapport/9fd11c35-6260-482f-a6d2-8b1ff78e8af8?open=true).

- NAM (2020). Petrel geological model of the Groningen gas field, the Netherlands. Open access through EPOS-NL. Yoda data publication platform Utrecht University. [DOI 10.24416/UU01-1QH0MW.](https://public.yoda.uu.nl/geo/UU01/1QH0MW.html)

- Osypov, K., Yang, Y., Fournier, A., Ivanova, N., Bachrach, R., Yarman, C.E., You, Y., Nichols, D. and Woodward, M. (2013), Model-uncertainty quantification in seismic tomography: method and applications. Geophysical Prospecting, 61: 1114-1134. https://doi.org/10.1111/1365-2478.12058