from invoke import task

from src.data import download, make_sua_surfaces, make_summary
from src.visualization import visualize

# Set
ENV_NAME = "gsh_py_gp"

# Environment

# Run this first
# conda env create -f environment.yml
# conda activate gsh_py_gp


@task
def env_set_jupyter(c):
    print("Setting up jupyter kernel")
    c.run(
        f"ipython kernel install --name {ENV_NAME} --display-name {ENV_NAME}"
        " --sys-prefix"
    )
    print("Adding nbextensions")
    c.run("jupyter nbextensions_configurator enable --user")
    print("Enable ipywidgets")
    c.run("jupyter nbextension enable --py widgetsnbextension")
    print("Done!")


@task
def env_to_freeze(c):
    c.run(
        f"conda env export --name {ENV_NAME} --file environment_to_freeze.yml"
    )
    print("Exported freeze environment to: environment_to_freeze.yml")


@task
def env_update(c):
    c.run(f"conda env update --name {ENV_NAME} --file environment.yml --prune")


@task
def env_remove(c):
    c.run(f"conda remove --name {ENV_NAME} --all")


@task
def data_download(c):
    download.main()


@task(pre=[data_download])
def make_surfaces(
    c,
    anhydrite_perc_min: int = 5,
    anhydrite_perc_max: int = 33,
    anhydrite_perc_step: int = 1,
):
    make_sua_surfaces.main(
        anhydrite_perc_min, anhydrite_perc_max, anhydrite_perc_step
    )


@task(pre=[make_surfaces])
def make_quantiles(c, overwrite: bool = False):
    make_summary.main(overwrite=overwrite)


@task(pre=[make_quantiles])
def make_figures(c):
    visualize.main()
