from invoke import task

# Set
ENV_NAME = 'gsh_py_gp'

# Environment

# Run this first
# conda env create -f environment.yml
# conda activate gsh_py_gp

@task
def env_set_jupyter(c):
    print('Setting up jupyter kernel')
    c.run(
        f"ipython kernel install --name {ENV_NAME} --display-name {ENV_NAME} --sys-prefix")
    print('Adding nbextensions')
    c.run("jupyter nbextensions_configurator enable --user")
    print('Enable ipywidgets')
    c.run("jupyter nbextension enable --py widgetsnbextension")
    print('Done!')


@task
def env_to_freeze(c):
    c.run(f"conda env export --name {ENV_NAME} --file environment_to_freeze.yml")
    print('Exported freeze environment to: environment_to_freeze.yml')


@task
def env_update(c):
    c.run(f"conda env update --name {ENV_NAME} --file environment.yml --prune")


@task
def env_remove(c):
    c.run(f"conda remove --name {ENV_NAME} --all")
