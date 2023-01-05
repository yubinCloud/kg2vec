import typer

from typer_apps.rescal import app as rescal_app
from typer_apps.transe import app as transe_app
from typer_apps.transh import app as transh_app
from typer_apps.distmult import app as distmult_app
from typer_apps.transr import app as transr_app



app = typer.Typer()

app.add_typer(rescal_app, name='RESCAL')
app.add_typer(transe_app, name='TransE')
app.add_typer(transh_app, name='TransH')
app.add_typer(distmult_app, name='DistMult')
app.add_typer(transr_app, name='TransR')


if __name__ == '__main__':
    app()
