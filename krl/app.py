import typer

from apps.rescal import app as rescal_app
from apps.transe import app as transe_app
from apps.transh import app as transh_app
from apps.distmult import app as distmult_app



app = typer.Typer()

app.add_typer(rescal_app, name='RESCAL')
app.add_typer(transe_app, name='TransE')
app.add_typer(transh_app, name='TransH')
app.add_typer(distmult_app, name='DistMult')


if __name__ == '__main__':
    app()
