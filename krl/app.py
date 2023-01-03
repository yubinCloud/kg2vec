import typer

from apps.rescal import app as rescal_app
from apps.transe import app as transe_app
from apps.transh import app as transh_app



app = typer.Typer()

app.add_typer(rescal_app, name='RESCAL')
app.add_typer(transe_app, name='TransE')
app.add_typer(transh_app, name='TransH')


if __name__ == '__main__':
    app()
