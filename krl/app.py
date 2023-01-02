import typer

from apps.transe import app as transe_app
from apps.transh import app as transh_app



app = typer.Typer()

app.add_typer(transe_app, name='TransE')
app.add_typer(transh_app, name='TransH')


if __name__ == '__main__':
    app()
