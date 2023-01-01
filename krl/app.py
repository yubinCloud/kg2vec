import typer

from apps.transe import app as transe_app



app = typer.Typer()

app.add_typer(transe_app, name='TransE')


if __name__ == '__main__':
    app()
