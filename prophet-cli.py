import click
import sys
import pandas as pd
from fbprophet import Prophet


@click.command()
@click.argument('filename')
@click.option('--demo',
              is_flag=True,
              default=False,
              help='Run the Peyton Manning Wikipedia demo forecast.')
@click.option('--output',
              default='forecast.png',
              help='Save generated forecast figure to this file.')
@click.option('--periods',
              default=365,
              help='Number of periods (days) ahead to forecast.')
def main(filename, demo, output, periods):
    """Command line interface for Facebook Prophet forecasting procedure.
       Reads from FILENAME if given, or from stdin if not."""

    if demo:
        df = pd.read_csv('./test/example_wp_log_peyton_manning.csv')
        click.echo(df.head())
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=365)
        click.echo(future.tail())
        forecast = m.predict(future)
        click.echo(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        fig1 = m.plot(forecast)
        fig1.savefig(output)

    df = pd.read_csv(sys.stdin if (
        filename == '' or filename == '-') else filename)
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    m.plot(forecast).savefig(output)


if __name__ == '__main__':
    main()
