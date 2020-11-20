import sys
import click
import pandas as pd
from fbprophet import Prophet


@click.command()
@click.option('--infile',
              default='-',
              help='File to read input from. Stdin if not specified.')
@click.option('--outfile',
              default='forecast.png',
              help='Save forecast figure to this file.')
@click.option('--periods',
              default=365,
              help='Number of periods (days) ahead to forecast.')
@click.option('--demo',
              is_flag=True,
              default=False,
              help='Run the Peyton Manning Wikipedia demo forecast.')
def main(infile, outfile, periods, demo):
    """Command line interface for Facebook Prophet forecasting procedure."""
    if demo:
        payton(outfile)
    else:
        df = pd.read_csv(sys.stdin if (
            infile == '' or infile == '-') else infile)
        df.rename(columns={
            df.columns[0]: 'ds',
            df.columns[1]: 'y'
        },
                  inplace=True)
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        m.plot(forecast).savefig(outfile)


def payton(outfile):
    df = pd.read_csv('./test/example_wp_log_peyton_manning.csv')
    click.echo(df.head())
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    click.echo(future.tail())
    forecast = m.predict(future)
    click.echo(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    m.plot(forecast).savefig(outfile)


if __name__ == '__main__':
    main()
