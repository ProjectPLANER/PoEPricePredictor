from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class PoEDataImporter:
    def __init__(self):
        self.currency_file_path_list = self.get_currency_file_paths()
        self.df_files = self.get_file_df(self.currency_file_path_list)
        self.latest_currency = self.get_latest_currency_list(self.df_files)
        self.league_names = list(self.df_files["League"])
        self.df = self.get_data(self.df_files, self.latest_currency)

    def get_currency_file_paths(self, dir_path="*/*", extension=".csv"):
        """ Get all currency files. Downloaded and unzipped data dumps from poe.ninja.
            :param string: dir_path:
                The directory to search for files in. Defaults to current dir.
            :param string: extension:
                The file extension to search for, defaults to ".csv".

            :returns list:
                The directory and file name of each file containing the word "currency". 
        """
        csv_files = [
            csv_file for csv_file in glob(dir_path + extension, recursive=True)
        ]
        return [
            currency_file for currency_file in csv_files if "currency" in currency_file
        ]

    def get_file_df(self, file_list):
        """ Builds a df from each league, date and file from the specified file list. 

            :param list: file_list
                The list of file paths.

            :returns pd.DataFrame:
                A dataframe containing the league name, date and file path.
        """
        file_dict = {
            file.split(".")[0]: {"Date": file.split(".")[1], "File": file}
            for file in file_list
        }
        df = pd.DataFrame(file_dict).T
        df["Date"] = pd.to_datetime(df["Date"])
        df["File"] = df["File"].astype("string")
        df = df.reset_index()
        df.rename(columns={"index": "League"}, inplace=True)
        df = df.sort_values(by=["Date"], ascending=False)
        return df

    def extract_df(self, file_path):
        """ Extracts a dataframe from a csv file path.

            :param string: file_path
                The path of a csv file.

            :returns pd.DataFrame:
                The processed and extracted dataframe for the given poe csv data. 
        """
        df = pd.read_csv(file_path, sep=";")
        df.rename(columns={"Get": "Currency"}, inplace=True)
        df = df[df["Pay"] == "Chaos Orb"]
        df = df[["League", "Date", "Currency", "Value"]]
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = df["Date"] - df.loc[0]["Date"]
        return df

    def group_df(self, df):
        """ Groups a dataframe by date. 

            :param pd.DataFrame: df
                The single league dataframe.

            :returns pd.DataFrame:
                A dataframe grouped to be indexed by date, with currency as columns. 
                Multiindexed to have league name.
        """
        return df.groupby(["Date", "Currency", "League",]).sum().unstack(level=0).T

    def get_latest_currency_list(self, df):
        """ Gets the sorted list of the most recent league currency.

            :param pd.DataFrame: df
                The dataframe containing leagues, files and dates.

            :returns list:
                A list of the most recent leagues currency.
        """
        current_list = self.get_currency_list(self.get_latest_league_data(df))
        current_list.sort()
        return current_list

    def get_latest_league_data(self, df):
        """ Gets the most recent league df from all the files.

            :param pd.DataFrame: df
                The dataframe containing leagues, files and dates.

            :returns pd.DataFrame:
                The extracted dataframe for the most recent league.
        """
        max_date = pd.to_datetime(df["Date"]).max()
        df = df[df["Date"] == max_date]
        [latest_league_file_dir] = df["File"].values
        df = self.extract_df(latest_league_file_dir)
        return df

    def get_currency_list(self, df):
        """ Gets a list of all the unique currency names from df.

            :param pd.DataFrame: df
                The single league dataframe.

            :returns list:
                Each currency type from the dataframe.
        """
        return list(df["Currency"].unique())

    def fill_league_currency(self, df, latest_currency_list):
        """ Creates an empty column in the df for any currency missing 
            in the latest_currency_list.

            :param pd.DataFrame: df
                The single league dataframe, grouped.

            :returns pd.DataFrame:
                The dataframe ensuring each currency type is included.
        """
        league_currency_list = [currency[0] for currency in df.columns]
        for lastest_currency in latest_currency_list:
            if lastest_currency not in league_currency_list:
                df[lastest_currency, df.columns[0][1]] = np.nan
        df = df.sort_index(axis=1)
        return df

    def get_data(self, df, latest_currency):
        """ Extracts, groups and joins all the leagues given in the df.

            :param pd.DataFrame: df
                The dataframe containing leagues, files and dates.
            :param list: latest_currency
                The list of the latest currency names.

            :returns pd.DataFrame:
                The dataframe for all the given files, grouped.
        """
        file_paths = list(df["File"])
        df = self.extract_df(file_paths[0])
        df = self.group_df(df)
        df = self.fill_league_currency(df, latest_currency)
        for file_path in file_paths[1:]:
            league = self.extract_df(file_path)
            league_grp = self.group_df(league)
            league_grp = self.fill_league_currency(league_grp, latest_currency)
            df = df.join(league_grp)
        df = df.reset_index(drop=True)
        return df

    def plot_currency(self, df=None, league_names=None, currency_list=None):
        """ A plotly scatter plot of each currency over the days since the league started.
            The currency plotted can be adjusted in the dropdown menu.
            Leagues can be toggled on the right hand legend.

            :param pd.DataFrame: df
                The single league dataframe, grouped.
            :param list: league_names
                A list of all the leagues of interest.
            :param list: currency_list
                A list of all the currencies of interest.

            :returns
                A plotly figure.
        """
        if not df:
            df = self.df
        if not league_names:
            league_names = self.league_names
        if not currency_list:
            currency_list = self.latest_currency

        df = df.fillna(0)

        fig = go.Figure()

        for league_name in league_names:
            fig.add_traces(
                go.Scatter(
                    x=df.index,
                    y=df[df.columns[0][0]][league_name].values,
                    name=league_name,
                )
            )

        updatemenus = [
            {
                "buttons": [
                    {
                        "method": "restyle",
                        "label": currency,
                        "args": [
                            {"y": [df[currency, league] for league in league_names]}
                        ],
                    }
                    for currency in currency_list
                ],
                "direction": "down",
                "showactive": True,
                "xanchor": "center",
                "yanchor": "bottom",
                "x": 0.5,
                "y": 1.1,
            },
        ]

        fig.update_layout(
            updatemenus=updatemenus, xaxis_title="Date", yaxis_title="Chaos Orb",
        )
        fig.show()
