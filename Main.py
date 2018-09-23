from Clusterings import MS
def edareamaliat_churnprediction_merchants(dataframe):
    result = MS.clusters_statistics(merchant_data_df=dataframe, number_of_clusters=6, write_to_excel_file=True)
