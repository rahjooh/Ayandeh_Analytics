def clusters_statistics(self, merchant_data_df, number_of_clusters, i, write_to_excel_file=False):
    """

    :param merchant_data_df: dadeye bad az faze preparartion; ke in dade shamele 4 setune Merchant#,R,F,Mmibashad
    :type merchant_data_df: Dataframe
    :param i: shomare mah
    :type i: integer
    :param write_to_excel_file: aya tamayol be zakhire dar file excel mibashad ya kheir
    :type write_to_excel_file: boolean
    :return: None
    :rtype: -
    """



    kmeans_result, sum_amount_real_traces = kmeans(num_clusters=number_of_clusters, visualize=False,
                                                            ind_month=i)  # Kmeans_result is all_merchants_df like M# (index), M Segment#, F Segment#, R segment#

    statistics_result = []
    statistics_result_sum_amount_base = []

    mean_sum_amounts = []

    max_sum_amounts = []

    for i in range(number_of_clusters):
        cluster_part = kmeans_result[kmeans_result["labels"] == i]
        cluster_part = cluster_part[["sum_amounts", "no_transactions", "harmonic"]]  # Segment Number
        cluster_part_df = pd.DataFrame(data=cluster_part,
                                       columns=["sum_amounts", "no_transactions", "harmonic"])  # Segment Number

        statistics_result.append(cluster_part.describe(include='all'))

        # mean_sum_amounts.append(sum_amount_real_traces[i].ix[ cluster_part.index ][ "sum_amounts" ].mean())

        cluster_part_sum_amount_base = sum_amount_real_traces[i]

        mean_sum_amounts.append(cluster_part_sum_amount_base[["sum_amounts"]].values.mean())

        # max_sum_amounts.append(sum_amount_real_traces[i].ix[ cluster_part.index ][ "sum_amounts" ].max())


        max_sum_amounts.append(cluster_part_sum_amount_base[["sum_amounts"]].values.max())

        # cluster_part_sum_amount_base = sum_amount_real_traces[ i ].ix[ cluster_part.index ][ "sum_amounts" ]


        cluster_part_sum_amount_base_df = pd.DataFrame(data=cluster_part_sum_amount_base[["sum_amounts"]].values,
                                                       # sum_amounts as a real value
                                                       index=cluster_part_sum_amount_base.index,
                                                       columns=["sum_amounts_base"])

        merged_df = cluster_part_df.join(cluster_part_sum_amount_base_df)

        statistics_result_sum_amount_base.append(merged_df.describe(include='all'))

    # statistics_result_df = pd.concat(statistics_result, keys=list(range(number_of_clusters)))

    statistics_result_df = pd.concat(statistics_result_sum_amount_base, keys=list(range(number_of_clusters)))

    ####################################
    # if write_to_excel_file:
    #     out_file = os.path.join(DATASET_DIR, "Merchants_18_statistic_results_960830_V2.xlsx")
    #     statistics_result_df.to_excel(out_file)


    return statistics_result_sum_amount_base


def kmeans(self, num_clusters=6, visualize=False, ind_month=19):
    """

    :param num_clusters: Tedade Cluster ha
    :type num_clusters:  integer
    :param visualize: Aya tamayol be mosavarsazi hast ya kheir
    :type visualize: boolean
    :param ind_month: Shomare mah baraye ejraye algorithme clustering
    :type ind_month: integer
    :return: 
    :rtype: 
    """

    print(self._merchant_data_df.shape)

    # saving Real Values of R,F, M for each merchant
    out_file = os.path.join(DATASET_DIR, "Merchants_Real Features_" + str(ind_month) + "_970425.xlsx")
    merchant_real_data = pd.DataFrame(data=self._merchant_data_df.values, index=self._merchant_data_df.index,
                                      columns=["no_transactions_R", "harmonic_R", "sum_amounts_R"])
    merchant_real_data["merchant_number"] = self._merchant_data_df.index
    merchant_real_data.to_excel(out_file)
    # End of saving Real Values of R,F, M for each merchant

    # print(all_merchant_labels_df)

    # Kole baze zamani
    if (ind_month == 19):
        all_merchant_labels_df = self.get_weighted_merchant_df(sum_amount_weight=6,
                                                               no_transactions_weight=4,
                                                               harmonic_weight=1)

    # mahane
    else:
        all_merchant_labels_df = self.get_weighted_merchant_df(sum_amount_weight=6,
                                                               no_transactions_weight=4,
                                                               # Be khatere meghyase zamanie mahane
                                                               harmonic_weight=0)

    # For saving Segment# Values of R,F, M for each merchant
    out_file = os.path.join(DATASET_DIR, "Merchants_Tagged Features_" + str(ind_month) + "_970425.xlsx")

    # It seems redundant, it is in Merchants_Tagged_15_960908.xlsx
    # all_merchant_labels_df.to_excel(out_file)


    X = all_merchant_labels_df.as_matrix().astype(np.float)

    # Emale Kmeans ruye khoruji haye Weighted RFM
    y_pred = KMeans(n_clusters=num_clusters).fit_predict(X)
    cluster_number_series = pd.Series(data=y_pred, index=all_merchant_labels_df.index)

    all_merchant_labels_df["labels"] = cluster_number_series

    all_merchant_labels_df["merchant_number"] = cluster_number_series.index

    cluster_numbering = ClusterNumbering()

    # ghabalan shomare cluster haye har merchant bi mani bud vali alan shomare merchant bayangare arzeshe an merchant hast ke 4*M +2*F+ R behtari darad (yani 4>3>2>1>0)
    all_merchant_labels_df = cluster_numbering.renumber(all_merchant_labels_df)
    all_merchant_labels_df["merchant_number"] = all_merchant_labels_df.index

    # Saving tagged (label khorde) merchants to an excel file Shamele Merchant#, shomare segment M (non real), shomare segment F (non real), shomare segment R (non real)
    out_file_tagged = os.path.join(DATASET_DIR, "Merchants_Tagged_" + str(ind_month) + "_970425.xlsx")
    all_merchant_labels_df.to_excel(out_file_tagged)
    # End of Saving tagged (label khorde) merchants to an excel file Shamele Merchant#, shomare segment M (non real), shomare segment F (non real), shomare segment R (non real)


    ########################
    # To Do for making One merged file instead of two, Real, Segment

    # Saving tagged all merchants (label khorde va real) to an excel file Shamele Merchant#, shomare segment M (non real), shomare segment F (non real), shomare segment R (non real) va maghadire reale F, M va R
    merchant_label_complete = pd.merge(merchant_real_data, all_merchant_labels_df, on='merchant_number')
    out_file_tagged = os.path.join(DATASET_DIR, "Merchants_Tagged_all_" + str(ind_month) + "_970425.xlsx")
    merchant_label_complete.to_excel(out_file_tagged)
    # End of Saving tagged all merchants (label khorde va real) to an excel file Shamele Merchant#, shomare segment M (non real), shomare segment F (non real), shomare segment R (non real) va maghadire reale F, M va R


    kmeans_result_traces = []
    sum_amount_real_traces = []

    # Sakhte file haye Excele Merchants_Statistic_Results
    for cluster_num in range(num_clusters):  # 7 ta fixe tedad cluster haye khorujie kmeans ba 3/2 feature

        merchants_of_cluster_number = all_merchant_labels_df["labels"] == (cluster_num)

        # maghadire dade haye clustere jadid ra be edameye ghabli ha ezafe mikonad
        kmeans_result_traces.append(
            all_merchant_labels_df[merchants_of_cluster_number]
        )

        sum_amount_real_traces_data = pd.DataFrame(
            data=self._merchant_data_df.loc[(merchants_of_cluster_number[merchants_of_cluster_number != False].index)],
            index=(merchants_of_cluster_number[merchants_of_cluster_number != False].index))

        # Returns R,F,M
        # Baraye mohasebeye sar jam az maghadire reale M estefade mishavad
        sum_amount_real_traces.append(
            sum_amount_real_traces_data
        )

    #####
    # For checking the results

    statistics_result = []
    statistics_result_sum_amount_base = []

    mean_sum_amounts = []

    max_sum_amounts = []

    for i in range(num_clusters):
        # cluster_part = kmeans_result[ kmeans_result[ "labels" ] == i ]


        cluster_part = all_merchant_labels_df[all_merchant_labels_df["labels"] == i]

        # Shomare Segment haye F, M, R
        cluster_part = cluster_part[["sum_amounts", "no_transactions", "harmonic"]]  # Segment Number
        cluster_part_df = pd.DataFrame(data=cluster_part,
                                       columns=["sum_amounts", "no_transactions", "harmonic"])  # Segment Number

        # maghadire statistics clustere jadid (mohasebe shode tavasote methode describe) ra be edameye maghadire statistics clustere haye ghabli ezafe mikonad
        statistics_result.append(cluster_part.describe(include='all'))

        # All 3 features R, F, M
        cluster_part_sum_amount_base = sum_amount_real_traces[i]

        # Miangine mabaleghe rialie cluster morede barrasi (i)
        mean_sum_amounts.append(cluster_part_sum_amount_base[["sum_amounts"]].values.mean())

        # Maximum mabaleghe rialie cluster morede barrasi (i)
        max_sum_amounts.append(cluster_part_sum_amount_base[["sum_amounts"]].values.max())

        # M riali (real ya base)
        cluster_part_sum_amount_base_df = pd.DataFrame(
            data=cluster_part_sum_amount_base[["sum_amounts"]].values,  # sum_amounts as a real value
            index=cluster_part_sum_amount_base.index,
            columns=["sum_amounts_base"])

        # F real
        cluster_part_num_trans_base_df = pd.DataFrame(
            data=cluster_part_sum_amount_base[["all_transactions"]].values,  # sum_amounts as a real value
            index=cluster_part_sum_amount_base.index,
            columns=["no_transactions_base"])

        # cluster_part_df(Shomare Segment F,M, R) ra dar kenare cluster_part_sum_amount_base_df (maghadire rialie M) migozarad
        merged_df = cluster_part_df.join(cluster_part_sum_amount_base_df)

        # merged_df mohasebe shode dar bala(Shomare Segment F, Shomare SegmentM, Shomare Segment R, M_real, F_Real) ra dar kenare cluster_part_num_trans_base_df (maghadire rialie F) migozarad
        merged_df = merged_df.join(cluster_part_num_trans_base_df)

        # maghadire statistics merged_df mohasebe shode dar bala(Shomare Segment F, Shomare SegmentM, Shomare Segment R, M_real, F_Real) tavasote methode describe mohasebe mishavad va be edameye maghadire statistics clustere haye ghabli ezafe migardad
        statistics_result_sum_amount_base.append(merged_df.describe(include='all'))

    statistics_result_df = pd.concat(statistics_result_sum_amount_base, keys=list(range(num_clusters)))

    ####################################

    # if write_to_excel_file:
    out_file = os.path.join(DATASET_DIR, "Merchants_statistic_results_" + str(ind_month) + "_970425.xlsx")
    statistics_result_df.to_excel(out_file)

    # End of Sakhte file haye Excele Merchants_Statistic_Results


    for i in range(num_clusters):
        print(mean_sum_amounts[i])

    ####

    # baraye khoruji haye ghabli ke az tarighe portale bank va application server ghabele moshahede bud
    if visualize:
        return plotlyvisualize.scatter3d(kmeans_result_traces, columns=["sum_amounts", "no_transactions", "harmonic"],
                                         title="Kmeans for Real Scale4",
                                         out_path=PLOT_OUT_DIR)

    return all_merchant_labels_df, sum_amount_real_traces


