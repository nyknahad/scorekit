# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import os
import gc
import copy
from .._utils import color_background, fig_to_excel, adjust_cell_width
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from textwrap import wrap
from concurrent import futures
from functools import partial

warnings.simplefilter('ignore')
plt.rc('font', family='Verdana', size=12)
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)
gc.enable()


class DataSamples:
    """
    Основной класс для хранения данных
    """
    def __init__(self, samples=None, target=None, features=None, cat_columns=None, min_nunique=20, time_column=None,
                 id_column=None, feature_descriptions=None, train_name=None, special_bins=None, result_folder='',
                 n_jobs=1, random_state=0):
        """
        :param samples: выборка для разработки. Задается в виде словаря {название_сэмпла: датафрейм}, может содержать любое кол-во сэмплов
        :param target: целевая переменная
        :param features: список переменных. При None берутся все поля числового типа и нечисловые (кроме target, time_column, id_column) с кол-вом уникльных значений меньше min_nunique
        :param cat_columns: список категориальных переменных. При None категориальными считаются все переменные с кол-вом уникальных значений меньше min_nunique
        :param min_nunique: кол-во уникальных значений, до которого переменная считается категориальной при автоматическом определении
        :param time_column: дата среза
        :param id_column: уникальный в рамках среза айди наблюдения
        :param feature_descriptions: датафрейм с описанием переменных. Должен содержать индекс с названием переменных и любое кол-во полей с описанием, которые будут подтягиваться в отчеты
        :param train_name: название сэмпла обучающей выборки. При None берется первый сэмпл
        :param special_bins: словарь вида {название бина: значение}, каждое из значений которого помещается в отдельный бин
        :param result_folder: папка, в которую будут сохраняться все результаты работы с этим ДатаСэмплом
        :param n_jobs: кол-во используемых рабочих процессов, при -1 берется число, равное CPU_LIMIT
        :param random_state: сид для генератора случайных чисел, используется во всех остальных методах, где необходимо
        """
        if isinstance(result_folder, str):
            if result_folder and not os.path.exists(result_folder):
                os.makedirs(result_folder)
            self.result_folder = result_folder + ('/' if result_folder and not result_folder.endswith('/') else '')
        else:
            self.result_folder = ''

        self.samples = {name: sample for name, sample in samples.items() if not sample.empty} if samples is not None else None
        self.train_name = list(self.samples.keys())[0] if train_name is None and self.samples is not None else train_name
        self.target = target
        if samples is not None and self.target is not None:
            extra_values = [x for x in self.samples[self.train_name][self.target].unique() if x not in [0, 1]]
            if extra_values:
                print(f'Attention! Target contains extra values {extra_values}')
        self.id_column = id_column
        self.feature_descriptions = feature_descriptions
        if feature_descriptions is not None:
            self.feature_descriptions = self.feature_descriptions[~self.feature_descriptions.index.duplicated(keep='first')]
        self.feature_titles = {index: index + '\n\n' +
                                      '\n'.join(['\n'.join(wrap(l, 75)) for l in row.astype('str').values.flatten().tolist() if not pd.isnull(l) and l != 'nan'])
                               for index, row in self.feature_descriptions.iterrows()} if self.feature_descriptions is not None else {}
        self.time_column = time_column if time_column and self.samples and time_column in self.samples[self.train_name] else None

        if features is not None:
            self.features = features
        elif self.samples is not None:
            self.features = [f for f in self.samples[self.train_name].columns 
                             if f not in [self.target, self.id_column, self.time_column]]
        else:
            self.features = []

        if cat_columns is None:
            if self.samples is not None:
                self.cat_columns = [f for f in self.features if self.samples[self.train_name][f].nunique() < min_nunique or
                                    (features is not None and not pd.api.types.is_numeric_dtype(self.samples[self.train_name][f]))]
            else:
                self.cat_columns = []
        else:
            self.cat_columns = cat_columns

        self.special_bins = special_bins if special_bins is not None else {}

        if self.samples is not None:
            if features is None:
                self.features = [f for f in self.features if pd.api.types.is_numeric_dtype(self.samples[self.train_name][f]) or f in self.cat_columns]
                print(f'Selected features: {self.features}')

            if cat_columns is None:
                print(f'Selected categorical features: {self.cat_columns}')

            f_inf = [f for f in self.features if pd.api.types.is_numeric_dtype(self.samples[self.train_name][f]) and np.isinf(self.samples[self.train_name][f]).any()]
            if f_inf:
                special_value = int(''.rjust(len(str(round(self.samples[self.train_name][f_inf].replace(np.inf, 10).max().max()))) + 1, '9'))
                while special_value in self.special_bins.values():
                    special_value = special_value * 10 + 9
                self.special_bins['Infinity'] = special_value
                for name in self.samples:
                    self.samples[name][f_inf] = self.samples[name][f_inf].replace([np.inf, -np.inf], special_value)
                print(f"Warning! Features {f_inf} contain infinities. They are replaced by {special_value} and special bin {{'Infinity': {special_value}}} is added.")

        try:
            n_jobs_max = int(float(os.environ.get('CPU_LIMIT')))
        except:
            n_jobs_max = 4
        if n_jobs > n_jobs_max:
            print(f'Warning! N_jobs exceeds CPU_LIMIT. Set to {n_jobs_max}')
            self.n_jobs = n_jobs_max
        elif n_jobs == -1:
            self.n_jobs = n_jobs_max
        else:
            self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_queue = self.n_jobs*2
        self.n_jobs_restr = 1
        self.bootstrap_base = None
        self.bootstrap = []
        self.ginis = {}
        self.ginis_in_time = {}
        self.gini_df = None

    def to_df(self, sample_field='sample'):
        """
        Конвертирование ДатаСэмпла в датафрейм
        :param sample_field: добавляемое поле, в которое будет записано название сэмплов

        :return: датафрейм
        """
        dfs = []
        for name, sample in self.samples.items():
            df = sample.copy()
            if sample_field:
                df[sample_field] = name
            dfs.append(df)
        return pd.concat(dfs)

    def stats(self, out=None, gini_in_time=True, targettrend=None):
        """
        Вычисление статистики по сэмплам
        :param out: название эксель файла для сохранения статистики
        :param gini_in_time: флаг для расчета динамики джини по срезам. На больших выборках с бутстрэпом может занимать много времени
        :param targettrend: название папки и листа в файле для сохранения графиков TargetTrend. При None TargetTrend не считается
        """
        stats_rows = []
        for name, sample in self.samples.items():
            if self.time_column is not None:
                period = f'{sample[self.time_column].min()} - {sample[self.time_column].max()}'
            else:
                period = 'NA'
            stats_rows.append([name, sample.shape[0], sample[self.target].sum(), sample[self.target].sum() / sample.shape[0],
                               period, len(self.features), len(self.cat_columns)])
        stats = pd.DataFrame(stats_rows, columns=['', 'amount', 'target', 'target_rate', 'period', 'features', 'categorical features']).set_index('').T
        if out:
            with pd.ExcelWriter(self.result_folder + out, engine='xlsxwriter') as writer:
                stats.to_excel(writer, sheet_name='Sample stats')
                ws = writer.sheets['Sample stats']
                format1 = writer.book.add_format({'num_format': '0.00%'})
                ws.set_row(3, None, format1)
                ws.set_column(0, len(self.samples) + 1, 30)

                for name, sample in self.samples.items():
                    tmp = sample[self.features].describe(percentiles=[0.05, 0.5, 0.95], include='all').T
                    try:
                        f_stats = f_stats.merge(tmp, left_index=True, right_index=True, how='left')
                    except:
                        f_stats = tmp.copy()
                f_stats.columns = pd.MultiIndex.from_product([list(self.samples.keys()), tmp.columns])
                f_stats.to_excel(writer, sheet_name='Feature stats')
                gini_df = self.calc_gini(fillna=-999999999, add_description=True)
                self.corr_mat().to_excel(writer, sheet_name='Correlation')
                adjust_cell_width(writer.sheets['Correlation'], gini_df)
                gini_df.to_excel(writer, sheet_name='Gini stats')
                adjust_cell_width(writer.sheets['Gini stats'], gini_df)
                if gini_in_time and self.time_column is not None:
                    self.calc_gini_in_time(fillna=-999999999).to_excel(writer, sheet_name='Gini stats', startrow=len(self.features) + 3)
                if targettrend and isinstance(targettrend, str):
                    ws = writer.book.add_worksheet(targettrend)
                    for i, fig in enumerate(self.targettrend(quantiles=10)):
                        fig_to_excel(fig, ws, row=i * 20, col=0, scale=0.85)
            print(f'Statistics saved in the file {self.result_folder + out}.')
        print(stats.to_string())

    @staticmethod
    def get_f_gini(df, fillna=None):
        if fillna is not None:
            df = df.copy().fillna(fillna)
        try:
            return (2 * roc_auc_score(df.iloc[:, 0], -df.iloc[:, 1]) - 1) * 100
        except:
            return 0

    @staticmethod
    def get_features_gini(df, fillna=None):
        return {f: DataSamples.get_f_gini(df[[df.columns[0], f]], fillna=fillna) for f in list(df.columns[1:])}

    @staticmethod
    def get_time_features_gini(df, time_column, fillna=None):
        return {time: DataSamples.get_features_gini(group.drop([time_column], axis=1), fillna=fillna) for time, group in df.groupby(time_column)}

    def calc_gini(self, features=None, fillna=None, add_description=False, abs=False, mode=0):
        """
        Вычисление джини всех переменных, словарь вида {название_сэмпла: {переменная: джини}} сохраняется в self.ginis
        :param features: список переменных для расчета. При None берется self.features
        :param fillna: значение для заполнения пропусков. При None пропуски не заполняются
        :param add_description: флаг для добавления в датафрейм описания перемнных из self.feature_descriptions
        :param abs: возвращать абсолютные значения джини
        :param mode: 0 - расчет джини на всех основных и бутсрэп сэмплах
                     1 - расчет джини только на всех основных сэмплах
                    -1 - расчет джини только на бутсрэп сэмплах

        :return: датафрейм с джини
        """
        if features is None:
            features = self.features
        if mode >= 0:
            self.ginis = {name: self.get_features_gini(sample[[self.target] + [f for f in features if f in sample.columns]], fillna=fillna)
                          for name, sample in self.samples.items()}
        if self.bootstrap_base is not None and mode <= 0:
            bts_features = [f for f in features if f in self.bootstrap_base.columns]
            if bts_features:
                if self.n_jobs_restr > 1:
                    with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                        ginis_bootstrap = []
                        jobs = []
                        iterations = len(self.bootstrap)
                        idx_iter = iter(self.bootstrap)
                        while iterations:
                            for idx in idx_iter:
                                jobs.append(pool.submit(self.get_features_gini, df=self.bootstrap_base.iloc[idx][[self.target] + bts_features], fillna=fillna))
                                if len(jobs) > self.max_queue:
                                    break
                            for job in futures.as_completed(jobs):
                                ginis_bootstrap.append(job.result())
                                jobs.remove(job)
                                iterations -= 1
                                break
                    gc.collect()
                else:
                    ginis_bootstrap = [self.get_features_gini(self.bootstrap_base.iloc[idx][[self.target] + bts_features], fillna=fillna)
                                       for idx in self.bootstrap]

                self.ginis['Bootstrap mean'] = {f: np.mean([ginis[f] for ginis in ginis_bootstrap]) for f in bts_features}
                self.ginis['Bootstrap std'] = {f: np.std([ginis[f] for ginis in ginis_bootstrap]) for f in bts_features}
        result = pd.DataFrame(self.ginis).round(2)
        if abs:
            result = result.abs()
        if add_description:
            try:
                tmp = self.feature_descriptions.copy()
                tmp.index += '_WOE'
                result = pd.concat([self.feature_descriptions, tmp]).merge(result, left_index=True, right_index=True, how='right')
            except:
                pass
        self.gini_df = result
        return result

    def calc_gini_in_time(self, features=None, fillna=None, ds_aux=None, abs=False):
        """
        Вычисление динамики джини по срезам для всех переменных, словарь вида {название_сэмпла: {переменная: {срез: джини}}} сохраняется в self.ginis_in_time.
        Доступно только если задано значение self.time_column
        :param features: писок переменных для расчета. При None берется self.features
        :param fillna: значение для заполнения пропусков. При None пропуски не заполняются
        :param ds_aux: вспомогательный ДатаСэмпл с полем среза
        :param abs: возвращать абсолютные значения джини

        :return: датафрейм с джини
        """
        if self.time_column is not None:
            time_column = self.time_column
            samples = self.samples
        elif ds_aux is not None and ds_aux.time_column is not None:
            time_column = ds_aux.time_column
            samples = copy.deepcopy(self.samples)
            for name in samples:
                samples[name][time_column] = ds_aux.samples[name][time_column]
        else:
            return pd.DataFrame()

        if features is None:
            features = self.features
        self.ginis_in_time = {name: self.get_time_features_gini(sample[[self.target, time_column] + [f for f in features if f in sample.columns]],
                                                                time_column=time_column, fillna=fillna)
                              for name, sample in samples.items()}

        if self.bootstrap_base is not None:
            bts_features = [f for f in features if f in self.bootstrap_base.columns]
            if bts_features:
                if self.time_column is not None:
                    bootstrap_base = self.bootstrap_base
                else:
                    bootstrap_base = self.bootstrap_base.copy()
                    bootstrap_base[time_column] = ds_aux.bootstrap_base[time_column]
                if self.n_jobs_restr > 1:
                    with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                        ginis_bootstrap = []
                        jobs = []
                        iterations = len(self.bootstrap)
                        idx_iter = iter(self.bootstrap)
                        while iterations:
                            for idx in idx_iter:
                                jobs.append(pool.submit(self.get_time_features_gini,
                                                        df=bootstrap_base.iloc[idx][[self.target, time_column] + [f for f in bts_features]],
                                                        time_column=time_column, fillna=fillna))
                                if len(jobs) > self.max_queue:
                                    break
                            for job in futures.as_completed(jobs):
                                ginis_bootstrap.append(job.result())
                                jobs.remove(job)
                                iterations -= 1
                                gc.collect()
                                break
                    gc.collect()
                else:
                    ginis_bootstrap = [self.get_time_features_gini(bootstrap_base.iloc[idx][[self.target, time_column] + bts_features],
                                                                   time_column=time_column, fillna=fillna)
                                       for idx in self.bootstrap]
                time_values = sorted(bootstrap_base[time_column].unique())
                self.ginis_in_time['Bootstrap mean'] = {time: {f: np.mean([ginis[time][f] for ginis in ginis_bootstrap if time in ginis])
                                                               for f in bts_features}
                                                        for time in time_values}
                self.ginis_in_time['Bootstrap std'] = {time: {f: np.std([ginis[time][f] for ginis in ginis_bootstrap if time in ginis])
                                                              for f in bts_features}
                                                       for time in time_values}

        time_values = sorted(list({time for name in self.ginis_in_time for time in self.ginis_in_time[name]}))
        result = pd.DataFrame([[time] + [self.ginis_in_time[name][time][f] if time in self.ginis_in_time[name] and f in self.ginis_in_time[name][time] else 0
                                         for f in features for name in self.ginis_in_time]
                               for time in time_values],
                            columns=[time_column] + [f'Gini {name} {f}' for f in features for name in self.ginis_in_time]).set_index(time_column).round(2)
        result.columns = pd.MultiIndex.from_product([features, list(self.ginis_in_time.keys())])
        if abs:
            result = result.abs()
        return result

    def corr_mat(self, sample_name=None, features=None, corr_method='pearson', corr_threshold=0.75):
        """
        Вычисление матрицы корреляций
        :param sample_name: название сэмпла, из которого берутся данные. По умолчанию self.train_name
        :param features: список переменных для расчета. По умолчанию берутся из self.features
        :param corr_method: метод расчета корреляций. Доступны варианты 'pearson', 'kendall', 'spearman'
        :param corr_threshold: трэшхолд значения корреляции. Используется для выделения значений цветом

        :return: датафрейм с матрицей корреляций
        """
        if features is None:
            features = self.features
        if sample_name is None:
            sample_name = self.train_name
        corr_df = self.samples[sample_name][features].corr(method=corr_method)
        if self.gini_df is None:
            gini_df = corr_df
        else:
            gini_df = self.gini_df.merge(corr_df, left_index=True, right_index=True, how='right')
        nums = list(range(1, len(corr_df.columns) + 1))
        gini_df.columns = [f for f in gini_df.columns if f not in corr_df.columns] + nums
        gini_df.index = ['%s (%d)' % (x, i + 1) for i, x in enumerate(gini_df.index)]
        return gini_df.round(2).style.applymap(
                lambda x: 'color: black' if isinstance(x, str) or x > 1 or x < -1 else 'color: red'
                if abs(x) > corr_threshold else 'color: orange'
                if abs(x) > corr_threshold ** 2 else 'color: green', subset=pd.IndexSlice[:, nums])

    def psi(self, time_column=None, sample_name=None, features=None, normalized=True, yellow_zone=0.1, red_zone=0.25,
            base_period_index=0, n_bins=5, legend_map=None):
        """
        Вычисление Population Stability Index
        StabilityIndex[t] = (N[i, t]/sum_i(N[i, t]) - (N[i, 0]/sum_i(N[i, 0])))* log(N[i, t]/sum_i(N[i, t])/(N[i, 0]/sum_i(N[i, 0])))
        где N[i, t]  - кол-во наблюдений со значением i в срезе t.

        :param time_column: название поля, по которому формируются срезы. По умочланию берется self.time_column
        :param sample_name: название сэмпла, из которого берутся данные. По умолчанию self.train_name
        :param features: список переменных для расчета. По умолчанию берется self.features
        :param normalized: расчет доли наблюдений вместо абсолютного кол-ва
        :param yellow_zone: нижняя граница желтой зоны значения PSI
        :param red_zone: нижняя граница красерй зоны значения PSI
        :param base_period_index: индекс основного среза в отсортированном списке значений срезов, относительного которого считается PSI остальных срезов
        :param n_bins: кол-во бинов на которые будут разбиты значения переменных, если кол-во уникальных значений > 20
        :param legend_map: словарь мэппинга легенды вида {переменная: {исходное значение: новое значение}}. Используется для добавления описания значений WOE в легенде графика PSI

        :return: кортеж (Датафрейм,  список из графиков PSI [plt.figure])
        """
        if sample_name is None:
            sample_name = self.train_name
        if time_column is None:
            if self.time_column is not None:
                time_column = self.time_column
            else:
                print('Please set time_column in DataSample for using this method.')
                return None

        if features is None:
            features = self.features

        tmp_dataset = self.samples[sample_name][features + [time_column, self.target]].copy()
        for c, feature in enumerate(features):
            if tmp_dataset[feature].nunique() > 20:
                if self.special_bins:
                    tmp_dataset = tmp_dataset[~tmp_dataset[feature].isin(list(self.special_bins.keys()))]
                tmp_dataset[feature] = pd.cut(tmp_dataset[feature], n_bins, include_lowest=True, duplicates='drop').astype('str')
            feature_stats = tmp_dataset[[feature, time_column, self.target]] \
                .groupby([feature, time_column]).agg(size=(self.target, 'size'), mean=(self.target, 'mean')) \
                .reset_index().rename({feature: 'value'}, axis=1)
            feature_stats['feature'] = feature
            if c == 0:
                all_stats = feature_stats
            else:
                all_stats = all_stats.append(feature_stats, ignore_index=True)

        all_stats['size'] = all_stats['size'].astype(float)
        all_stats['mean'] = all_stats['mean'].astype(float)
        figs = []
        stability1 = all_stats[all_stats.feature.isin(features)][['feature', 'value', time_column, 'size']] \
            .pivot_table(values='size', columns=time_column, index=['feature', 'value']).reset_index().fillna(0)
        stability1.columns.name = None
        dates = stability1.drop(['feature', 'value'], 1).columns.copy()
        stability2 = stability1[['feature', 'value']].copy()
        for date in dates:
            stability2[date] = list(stability1[date] / list(stability1.drop(['value'], 1).groupby(by='feature').sum()[date][:1])[0])
        start_date = dates[base_period_index]
        stability3 = stability2[['feature', 'value']]
        for date in dates:
            stability3[date] = round(((stability2[date] - stability2[start_date]) * np.log(
                stability2[date] / stability2[start_date])).replace([+np.inf, -np.inf], 0).fillna(0), 2)
        stability4 = stability3.drop(['value'], 1).groupby(by='feature').sum()
        result = stability4.reindex(index=all_stats['feature'].drop_duplicates()).style.apply(color_background, mn=0, mx=red_zone, cntr=yellow_zone)

        date_base = pd.DataFrame(all_stats[time_column].unique(), columns=[time_column]).sort_values(time_column)
        for num_f, feature in enumerate(features):
            cur_feature_data = all_stats[all_stats['feature'] == feature].copy()
            if normalized:
                for tt in sorted(cur_feature_data[time_column].unique(), reverse=True):
                    cur_feature_data.loc[cur_feature_data[time_column] == tt, 'percent'] = \
                    cur_feature_data[cur_feature_data[time_column] == tt]['size'] / \
                    cur_feature_data[cur_feature_data[time_column] == tt]['size'].sum()
            fig, ax = plt.subplots(1, 1, figsize=(7 + tmp_dataset[time_column].nunique(), 5))
            ax2 = ax.twinx()
            ax.grid(False)
            ax2.grid(False)

            sorted_values = sorted(cur_feature_data['value'].unique(), reverse=True)
            for value in sorted_values:
                to_visualize = 'percent' if normalized else 'size'
                value_filter = (cur_feature_data['value'] == value)
                er = date_base.merge(cur_feature_data[value_filter], on=time_column, how='left')['mean']
                height = date_base.merge(cur_feature_data[value_filter], on=time_column, how='left')[
                    to_visualize].fillna(0)
                bottom = date_base.merge(cur_feature_data[[time_column, to_visualize]][cur_feature_data['value'] > value]\
                                         .groupby(time_column, as_index=False).sum(), on=time_column, how='left')[to_visualize].fillna(0)

                ax.bar(range(date_base.shape[0]), height, bottom=bottom if value != sorted_values[0] else None,
                       edgecolor='white', alpha=0.3)
                if isinstance(value, str):
                    ax2.plot(range(date_base.shape[0]), er, label=value, linewidth=2)
                else:
                    ax2.plot(range(date_base.shape[0]), er, label=str(round(value, 3)), linewidth=2)
            plt.xticks(range(date_base.shape[0]), date_base[time_column])
            fig.autofmt_xdate()

            ax2.set_ylabel('Target Rate')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
            if normalized:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
                ax2.annotate('Amount:', xy=(0, 1), xycoords=('axes fraction', 'axes fraction'),
                             xytext=(-25, 35), textcoords='offset pixels', color='black', size=11)
                for i in range(date_base.shape[0]):
                    ax2.annotate(str(int(cur_feature_data[cur_feature_data[time_column] == date_base[time_column][i]]['size'].sum())),
                                 xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, 35),
                                 textcoords='offset pixels', ha='center', color='black', size=11)
                ax2.annotate('PSI:', xy=(0, 1), xycoords=('axes fraction', 'axes fraction'), xytext=(-25, 10),
                             textcoords='offset pixels', color='black', size=11)
                for i in range(date_base.shape[0]):
                    psi = round(stability4[stability4.index == feature][date_base[time_column][i]].values[0], 2)
                    ax2.annotate(str(psi), xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, 10),
                                 textcoords='offset pixels', ha='center', size=11,
                                 color='green' if psi < yellow_zone else 'orange' if psi < red_zone else 'red')
            ax.set_ylabel('Observations')
            plt.xlabel(time_column)
            plt.xlim([-0.5, len(date_base) + (2 if legend_map else 0.5)])
            plt.suptitle(self.feature_titles[feature] if feature in self.feature_titles else feature, fontsize=16, weight='bold')
            plt.tight_layout()
            handles, labels = ax2.get_legend_handles_labels()
            if legend_map:
                labels = [legend_map[feature][l] if feature in legend_map and l in legend_map[feature] else l for l in labels]
            ax2.legend(handles[::-1], labels[::-1], loc='upper right', fontsize=10)
            figs.append(fig)
        plt.close('all')
        return result, figs

    def plot_distribution(self, features=None, bins=20, round_digits=3, plot_flag=True):
        """
        Отрисовка распределения значений переменной. Работает как с непрервными, так и дескретными переменными
        :param features: список переменных для обработки
        :param bins: кол-во бинов в распределении. Если в переменной число уникальных значений больше этого кол-ва, то она перебинивается
        :param round_digits: кол-во знаков после запятой
        :param plot_flag: флаг для вывода распределения

        :return: список из графиков [plt.figure]
        """
        if features is None:
            features = self.features
        figs = []
        for feature in features:
            to_cut = self.samples[self.train_name][feature]
            if to_cut.nunique() > bins:
                if self.special_bins:
                    to_cut = to_cut[~to_cut.isin(list(self.special_bins.values()))]
                _, cuts = pd.cut(to_cut, bins=bins, right=False, precision=round_digits, retbins=True)
                cuts[0] = -np.inf
                cuts[-1] = np.inf
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111)
            num = 0
            for name, sample in self.samples.items():
                if to_cut.nunique() > bins:
                    stats = pd.cut(sample[feature], bins=cuts, right=False, precision=round_digits).value_counts().sort_index()
                else:
                    stats = to_cut.value_counts().sort_index()
                stats = stats/stats.sum()
                plt.bar(np.array(range(stats.shape[0])) + num*0.2, stats, width=0.2, label=name)
                if name == self.train_name:
                    plt.xticks(np.array(range(stats.shape[0])) + len(self.samples) * 0.2 / 2, stats.index.astype(str))
                num += 1

            fig.autofmt_xdate()
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
            plt.legend()
            plt.suptitle(self.feature_titles[feature] if feature in self.feature_titles else feature, fontsize=16, weight='bold')
            if plot_flag:
                plt.show()
            figs.append(fig)
        plt.close('all')
        return figs

    @staticmethod
    def targettrend_feature(df, special_bins, feature_titles, quantiles, plot_flag):
        magnify_trend = False
        magnify_std_number = 2
        hide_every_even_tick_from = 50
        min_size = 10
        target = df.columns[0]
        f = df.columns[1]
        tmp = df.copy()
        if special_bins:
            tmp = tmp[~tmp[f].isin(list(special_bins.values()))]
        if tmp[f].dtype not in (float, np.float32, np.float64, int, np.int32, np.int64) or tmp[f].unique().shape[0] < quantiles:
            summarized = tmp[[f, target]].groupby([f]).agg(['mean', 'size'])
        else:
            tmp = tmp.dropna()
            if tmp[f].shape[0] < min_size * quantiles:
                current_quantiles = int(tmp[f].shape[0] / min_size)
                if current_quantiles == 0:
                    return None
            else:
                current_quantiles = quantiles
            summarized = tmp[[target]].join(pd.qcut(tmp[f], q=current_quantiles, precision=4, duplicates='drop')).groupby([f]).agg(['mean', 'size'])
            small_quantiles = summarized[target][summarized[target]['size'] < min_size]['size']
            if small_quantiles.shape[0] > 0:
                current_quantiles = int(small_quantiles.sum() / min_size) + summarized[target][summarized[target]['size'] >= min_size].shape[0]
                summarized = tmp[[target]].join(pd.qcut(tmp[f], q=current_quantiles, precision=3, duplicates='drop')).groupby([f]).agg(['mean', 'size'])

        summarized.columns = summarized.columns.droplevel()
        summarized = summarized.reset_index()
        if pd.isnull(df[f]).any():
            with_na = df[[f, target]][pd.isnull(df[f])]
            summarized.loc[-1] = [np.nan, with_na[target].mean(), with_na.shape[0]]
            summarized = summarized.sort_index().reset_index(drop=True)
        if special_bins:
            add_rows = []
            for k, v in special_bins.items():
                special_group = df[[f, target]][df[f] == v]
                if special_group.shape[0] > 0:
                    add_rows.append([k, special_group[target].mean(), special_group.shape[0]])
            if add_rows:
                summarized = pd.concat([pd.DataFrame(add_rows, columns=[f, 'mean', 'size']), summarized])
        if summarized.shape[0] == 1:
            return None

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        ax.set_ylabel('Observations')
        # blue is for the distribution
        if summarized.shape[0] > hide_every_even_tick_from:
            plt.xticks(range(summarized.shape[0]), summarized[f].astype(str), rotation=60, ha="right")
            xticks = ax.xaxis.get_major_ticks()
            for i in range(len(xticks)):
                if i % 2 == 0:
                    xticks[i].label1.set_visible(False)
        else:
            plt.xticks(range(summarized.shape[0]), summarized[f].astype(str), rotation=45, ha="right")

        ax.bar(range(summarized.shape[0]), summarized['size'], zorder=0, alpha=0.3)
        ax.grid(False)
        ax.grid(axis='y', zorder=1, alpha=0.6)
        ax2 = ax.twinx()
        ax2.set_ylabel('Target Rate')
        ax2.grid(False)

        if magnify_trend:
            ymax = np.average(summarized['mean'], weights=summarized['size']) + magnify_std_number * np.sqrt(
                np.cov(summarized['mean'], aweights=summarized['size']))
            if pd.isnull(ymax):
                ymax = summarized['mean'].mean()
            ax2.set_ylim([0, ymax])
            for i in range(len(summarized['mean'])):
                if summarized['mean'][i] > np.average(summarized['mean'], weights=summarized['size']) + magnify_std_number * np.sqrt(np.cov(summarized['mean'], aweights=summarized['size'])):
                    ax2.annotate(str(round(summarized['mean'][i], 4)),
                                 xy=(i, np.average(summarized['mean'], weights=summarized['size']) + magnify_std_number * np.sqrt(
                                     np.cov(summarized['mean'], aweights=summarized['size']))),
                                 xytext=(i, np.average(summarized['mean'], weights=summarized['size']) + (magnify_std_number + 0.05) * np.sqrt(
                                     np.cov(summarized['mean'], aweights=summarized['size']))),
                                 rotation=60, ha='left', va='bottom', color='red', size=8.5
                                 )
        # red is for the target rate values
        ax2.plot(range(summarized.shape[0]), summarized['mean'], 'ro-', linewidth=2.0, zorder=4)
        plt.suptitle(feature_titles[f] if f in feature_titles else f, fontsize=16, weight='bold')
        plt.tight_layout()
        if plot_flag:
            plt.show()
        return fig
    
    def targettrend(self, features=None, quantiles=10, plot_flag=False):
        """
        Вычисление распределения таргета по каждой переменной из заданного списка
        :param features: список переменных
        :param quantiles: кол-во квантилей для разбиения непрерыных переменных
        :param plot_flag: флаг для вывода распределения

        :return: список из графиков [plt.figure]
        """
        if features is None:
            features = self.features
        if self.n_jobs_restr > 1:
            with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                figs = list(pool.map(partial(self.targettrend_feature, special_bins=self.special_bins,
                                             feature_titles=self.feature_titles, quantiles=quantiles, plot_flag=plot_flag),
                                     [self.samples[self.train_name][[self.target, f]] for f in features]))
        else:
            figs = [self.targettrend_feature(df=self.samples[self.train_name][[self.target, f]], special_bins=self.special_bins,
                                             feature_titles=self.feature_titles, quantiles=quantiles, plot_flag=plot_flag)
                    for f in features]
        plt.close('all')
        return figs

    def samples_split(self, df=None, test_size=0.3, validate_size=0, split_type='oos', stratify=True, id_column=None):
        """
        Разбивка датафрейма на сэмплы
        :param df: датафрейм из которого нарезаются сэмплы. При None берется self.samples[self.train_name]
        :param test_size: размер сэмпла test
        :param validate_size: размер сэмпла validate
        :param split_type: тип разбиения 'oos' = 'out-of-sample', 'oot' = 'out-of-time'
        :param stratify: стратификация по целевой переменной. Только для split_type='oos'
        :param id_column: название поля с айди. Если задано, то все одинаковые айди распределяются в один сэмпл,
                          в случае split_type='oot' из теста исключаются айди, присутсвующие в трэйне. Размер теста при этом может стать сильно меньше test_size
        """
        if df is None:
            df = self.samples[self.train_name].copy()
        else:
            self.train_name = 'Train'
        if split_type == 'oos':
            for_split = df[self.target] if id_column is None else df.groupby(by=id_column)[self.target].max()
            split = train_test_split(for_split, test_size=test_size, random_state=self.random_state, stratify=for_split if stratify else None)
            if validate_size > 0:
                split[0], validate = train_test_split(split[0], test_size=validate_size/(1 - test_size), random_state=self.random_state, stratify=split[0] if stratify else None)
                split.append(validate)

            if id_column is not None:
                dfs = []
                dr = df[self.target].mean()
                for sp in split:
                    tmp = df[df[id_column].isin(sp.index)]
                    dr_tmp = tmp[self.target].mean()
                    if dr_tmp > dr:
                        g = tmp[tmp[self.target] == 0]
                        tmp = pd.concat([g, tmp[tmp[self.target] == 1].sample(n=round(len(g) * dr / (1 - dr)), random_state=self.random_state)])
                    else:
                        b = tmp[tmp[self.target] == 1]
                        tmp = pd.concat([tmp[tmp[self.target] == 0].sample(n=round(len(b) * (1 - dr) / dr), random_state=self.random_state), b])
                    dfs.append(tmp)
            else:
                dfs = [df.loc[list(sp.index)] for sp in split]

        elif split_type == 'oot':
            if self.time_column is None:
                print ('Wich column contains time data? Please pay attention to time_column parameter.')
                return None
            if validate_size > 0:
                print ('Validation for oot is unavailable.')
            else:
                tmp_dataset = copy.deepcopy(df).sort_values(by=self.time_column)
                test_threshold = list(tmp_dataset[self.time_column].drop_duplicates())[int(round((1 - test_size)*len(tmp_dataset[self.time_column].drop_duplicates()), 0))]
                dfs = [tmp_dataset[tmp_dataset[self.time_column] < test_threshold],
                       tmp_dataset[tmp_dataset[self.time_column] >= test_threshold]]
                if id_column is not None:
                    dfs[1] = dfs[1][~dfs[1][id_column].isin(dfs[0][id_column])]
        else:
            print ('Wrong split type. Please use oot or oos.')
            return None

        sample_names = [self.train_name, 'Test', 'Validate']
        self.samples = {sample_names[i]: df1 for i, df1 in enumerate(dfs)}
        print('Actual parts of samples:')
        for name, sample in self.samples.items():
            print(f'{name}: {round(sample.shape[0]/df.shape[0], 4)}')

    def bootstrap_split(self, df, bootstrap_part=0.75, bootstrap_number=10, stratify=True, replace=True):
        """
        Создание подвыборок для бутстрэпа
        :param df: датафрейм, основа для нарезания подвыборок
        :param bootstrap_part: размер каждой подвыборки
        :param bootstrap_number: кол-во подвыборок
        :param stratify: стратификация каждой подвыборки по целевой переменной
        :param replace: разрешается ли повторять каждое наблюдение множество раз в подвыборке
        """
        self.bootstrap_base = df
        if 'Infinity' in self.special_bins:
            self.bootstrap_base[self.features] = self.bootstrap_base[self.features].replace([np.inf, -np.inf], self.special_bins['Infinity'])
        self.bootstrap = []
        if stratify:
            class0 = self.bootstrap_base[self.bootstrap_base[self.target] == 0]
            class1 = self.bootstrap_base[self.bootstrap_base[self.target] == 1]

        for bi in range(bootstrap_number):
            if stratify:
                index_1 = class1.sample(frac=bootstrap_part, replace=replace, random_state=self.random_state+bi).index
                index_0 = class0.sample(frac=bootstrap_part, replace=replace, random_state=self.random_state+bi).index
                bootstrap_current = [self.bootstrap_base.index.get_loc(idx) for idx in index_1.append(index_0)]
            else:
                bootstrap_current = [self.bootstrap_base.index.get_loc(idx)
                                     for idx in self.bootstrap_base.sample(frac=bootstrap_part, replace=replace, random_state=self.random_state+bi).index]
            self.bootstrap.append(bootstrap_current)

    def CorrelationAnalyzer(self, sample_name=None, features=None, hold=None, method='pearson', threshold=0.6,
                             drop_with_most_correlations=True, verbose=False):
        """
        Корреляционный анализ переменных на выборке, формирование словаря переменных с причиной для исключения
        :param sample_name: название сэмпла на котором проводится отбор. При None берется ds.train_sample
        :param features: исходный список переменных для анализа. При None берется self.features
        :param hold: список переменных, которые обязательно должны войти в модель
        :param method: метод расчета корреляций. Доступны варианты 'pearson', 'kendall', 'spearman'
        :param threshold: граница по коэффициенту корреляции
        :param drop_with_most_correlations:  при True - итерационно исключается фактор с наибольшим кол-вом коррелирующих с ним факторов с корреляцией выше threshold
                                             при False - итерационно исключается фактор с наименьшим джини из списка коррелирующих факторов
        :param verbose: флаг для вывода подробных комментариев в процессе работы

        :return: словарь переменных для исключения вида {переменная: причина исключения}
        """
        if hold is None:
            hold = []

        if sample_name is None:
            sample_name = self.train_name        

        if not self.ginis or not self.ginis[sample_name]:
            self.calc_gini()

        if features is None:
            features = self.features

        correlations = self.samples[sample_name][features].corr(method=method).abs()
        to_check_correlation=True
        features_to_drop = {}
        while to_check_correlation:
            to_check_correlation=False
            corr_number = {}
            significantly_correlated={}
            for var in correlations:
                var_corr = correlations[var]
                var_corr = var_corr[(var_corr.index != var) & (var_corr > threshold)].sort_values(ascending=False).copy()
                corr_number[var] = var_corr.shape[0]
                significantly_correlated[var] = str(var_corr.index.tolist())
            if drop_with_most_correlations:
                with_correlation = {x: self.ginis[sample_name][x] for x in corr_number
                                    if corr_number[x] == max([corr_number[x] for x in corr_number if x not in hold])
                                    and corr_number[x] > 0 and x not in hold}
            else:
                with_correlation = {x: self.ginis[sample_name][x] for x in corr_number if corr_number[x] > 0 and x not in hold}
            if len(with_correlation)>0:
                feature_to_drop=min(with_correlation, key=with_correlation.get)
                features_to_drop[feature_to_drop] = f'High correlation with features: {significantly_correlated[feature_to_drop]}'
                correlations = correlations.drop(feature_to_drop, axis=1).drop(feature_to_drop, axis=0).copy()
                to_check_correlation = True
        if verbose:
            print(f'Dropped correlated features: {list(features_to_drop.keys())}')
        return features_to_drop

    def VIF(self, sample_name=None, features=None):
        """
        Рассчет VIF для списка переменных
        :param sample_name: название сэмпла на котором проводится расчет. При None берется ds.train_sample
        :param features: список переменных для анализа. При None берется self.features

        :return: ДатаФрейм с индексом из списка переменных и полем с расчитанным VIF
        """

        if sample_name is None:
            sample_name = self.train_name

        if features is None:
            features = self.features
        for f in features:
            if not pd.api.types.is_numeric_dtype(self.samples[sample_name][f]):
                print('All features must be a numeric!')
                return None
        # Break into left and right hand side; y and X
        y_, X_ = dmatrices(formula_like=self.target + ' ~ ' + '+'.join(['Q("' + f + '")' for f in features]),
                           data=self.samples[sample_name], return_type="dataframe")

        # For each Xi, calculate VIF
        return pd.DataFrame({features[i - 1]: variance_inflation_factor(X_.values, i) for i in range(1, X_.shape[1])}, index=[0]).T