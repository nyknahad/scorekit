# -*- coding: utf-8 -*-

from ..data import DataSamples
from ..woe import WOE
from ..bankmetrics import *
from .._utils import fig_to_excel, adjust_cell_width, add_suffix, rem_suffix
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, r2_score
from scipy.stats import chi2, chisquare, ks_2samp, ttest_ind
import warnings
from abc import ABCMeta, abstractmethod
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import ast
import os
import xlsxwriter
from PIL import Image
import datetime
from dateutil.relativedelta import *
from scipy.optimize import minimize
import scipy.stats as sts
import copy
import itertools
import calendar
import openpyxl
import json
from concurrent import futures
from functools import partial
import gc
import cloudpickle
import base64

warnings.simplefilter('ignore')
plt.rc('font', family='Verdana', size=12)
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)
gc.enable()


class ScoringModel(metaclass=ABCMeta):
    '''
    Base class for binary scoring models
    '''

    @abstractmethod
    def __init__(self, clf):
        self.clf = clf
        self.features = []

    @abstractmethod
    def fit(self, data):
        pass
#---------------------------------------------------------------


class DecisionTreeModel(ScoringModel):
    '''
    Decision tree classifier
    '''
    def __init__(self, **args):
        self.clf = DecisionTreeClassifier(**args)
        self.features = []

    def fit(self, ds):
        self.clf.fit(ds.samples[ds.train_name][ds.features], ds.samples[ds.train_name][ds.target])

#---------------------------------------------------------------

class LogisticRegressionModel(ScoringModel):
    """
    Классификатор лог регрессии
    """

    def __init__(self, clf=None, ds=None, transformer=None, round_digits=3, name=None):
        """
        :param clf: классификатор модели (должен иметь метод fit() и атрибуты coef_, intercept_). При None выбирается SGDClassifier(alpha=0.001, loss='log', max_iter=100)
        :param ds: Привязанный к модели ДатаСэмпл. Если задан, то он по умолчанию будет использоваться во всех методах
        :param transformer: объект класса WOE для трансформации факторов
        :param round_digits: округление коэффициентов до этого кол-ва знаков после запятой
        :param name: название модели
        """
        self.ds = copy.deepcopy(ds)
        self.transformer = transformer
        if self.ds is not None and self.transformer is not None:
            self.ds = self.transformer.transform(self.ds, verbose=True)
        try:
            random_state = self.ds.random_state
        except:
            random_state = 0
        if clf is not None:
            self.clf = clf
        else:
            self.clf = SGDClassifier(loss='log', penalty='l2', max_iter=1000, alpha=0.001, random_state=random_state)
        print(f'Chosen model classifier is {self.clf}')
        self.round_digits = round_digits
        self.name = name
        self.coefs = {}
        self.intercept = None
        self.features = []
        self.calibration = None
        self.scale = None

    def save_model(self, file_name='model.json'):
        """
        Сохранение факторов, коэффициентов, калибровки, шкалы и биннинга в файл
        :param file_name: название json файла для сохранения модели. При None json возвращается методом
        """
        model = {p: self.__dict__[p] for p in ['name', 'coefs', 'intercept', 'calibration', 'scale', 'round_digits']}
        model['clf'] = base64.b64encode(cloudpickle.dumps(self.clf)).decode()
        if self.transformer is not None:
            model['scorecard'] = base64.b64encode(cloudpickle.dumps(self.transformer.export_scorecard(features=[f[:-4] for f in self.coefs], full=False).drop(['target_rate', 'sample_part'], axis=1))).decode()
        if file_name is not None:
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(model, file, ensure_ascii=False, indent=4, cls=NpEncoder)
            print(f'The model was successfully saved to file {file_name}')
        else:
            return model

    def load_model(self, file_name='model.json', verbose=True):
        """
        Загрузка факторов, коэффициентов, калибровки, шкалы и биннинга из файла
        :param file_name: название json файла или словарь для загрузки модели
        """
        if isinstance(file_name, str):
            model = json.load(open(file_name, 'rt', encoding='utf-8'))
            print(f'The model was loaded from file {file_name}')
        elif isinstance(file_name, dict):
            model = file_name
        else:
            print('file_name type must be str or dict!')
            return None
        if 'clf' in model:
            self.clf = cloudpickle.loads(base64.b64decode(model['clf']))
        else:
            self.clf = LogisticRegression(random_state=model['random_state'] if 'random_state' in model else 0,
                                          C=model['regularization_value'] if 'regularization_value' in model else 1000,
                                          solver=model['solver'] if 'solver' in model else 'saga',
                                          penalty=model['regularization'] if 'regularization' in model else 'l2')
        if verbose:
            print(f'clf = {self.clf}')
        for p in ['name', 'coefs', 'intercept', 'calibration', 'scale', 'round_digits']:
            if p in model:
                self.__dict__[p] = model[p]
                if verbose:
                    print(f'{p} = {model[p]}')
        if 'scorecard' in model:
            try:
                scorecard = pd.DataFrame(model['scorecard'])
            except:
                scorecard = cloudpickle.loads(base64.b64decode(model['scorecard']))
            self.transformer = WOE(scorecard=scorecard)
        if self.coefs:
            self.features = list(self.coefs.keys())

    def auto_logreg(self, data=None, target=None, time_column=None, id_column=None, feature_descriptions=None, n_jobs=None,
                    result_folder='', method='opt', validate=False, out='auto_model.xlsx', save_model='auto_model.json'):
        """
        Построение модели в автоматическом режиме с минимальным набором параметров
        --- Выборка ---
        :param data: ДатаФрейм или ДатаСэмпл.
                - если передается ДатаФрейм, то он разбивается на трейн/тест 70%/30%
                - если передается ДатаСэмпл, то все остальные параметры блока "Выборка" не используются
        :param target: целевая переменная
        :param time_column: дата среза
        :param id_column: уникальный в рамках среза айди наблюдения
        :param feature_descriptions: датафйрем с описанием переменных. Должен содержать индекс с названием переменных и любое кол-во полей с описанием, которые будут подтягиваться в отчеты

        --- Параметры метода --
        :param n_jobs: кол-во используемых рабочих процессов, при -1 берется число, равное CPU_LIMIT
        :param result_folder: папка, в которую будут сохраняться все результаты работы
        :param method: Метод автобиннинга: 'tree' - биннинг деревом, 'opt' - биннинг деревом с последующей оптимизацией границ бинов библиотекой optbinning
        :param validate: флаг для выполнения валидацонных тестов
        :param out: либо строка с названием эксель файла, либо объект pd.ExcelWriter для сохранения отчета
        :param save_model: название json файла для сохранения модели
        """
        if data is not None:
            if isinstance(data, pd.DataFrame):
                print(f"\n{' Creating DataSamples ':-^150}\n")
                self.ds = DataSamples(samples={'train': data}, target=target, time_column=time_column, id_column=id_column,
                                 feature_descriptions=feature_descriptions, result_folder=result_folder, n_jobs=n_jobs)
                self.ds.samples_split()
            else:
                self.ds = copy.deepcopy(data)
                if n_jobs is not None:
                    self.ds.n_jobs = n_jobs
                if result_folder and isinstance(result_folder, str):
                    if result_folder and not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    self.ds.result_folder = result_folder + ('/' if result_folder and not result_folder.endswith('/') else '')

        print(f"\n{' SFA ':-^150}\n")
        if self.transformer is None:
            self.transformer = WOE(ds=self.ds)
            self.transformer.auto_fit(plot_flag=-1, method=method, SM_on=False, BL_allow_Vlogic_to_increase_gini=15,
                                      G_on=False, WOEO_on=False)
        else:
            print('Using existing self.transformer.')
        print(f"\n{' MFA ':-^150}\n")
        if not self.coefs:
            self.ds = self.transformer.transform(self.ds, verbose=True)
            self.mfa(gini_threshold=10, PSI=validate, result_file=f'{self.name + "_" if self.name else ""}mfa.xlsx')
        else:
            print('Using existing self.coefs.')
            self.report(out=out, sheet_name=None, pvalue_threshold=0.05, PSI=validate, verbose=False)

        if save_model:
            self.save_model(file_name=f'{self.ds.result_folder}{save_model}')
        if validate:
            print(f"\n{' Validation ':-^150}\n")
            self.validate(result_file='auto_validation.xlsx')

    def mfa(self, ds=None, features=None, hold=None, verbose=True, gini_threshold=5,
            corr_method='pearson', corr_threshold=0.75, drop_with_most_correlations=False,
            selection_type='stepwise', pvalue_threshold=0.05, pvalue_priority=False,
            scoring='gini', score_delta=0.1, cv=None, drop_positive_coefs=True,
            result_file='mfa.xlsx', PSI=True):
        """
        Многофакторный отбор. Проходит в 4 основных этапа:

        1) Отбор по джини (исключаются все факторы с джини ниже gini_threshold)

        2) Корреляционный анализ. Доступны два варианта работы:
            drop_with_most_correlations=False - итерационно исключается фактор с наименьшим джини из списка коррелирующих факторов
            drop_with_most_correlations=True - итерационно исключается фактор с наибольшим кол-вом коррелирующих с ним факторов

        3) Итерационный отобор. Доступны три типа отбора:
            selection_type='forward' - все доступные факторы помещаются в список кандидатов, на каждом шаге из списка кандидатов определяется лучший* фактор и перемещается в модель
            selection_type='backward' - в модель включаются все доступные факторы, затем на каждом шаге исключается худший* фактор
            selection_type='stepwise' - комбинация 'forward' и 'backward'. Каждый шаг состоит из двух этапов:
                    на первом из списка кандидатов отбирается лучший* фактор в модель,
                    на втором из уже включенных факторов выбирается худший* и исключается

            *Определение лучшего фактора:
            При pvalue_priority=False лучшим фактором считается тот, который увеличивает метрику scoring модели на наибольшую величину.
                Если величина такого увеличения ниже score_delta, то лучший фактора не определяется, и добавления не происходит
            При pvalue_priority=True лучшим фактором считается фактор, который после добавления в модель имеет наименьшее p-value.
                Если величина этого p-value выше pvalue_threshold, то лучший фактора не определяется, и добавления не происходит

            *Определение худшего фактора:
            Худшим фактором в модели считается фактор с наибольшим p-value.
                Если величина этого p-value ниже pvalue_threshold, то худший фактора не определяется, и исключения не происходит

        4) Если выставлен флаг drop_positive_coefs=True, то из списка отобранных на этапе 3 факторов итерационно
            исключаются факторы с положительными коэффициентами и факторы с p_value > pvalue_threshold
        :param ds: ДатаСэмпл. В случае, если он не содержит трансформированные переменные, то выполняется трансформация трансформером self.transformer.
                   При None берется self.ds
        :param features: исходный список переменных для МФА. При None берутся все переменные, по которым есть активный биннинг
        :param hold: список переменных, которые обязательно должны войти в модель
        :param verbose: флаг для вывода подробных комментариев в процессе работы
        :param gini_threshold: граница по джини для этапа 1
        :param corr_method: метод расчета корреляций для этапа 2. Доступны варианты 'pearson', 'kendall', 'spearman'
        :param corr_threshold: граница по коэффициенту корреляции для этапа 2
        :param drop_with_most_correlations: вариант исключения факторов в корреляционном анализе для этапа 2
        :param selection_type: тип отбора для этапа 3
        :param pvalue_threshold: граница по p-value для этапа 3
        :param pvalue_priority: вариант определения лучшего фактора для этапа 3
        :param scoring: максимизируемая метрика для этапа 3.
                Варианты значений: 'gini', 'AIC', 'BIC' + все метрики доступные для вычисления через sklearn.model_selection.cross_val_score.
                Все информационные метрики после вычисления умножаются на -1 для сохранения логики максимизации метрики.
        :param score_delta: минимальный прирост метрики для этапа 3
        :param cv: параметр cv для вычисления скора sklearn.model_selection.cross_val_score для этапа 3.
                    При None берется StratifiedKFold(5, shuffle=True)
        :param drop_positive_coefs: флаг для выполнения этапа 4
        :param result_file: файл, в который будут сохраняться результаты мфа
        :param PSI: флаг для проведения анализа PSI отобранных в модель факторов
        """
        if ds is not None:
            self.ds = copy.deepcopy(ds)
            if self.transformer is not None:
                self.ds = self.transformer.transform(self.ds, features=features, verbose=verbose)
        if features is None:
            features = self.ds.features
        else:
            features = [add_suffix(f) if add_suffix(f) in self.ds.features else f for f in features]
        if hold is None:
            hold = []
        else:
            hold = [add_suffix(f) if add_suffix(f) in self.ds.features else f for f in hold]
        if cv is None:
            cv = StratifiedKFold(5, shuffle=True)

        sample_name = self.ds.train_name
        gini_df = self.ds.calc_gini(add_description=True, features=features).round(self.round_digits)
        ginis = self.ds.ginis[sample_name]
        
        drop_features_gini = [f for f in ginis if abs(ginis[f]) < gini_threshold and f not in hold]
        features = [f for f in sorted(ginis, key=ginis.get, reverse=True) if f not in drop_features_gini]
        if verbose:
            print(f'\n---------------------------------------------- Step 1 ----------------------------------------------')
            print(f'Dropped features with gini lower {gini_threshold}: {drop_features_gini}')
        if not features:
            print('Features set is empty. Break.')
            return None
        if verbose:
            print(f'\n---------------------------------------------- Step 2 ----------------------------------------------')
        drop_features_corr = self.ds.CorrelationAnalyzer(sample_name=sample_name, features=features, hold=hold,
                                                         drop_with_most_correlations=drop_with_most_correlations,
                                                         method=corr_method, threshold=corr_threshold, verbose=verbose)
        with pd.ExcelWriter(self.ds.result_folder + result_file,  engine='xlsxwriter') as writer:
            gini_df[gini_df.index.isin(drop_features_gini)].to_excel(writer, sheet_name='Gini below threshold')
            adjust_cell_width(writer.sheets['Gini below threshold'], gini_df)
            gini_df_copy = self.ds.gini_df.copy()
            self.ds.gini_df['Drop reason'] = self.ds.gini_df.index.map(drop_features_corr)
            self.ds.corr_mat(sample_name=sample_name).to_excel(writer, sheet_name='Correlation analysis')
            adjust_cell_width(writer.sheets['Correlation analysis'], self.ds.gini_df)
            self.ds.gini_df = gini_df_copy.copy()
            features = [f for f in features if f not in drop_features_corr]
            if verbose:
                print(f'\n---------------------------------------------- Step 3 ----------------------------------------------')
            if selection_type in ['stepwise', 'forward', 'backward']:
                features, selection_fig = self.stepwise_selection(features=features, hold=hold, verbose=verbose,
                                                                  score_delta=score_delta, scoring=scoring, cv=cv,
                                                                  pvalue_threshold=pvalue_threshold,
                                                                  pvalue_priority=pvalue_priority, selection_type=selection_type)
            else:
                selection_type = 'regularized'
                selection_fig = None
            if drop_positive_coefs:
                if verbose:
                    print(f'\n---------------------------------------------- Step 4 ----------------------------------------------')
                features, regularized_fig = self.regularized_selection(features=features, hold=hold, scoring=scoring, cv=cv,
                                                                       pvalue_threshold=pvalue_threshold, verbose=verbose)
            else:
                regularized_fig = None
            features = [f for f in sorted(ginis, key=ginis.get, reverse=True) if f in features]
            print(f'\n------------------------------------------- Final model --------------------------------------------')
            self.fit(features=features)
            self.report(out=writer, sheet_name=selection_type, pvalue_threshold=pvalue_threshold, PSI=PSI, verbose=verbose)
            row = 0
            if selection_fig:
                fig_to_excel(selection_fig, writer.sheets[selection_type], row=row, col=len(gini_df.columns) + max(len(self.features), len(self.ds.samples)) + 21)
                row = +30
            if regularized_fig:
                fig_to_excel(regularized_fig, writer.sheets[selection_type], row=row, col=len(gini_df.columns) + max(len(self.features), len(self.ds.samples)) + 21)
        plt.close('all')

    def report(self, ds=None, out='report.xlsx', sheet_name=None, pvalue_threshold=0.05, PSI=False, verbose=False):
        """
        Генерация отчета по обученной модели.
        :param ds: ДатаСэмпл. В случае, если он не содержит трансформированные переменные, то выполняется трансформация трансформером self.transformer
        :param out: либо строка с названием эксель файла, либо объект pd.ExcelWriter для сохранения отчета
        :param sheet_name: название листа в экселе
        :param pvalue_threshold: граница по p-value. Используется только для выделения значений p-value цветом
        :param PSI: флаг проведение тестов PSI
        :param verbose: флаг вывода комментариев в процессе работы
        """
        if ds is None:
            ds = self.ds
        if sheet_name is None:
            if self.name:
                sheet_name = self.name
            else:
                sheet_name = 'model'
        if not self.coefs:
            print('Please fit your model before calling this method.')
            return None
        if isinstance(out, str):
            writer = pd.ExcelWriter(ds.result_folder + out, engine='xlsxwriter')
        elif isinstance(out, pd.ExcelWriter):
            writer = out
        else:
            print('Parameter out must have str or pd.ExcelWriter type.')
            return None
        if self.transformer is not None:
            ds = self.transformer.transform(ds, features=self.features, verbose=verbose)
        if ds.gini_df is not None:
            gini_df = ds.gini_df.copy()
        else:
            gini_df = ds.calc_gini(features=self.features, add_description=True, abs=True).round(self.round_digits)
        gini_df = gini_df.rename_axis('feature').reset_index()
        for name, sample in ds.samples.items():
            wald_df = self.wald_test(ds, sample_name=name, features=self.features)
            if name != ds.train_name:
                wald_df = wald_df[['p-value', 'feature']]
            wald_df = wald_df.rename({'p-value': f'p-value {name}'}, axis=1)
            gini_df = gini_df.merge(wald_df, on='feature', how='right')
        gini_df = gini_df.merge(ds.VIF(features=self.features).reset_index().set_axis(['feature', 'VIF'], axis=1, inplace=False),
            on='feature', how='left').sort_values(by=[ds.train_name], ascending=False, na_position='last')

        if verbose:
            self.draw_coefs()
        ginis_model, roccurve_fig = self.roc_curve(ds, verbose=verbose)
        print('Generating report...')
        gini_df = pd.concat([gini_df, pd.DataFrame.from_dict({**ginis_model, **{'feature': 'model'}}, orient='index').T]).set_index('feature')
        gini_df.style.applymap(lambda x: 'color: red' if x > pvalue_threshold else 'color: orange' if x > pvalue_threshold / 5 else 'color: black',
                               subset=pd.IndexSlice[:, [f for f in gini_df if f.startswith('p-value')]]) \
            .to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, float_format=f'%0.{self.round_digits}f')
        ds.gini_df = None
        ds.corr_mat(features=self.features)\
            .to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=len(gini_df.columns) + 3)
        ws = writer.sheets[sheet_name]
        descr_len = sum([1 for f in gini_df.columns if not pd.api.types.is_numeric_dtype(gini_df[f]) and f not in ginis_model])
        ws.set_column(0, 0 + descr_len, 30)
        ws.set_column(1 + descr_len, gini_df.shape[1], 15)
        ws.set_column(len(gini_df.columns) + 3, gini_df.shape[1] + 3, 30)
        ws.write(0, 0, 'Features in model:')
        ws.write(0, len(gini_df.columns) + 3, 'Correlations matrix:')
        fig_to_excel(roccurve_fig, ws, row=0, col=gini_df.shape[1] + max(len(self.features), len(ds.samples)) + 6)

        if ds.time_column:
            ws.write(len(self.features) + 5, gini_df.shape[1] + 3, 'Model Gini dynamics:')
            model_gini = self.calc_gini_in_time(ds)
            model_gini.to_excel(writer, sheet_name=sheet_name, startrow=len(self.features) + 6, startcol=gini_df.shape[1] + 3)
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for name in ds.samples:
                ax.plot(model_gini.index, model_gini[name], label=name, marker='o')
            if ds.bootstrap_base is not None:
                ax.plot(model_gini.index, model_gini['Bootstrap mean'], label='Bootstrap mean', marker='o')
                ax.fill_between(model_gini.index,
                                model_gini['Bootstrap mean'] - 1.96 * model_gini['Bootstrap std'],
                                model_gini['Bootstrap mean'] + 1.96 * model_gini['Bootstrap std'],
                                alpha=0.1, color='blue', label='95% conf interval')
            ax.set_ylabel('Gini')
            ax.set_title('Model Gini dynamics')
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
            ax.tick_params(axis='both', which='both', length=5, labelbottom=True)
            ax.xaxis.get_label().set_visible(False)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
            fig_to_excel(fig, ws, row=len(self.features) + len(model_gini) + 9, col=len(gini_df.columns) + 1, scale=0.75)
            ws.write(len(self.features) + len(model_gini) + 34, len(gini_df.columns) + 3, 'Features Gini dynamics:')
            features_gini = ds.calc_gini_in_time(features=self.features, abs=True)
            features_gini.to_excel(writer, sheet_name=sheet_name, startrow=len(self.features) + len(model_gini) + 35,
                                   startcol=gini_df.shape[1] + 3)

        if self.transformer is not None:
            ws.write(len(self.features) + 5, 0, 'Scorecard:')
            scorecard = self.transformer.export_scorecard(features=[f.rsplit('_WOE', maxsplit=1)[0] for f in self.features], full=False)
            scorecard.set_index('feature').to_excel(writer, sheet_name=sheet_name,
                                                    startrow=len(self.features) + 6, startcol=0,
                                                    float_format=f'%0.{self.round_digits}f')
            figs = self.transformer.plot_bins(features=[f.rsplit('_WOE', maxsplit=1)[0] for f in self.features], folder=None,
                                              plot_flag=verbose)
            for i, fig in enumerate(figs):
                fig_to_excel(fig, ws,
                             row=len(scorecard) + len(self.features) + 9 + i * (20 if ds.time_column is None else 30),
                             col=0, scale=0.7)

        if PSI:
            if self.transformer is not None:
                legend_map = {f: {str(self.transformer.feature_woes[f.rsplit('_WOE', maxsplit=1)[0]].woes[
                                          g]): f"{v}. WOE {self.transformer.feature_woes[f.rsplit('_WOE', maxsplit=1)[0]].woes[g]}"
                                  for g, v in self.transformer.feature_woes[f.rsplit('_WOE', maxsplit=1)[0]].groups.items()} for f in self.features}
            else:
                legend_map = None
            result, figs = DataSamples(samples={'Train': ds.to_df(sample_field='sample')[self.features + [ds.target, 'sample']]},
                                       target=ds.target, features=self.features, cat_columns=[],
                                       time_column='sample').psi(legend_map=legend_map)
            result.to_excel(writer, sheet_name='PSI_samples')
            ws = writer.sheets['PSI_samples']
            ws.set_column(0, 0, 40)
            ws.set_column(1, len(result.columns) + 1, 12)
            for i, fig in enumerate(figs):
                fig_to_excel(fig, ws, row=i * (20 + 1) + len(self.features) + 3, col=0, scale=0.9)
            if ds.time_column:
                result, figs = ds.psi(features=self.features, legend_map=legend_map)
                result.to_excel(writer, sheet_name='PSI_time')
                ws = writer.sheets['PSI_time']
                ws.set_column(0, 0, 40)
                ws.set_column(1, len(result.columns) + 1, 12)
                for i, fig in enumerate(figs):
                    fig_to_excel(fig, ws, row=i * (20 + 1) + len(self.features) + 3, col=0, scale=0.9)
        if isinstance(out, str):
            writer.close()
    plt.close('all')

    @staticmethod
    def get_cv_gini(df, target, features, clf):
        try:
            return (2 * cross_val_score(clf, df[features], df[target], cv=5, scoring='roc_auc').mean() - 1) * 100
        except:
            return 0

    @staticmethod
    def get_time_gini(df, time_column, target, features, clf):
        if len(features) == 1:
            return {time: DataSamples.get_f_gini(group.drop([time_column], axis=1)) for time, group in df.groupby(time_column)}
        else:
            return {time: LogisticRegressionModel.get_cv_gini(group, target, features, clf)
                    for time, group in df.groupby(time_column)}

    def roc_curve(self, ds=None, verbose=True):
        """
        Рассчет джини модели на всех сэмплах и построение ROC-кривой
        :param ds: ДатаСэмпл
        :param verbose: флаг для вывода ROC-кривой в аутпут

        :return: кортеж (словарь {сэмпл: джини}, ROC-кривая в виде plt.fig)
        """
        if ds is None:
            ds = self.ds
        tpr = {}
        fpr = {}
        ginis = {}
        for name, sample in ds.samples.items():
            fpr[name], tpr[name], _ = roc_curve(sample[ds.target], self.predict_proba(ds, sample_name=name))
            ginis[name] = round((auc(fpr[name], tpr[name]) * 2 - 1) * 100, self.round_digits)
        if ds.bootstrap_base is not None:
            if ds.n_jobs > 1:
                with futures.ProcessPoolExecutor(max_workers=ds.n_jobs) as pool:
                    ginis_bootstrap = []
                    jobs = []
                    iterations = len(ds.bootstrap)
                    idx_iter = iter(ds.bootstrap)
                    while iterations:
                        for idx in idx_iter:
                            jobs.append(pool.submit(self.get_cv_gini,
                                                    df=ds.bootstrap_base.iloc[idx][[ds.target] + self.features],
                                                    target=ds.target, features=self.features, clf=self.clf))
                            if len(jobs) > ds.max_queue:
                                break
                        for job in futures.as_completed(jobs):
                            iterations -= 1
                            ginis_bootstrap.append(job.result())
                            jobs.remove(job)
                            break
                gc.collect()
            else:
                ginis_bootstrap = [self.get_cv_gini(df=ds.bootstrap_base.iloc[idx][[ds.target] + self.features],
                                                    target=ds.target, features=self.features, clf=self.clf)
                                   for idx in ds.bootstrap]
            ginis['Bootstrap mean'] = round(np.mean(ginis_bootstrap), self.round_digits)
            ginis['Bootstrap std'] = round(np.std(ginis_bootstrap), self.round_digits)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        # Plot tpr vs 1-fpr
        for name in fpr:
            ax.plot(fpr[name], tpr[name], label=f'{name} (Gini = {ginis[name]})')
        ax.plot(tpr[list(tpr)[0]],tpr[list(tpr)[0]], 'r')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        if verbose:
            plt.show()
        print(pd.DataFrame(ginis, index=['Gini']).to_string())
        return ginis, fig

    def calc_gini_in_time(self, ds=None, score_field=None):
        """
        Расчет динамики джини модели по срезам для всех сэмплов
        :param ds: ДатаСэмпл
        :param score_field: поле со скором модели. При None вызывается метод self.scoring() для расчета скора

        :return: ДатаФрейм с джини
        """
        if ds is None:
            ds = self.ds
        if ds.time_column is None:
            print('Please set time_column in DataSample for using this method.')
            return None
        if score_field is None:
            score_field = 'score'
            ds = self.scoring(ds, score_field=score_field)
        model_gini = {name: self.get_time_gini(sample[[ds.target, score_field, ds.time_column]], time_column=ds.time_column, target=ds.target, features=[score_field], clf=None)
                      for name, sample in ds.samples.items()}

        if ds.bootstrap_base is not None:
            if ds.n_jobs > 1:
                with futures.ProcessPoolExecutor(max_workers=ds.n_jobs) as pool:
                    ginis_bootstrap = []
                    jobs = []
                    iterations = len(ds.bootstrap)
                    idx_iter = iter(ds.bootstrap)
                    while iterations:
                        for idx in idx_iter:
                            jobs.append(pool.submit(self.get_time_gini,
                                                    df=ds.bootstrap_base.iloc[idx][[ds.target, ds.time_column] + self.features],
                                                    time_column=ds.time_column, target=ds.target,
                                                    features=self.features, clf=self.clf))
                            if len(jobs) > ds.max_queue:
                                break
                        for job in futures.as_completed(jobs):
                            iterations -= 1
                            ginis_bootstrap.append(job.result())
                            jobs.remove(job)
                            break
                gc.collect()
            else:
                ginis_bootstrap = [self.get_time_gini(df=ds.bootstrap_base.iloc[idx][[ds.target, ds.time_column] + self.features],
                                                      time_column=ds.time_column, target=ds.target, features=self.features,
                                                      clf=self.clf)
                                   for idx in ds.bootstrap]
            time_values = sorted(ds.bootstrap_base[ds.time_column].unique())
            model_gini['Bootstrap mean'] = {time: round(np.mean([ginis[time] for ginis in ginis_bootstrap if time in ginis]), self.round_digits) for time in time_values}
            model_gini['Bootstrap std'] = {time: round(np.std([ginis[time] for ginis in ginis_bootstrap if time in ginis]), self.round_digits) for time in time_values}

        time_values = sorted(list({time for name in model_gini for time in model_gini[name]}))
        return pd.DataFrame([[time] + [model_gini[name][time] if time in model_gini[name] else 0 for name in model_gini] for time in time_values],
                            columns=[ds.time_column] + [name for name in model_gini]).set_index(ds.time_column).abs().round(self.round_digits)

    def validate(self, ds=None, result_file='validation.xlsx', score_field='score', pd_field='pd', scale_field=None):
        """
        Валидационные тесты модели на заданном ДатаСэмпле библиотекой bankmetrics
        :param ds: ДатаСэмпл
        :param result_file: результирующий эксель файл
        :param score_field: поле со расчитанным скором (при отсутcnвии этого поля в выборке будет вызван метод self.scoring)
        :param pd_field: поле с расчитанным PD
        :param scale_field: поле с расчитанным грейдом
        """
        if ds is None:
            ds = self.ds
        if self.transformer is not None:
            ds = self.transformer.transform(ds)
        if (score_field and score_field not in ds.samples[ds.train_name].columns) or \
                (pd_field and pd_field not in ds.samples[ds.train_name].columns) or \
                (scale_field and scale_field not in ds.samples[ds.train_name].columns):
            ds = self.scoring(ds, score_field=score_field, pd_field=pd_field, scale_field=scale_field)

        df = ds.samples[ds.train_name]
        with pd.ExcelWriter(ds.result_folder + result_file, engine='openpyxl') as writer:
            test_to_excel(Iv_test(df, col=self.features, cel=ds.target), writer.book)
            test_to_excel(Ks_test(df, s0=score_field, cel=ds.target), writer.book)
            test_to_excel(Gini_test(df, col=self.features, cel=ds.target, unidir=False), writer.book)
            test_to_excel(Cal_test(df, pd0=pd_field, cel=ds.target), writer.book)
            test_to_excel(Corr_test(df, col=self.features), writer.book)
            test_to_excel(Vif_test(df, col=self.features, s0=''), writer.book)
            test_to_excel(Woe_test(df, col=self.features, cel=ds.target), writer.book, test_name='WOE')
            if scale_field:
                test_to_excel(Bin_test(df, m0=scale_field, pd0=pd_field, cel=ds.target), writer.book)
                test_to_excel(Chi2_test(df, odr=ds.target, m=scale_field, pd2=pd_field), writer.book)
                test_to_excel(Herfindal_test(df, m0=scale_field), writer.book)
            if len(ds.samples) > 1:
                test_name = [name for name in ds.samples if name != ds.train_name][0]
                test_to_excel(Gini_model_test(df2=df, df1=ds.samples[test_name], name2=ds.train_name, name1=test_name, col=self.features, cel=ds.target, unidir=False), writer.book, test_name='Gini_model')
                if scale_field:
                    test_to_excel(Psi_test(df20=df, df1=ds.samples[test_name], name2=ds.train_name, name1=test_name, col=self.features, m0=scale_field), writer.book)
        print(f'Results of validation tests was saved to file {ds.result_folder + result_file}')
        plt.close('all')

    def to_scale(self, PD):
        if self.scale is None:
            return np.nan
        for s in self.scale:
            if PD < self.scale[s]:
                return s
        return 'MSD'

    def calibration_test(self, df, target, pd_field, w=1):
        """
        Тест на калибровку
        :param df: ДатаФрейм
        :param target: поле с таргетом
        :param pd_field: поле с рассчитанным PD
        :param w: вес дефолтных наблюдений для теста

        :return: ДатаФрейм с результатом теста
        """

        def group_test(df, label):
            ci_95 = sts.norm.interval(0.95)[1]
            ci_99 = sts.norm.interval(0.99)[1]
            n = df.shape[0]
            d = df[target].sum()
            nw = n - d + d * w
            pd1_v = df[pd_field].sum() / n
            odr_v = d * w / nw
            k = (np.abs(pd1_v * (1 - pd1_v) / n)) ** 0.5
            return [label, pd1_v, odr_v, n, d, nw, d * w,
                    pd1_v - k * ci_99, pd1_v - k * ci_95,
                    pd1_v + k * ci_95, pd1_v + k * ci_99,
                    '' if n == 0 else
                    'З' if np.abs(odr_v - pd1_v) < k * ci_95 else
                    'К' if np.abs(odr_v - pd1_v) > k * ci_99 else 'Ж']

        df['scale_calibr'] = df[pd_field].apply(self.to_scale)
        rows = []
        for group in self.scale:
            rows.append(group_test(df[df['scale_calibr'] == group], group))
        rows.append(group_test(df, 'All'))
        res = pd.DataFrame.from_dict(self.scale, orient='index', columns=['upper PD']) \
            .merge(pd.DataFrame(rows, columns=['grade', 'Avg PD', 'Target rate', 'Amount', 'Target amount',
                                               'Amount weighted', 'Target amount weighted',
                                               '1%', '5%', '95%', '99%', 'Signal']).set_index('grade'),
                   left_index=True, right_index=True, how='right')
        res.loc[(res['Avg PD'] > res['Target rate']) & (res['Signal'] != 'З'), 'Signal'] += '*'
        return res

    def calibrate(self, CT, ds=None, method=0, sample_name=None, scale=None, score_field=None, result_file='calibration.xlsx',
                  plot_flag=True, fun=None, x0=None, args=None, lambda_ab=None):
        """
        Калибровка скора модели линейной функцией score_calibr = a + b*score. Результат сохраняется в self.calibration в виде списка [a, b]
        :param CT: значение центральной тенденции, к которому приводится среднее PD модели
        :param ds: ДатаСэмпл. При None берется self.ds
        :param method: метод калибровки. Доступны два варианта:
                        0 - Строится логрег скора на таргет, коэффициент b приравнивается полученному коэффициенту при скоре,
                            коэффицент a затем подбирается солвером для попадания в ЦТ при фиксированном b
                        1 - Расчитываются веса наблюдений и строится логрег скора на таргет с весами, a и b приравниваются коэффициентам логрега
                        2 - Коэффициенты рассчитываются минимизацией заданной функции через вызов scipy.optimize.minimize(fun=fun, x0=x0, args=args, method='nelder-mead')
                        любое другое значение - перерасчет коэффициентов не происходит, проводится тест на коэффициентах из self.calibration
        :param sample_name: название сэмпла, на котором проводится калибровка
        :param scale: шкала, на которой будет проведен биноминальный тест. Задается в виде словаря {грейд: верхняя граница PD грейда}. По умолчанию берется мастер-шкала
        :param score_field: поле со скором. Если оно отсутвует в ДатаСэмпле, то будет вызван метод self.scoring
        :param result_file: название эксель файла, в который будут записаны результаты
        :param plot_flag: флаг для вывода графика
        --- Метод калибровки 2 ---
        :param fun: пользовательская функция. Должна иметь вид
             def fun(params, **args):
                ...
                return result_to_minimize

             где params - список из подбираемых параметров. Например, [a], [b], [a, b]
                 result_to_minimize - результирующее значение, которое будет минимизироваться солвером
        :param x0: начальные значения параметров
        :param args: кортеж аргументов
        :param lambda_ab: функция для формирования списка [a, b] из результирующих параметров солвера. При None берется lambda x: x

        Примеры использования для калибровки с ограничением минимального PD модели значением minPD:

        Вариант 1) Минимизация функции двух переменных
            def CT_minPD_delta(params, score, CT, minPD):
                a, b = params
                pd = 1 / (1 + np.exp(-(a + score * b)))
                return (CT - pd.mean()) ** 2 + 10 * (minPD - pd.min())**2

            fun=CT_minPD_delta, x0=[0, 1], args=(ds.samples[ds.train_name]['score'], CT, minPD), lambda_ab=None

        Вариант 2) Минимизация функции одной переменной, вычисление коэффициента b через связь minPD и лучшего скора по выборке
            def CT_delta(params, score, CT, minPD, best_score):
                a = params
                b = (-log(1 / minPD - 1) - a) / best_score
                pd = 1 / (1 + np.exp(-(a + score * b)))
                return (CT - pd.mean()) ** 2

            best_score = ds.samples[ds.train_name]['score'].min()
            fun=CT_delta, x0=0, args=(ds.samples[ds.train_name]['score'], CT, minPD, best_score), lambda_ab=lambda x: (x[0], (-log(1 / minPD - 1) - x[0]) / best_score))

        :return: коэффициенты калибровки [a, b]
        """
        if ds is None:
            ds = self.ds
        if sample_name is None:
            sample_name = ds.train_name
        df = ds.samples[sample_name].copy()
        if not score_field:
            score_field = 'score_tmp'
            df = self.scoring(df, score_field=score_field, pd_field=None)

        if scale is not None:
            self.scale = scale
        elif self.scale is None:
            self.scale = {
                        'MA1': 0.0005,
                        'MA2': 0.000695,
                        'MA3': 0.000976,
                        'MB1': 0.001372,
                        'MB2': 0.001927,
                        'MB3': 0.002708,
                        'MC1': 0.003804,
                        'MC2': 0.005344,
                        'MC3': 0.007508,
                        'MD1': 0.010549,
                        'MD2': 0.014821,
                        'MD3': 0.020822,
                        'ME1': 0.029254,
                        'ME2': 0.041101,
                        'ME3': 0.057744,
                        'MF1': 0.081128,
                        'MF2': 0.11398,
                        'MF3': 0.160137,
                        'MG1': 0.224984,
                        'MG2': 0.31609,
                        'MG3': 1
                       }

        def CT_meanPD_b_fix(params, score, CT, b):
            a = params
            pd = 1 / (1 + np.exp(-(a + score * b)))
            return (CT - pd.mean()) ** 2
        # расчет веса дефолтных наблюдений

        n = df.shape[0]
        d = df[ds.target].sum()
        w = CT * (n - d) / ((1 - CT) * d)
        lr = copy.deepcopy(self.clf)
        if method == 0:
            lr.fit(df[[score_field]], df[ds.target])
            b = lr.coef_[0][0]
            res = minimize(fun=CT_meanPD_b_fix, x0=0, args=(df[score_field], CT, b), method='nelder-mead')
            self.calibration = [round(res['x'][0], self.round_digits + 2), round(b, self.round_digits + 2)]
        elif method == 1:
            lr.fit(df[[score_field]], df[ds.target], sample_weight=np.where(df[ds.target] == 1, w, 1))
            self.calibration = [round(lr.intercept_[0], self.round_digits + 1), round(lr.coef_[0][0], self.round_digits + 1)]
        elif method == 2:
            res = minimize(fun=fun, x0=x0, args=args, method='nelder-mead')
            if lambda_ab is None:
                lambda_ab = lambda x: x
            a, b = lambda_ab(res['x'])
            self.calibration = [round(a, self.round_digits + 2), round(b, self.round_digits + 2)]
        else:
            print('Incorrect method. Using for calibration existing coefficients in self.calibration')

        if result_file and self.scale:
            with pd.ExcelWriter(ds.result_folder + result_file, engine='xlsxwriter') as writer:
                # расчет откалиброванного скора и PD
                df['score_calibr'] = self.calibration[0] + df[score_field] * self.calibration[1]
                df['PD_calibr'] = 1 / (1 + np.exp(-df['score_calibr']))

                # тест на калибровку для каждого грейда и всей выборки целиком
                res = self.calibration_test(df, ds.target, 'PD_calibr', w)
                res.style.applymap(lambda x: 'color: black' if not isinstance(x, str) or not x else
                                            'color: red' if x[0] == 'К' else
                                            'color: orange' if x[0] == 'Ж' else
                                            'color: green' if x[0] == 'З' else
                                            'color: black').to_excel(writer, sheet_name='MS', float_format=f'%0.{self.round_digits + 1}f')
                gini = DataSamples.get_f_gini(df[[ds.target, score_field]])
                pd.DataFrame(
                    [['CT', CT],
                     ['a', self.calibration[0]], ['b', self.calibration[1]],
                     ['вес дефолтных наблюдений', w], ['вес недефолтных наблюдений', 1],
                     ['min PD', df['PD_calibr'].min()],
                     ['Gini', gini],
                     ],
                    columns=['Key', 'Value']).to_excel(writer, sheet_name='MS', startrow=len(self.scale) + 4,
                                                       startcol=3, index=False, float_format='%0.5f')

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                res = res[res.index != 'All']
                ax.set_ylabel('Observations')
                ax.set_xticks(range(res.shape[0]))
                ax.set_xticklabels(res.index, rotation=30, ha="right")
                amnt = res['Amount'].sum()
                ax.bar(range(res.shape[0]), (res['Amount'] - res['Target amount']) / amnt, zorder=0, color='forestgreen', label='Class 0')
                ax.bar(range(res.shape[0]), res['Target amount'] / amnt,  bottom=(res['Amount'] - res['Target amount']) / amnt, zorder=0, color='indianred', label='Class 1')
                ax.annotate('Amount:', xy=(-0.5, 1), xycoords=('data', 'axes fraction'), xytext=(0, 60), textcoords='offset pixels', color='black', size=10, ha='right')
                for i in range(res.shape[0]):
                    ax.annotate(str(res['Amount'][i]), xy=(i, 1), xycoords=('data', 'axes fraction'),
                                  xytext=(0, 60), textcoords='offset pixels', color='black', size=10, ha='center')
                ax.annotate('Target amount:', xy=(-0.5, 1), xycoords=('data', 'axes fraction'),
                            xytext=(0, 40), textcoords='offset pixels', color='black', size=10, ha='right')
                for i in range(res.shape[0]):
                    ax.annotate(str(round(res['Target amount'][i])), xy=(i, 1), xycoords=('data', 'axes fraction'),
                                xytext=(0, 40), textcoords='offset pixels', color='black', size=10, ha='center')
                ax.annotate('Signal:', xy=(-0.5, 1), xycoords=('data', 'axes fraction'),
                            xytext=(0, 20), textcoords='offset pixels', color='black', size=10, ha='right')
                for i, S in enumerate(res['Signal'].values):
                    ax.annotate(S, xy=(i, 1), xycoords=('data', 'axes fraction'),
                                xytext=(0, 20), textcoords='offset pixels',
                                color='red' if not S or S[0] == 'К' else
                                      'orange' if S[0] == 'Ж' else 'green',
                                size=10, ha='center')
                ax.grid(False)
                ax.grid(axis='y', zorder=1)
                ax2 = ax.twinx()
                ax2.set_ylabel('Target rate')
                ax2.grid(False)
                ax2.plot(range(res.shape[0]), res['Target rate'], 'bo-', linewidth=2.0, zorder=4, label='Target rate', color='black')
                ax2.fill_between(range(res.shape[0]), res['5%'], res['95%'],  alpha=0.1,  color='blue', label='95% conf interval')
                ax2.plot(range(res.shape[0]), res['Avg PD'], 'bo-', linewidth=2.0, zorder=4, label='Avg PD', color='blue')

                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=10)
                fig.tight_layout()
                fig_to_excel(fig, writer.sheets['MS'], row=0, col=len(res.columns) + 3)
                if plot_flag:
                    plt.show()
                plt.close('all')
            return self.calibration

    def predict_proba(self, ds=None, sample_name=None):
        """
        Вычисление вероятности целевого события
        :param ds: ДатаСэмпл. При None берется self.ds
        :param sample_name: название сэмпла для вычисление. При None берется ds.train_sample

        :return: np.array вероятностей
        """
        if ds is None:
            ds = self.ds
        if sample_name is None:
            sample_name = ds.train_name
        return 1 / (1 + np.exp(-(self.intercept + np.dot(ds.samples[sample_name][list(self.coefs.keys())], list(self.coefs.values())))))

    def wald_test(self, ds=None, sample_name=None, clf=None, features=None, fit=False):
        """
        Тест Вальда. Вычисление стандартной ошибки, Wald Chi-Square и p-value для всех коэффициентов модели на заданном ДатаСэмпле
        :param ds: ДатаСэмпл. При None берется self.ds
        :param sample_name: название сэмпла. При None берется ds.train_sample
        :param clf: классификатор модели. При None берется self.clf
        :param features: список переменных. При None берется self.features
        :param fit: флаг для обучения модели заново на текущих данных

        :return: дафрейм с полями 'feature', 'coefficient', 'se', 'wald', 'p-value'
        """
        if ds is None:
            ds = self.ds
        if features is None:
            features = self.features
        if sample_name is None:
            sample_name = ds.train_name
        if clf is None:
            coefs_list = [self.intercept] + list(self.coefs.values())
            predProbs = np.matrix([[1 - x, x] for x in self.predict_proba(ds, sample_name=sample_name)])
        else:
            if fit:
                clf.fit(ds.samples[sample_name][features], ds.samples[sample_name][ds.target])
            coefs_list = clf.intercept_.tolist() + clf.coef_[0].tolist()
            predProbs = np.matrix(clf.predict_proba(ds.samples[sample_name][features]))
        coefs_list_to_check = [c for i, c in enumerate(coefs_list) if i == 0 or c != 0]
        features_to_check = [f for i, f in enumerate(features) if coefs_list[i+1] != 0]

        # Design matrix -- add column of 1's at the beginning of your X_train matrix
        X_design = np.hstack((np.ones(shape = (ds.samples[sample_name][features_to_check].shape[0],1)),
                              ds.samples[sample_name][features_to_check]))

        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        V=np.multiply(predProbs[:,0], predProbs[:,1]).A1
        # Covariance matrix
        covLogit = np.linalg.inv(np.matrix(X_design.T * V) * X_design)
        # Output
        bse=np.sqrt(np.diag(covLogit))
        wald=(coefs_list_to_check / bse) ** 2
        pvalue=chi2.sf(wald, 1)
        return pd.DataFrame({'feature': ['intercept'] + features, 'coefficient': coefs_list})\
            .merge(pd.DataFrame({'feature': ['intercept'] + features_to_check, 'se': bse, 'wald': wald, 'p-value': pvalue}),
                   on='feature', how='left')

    def regularized_selection(self, ds=None, features=None, hold=None, scoring='gini',
                              cv=None, pvalue_threshold=0.05, verbose=False):
        """
        Отбор факторов на основе регуляризации - строится модель на всех переменных, затем итерационно исключаются
        переменные с нулевыми или положительными коэффициентами и низкой значимостью
        :param ds: ДатаСэмпл. При None берется self.ds
        :param features: исходный список переменных. При None берется self.features
        :param hold: список переменных, которые обязательно должны остаться после отбора
        :param scoring: расчитываемый скор модели
        :param pvalue_threshold: граница значимости по p-value
        :param verbose: флаг для вывода подробных комментариев в процессе работы

        :return: кортеж (итоговый список переменных, график со скором в процессе отбора в виде объекта plt.figure)
        """
        if ds is not None:
            self.ds = ds
        if hold is None:
            hold = []
        if features is None:
            features = self.features.copy()

        sample_name = self.ds.train_name
        # correctness check
        for feature in hold:
            if feature not in self.ds.features:
                print ('Feature is not available:', feature)
                return None

        if not self.ds.ginis:
            self.ds.calc_gini()
        ginis = self.ds.ginis[sample_name]
        scores = []
        features_change = ['Initial']
        to_refit = True
        if verbose:
            print(f'Dropping features with positive coefs and high p-values...')
        while to_refit:
            to_refit = False
            self.clf.fit(self.ds.samples[sample_name][features], self.ds.samples[sample_name][self.ds.target])
            new_score = self.get_score(scoring=scoring, cv=cv, features=features, fit=False)
            scores.append(new_score)
            positive_to_exclude = {features[i]:self.clf.coef_[0][i] for i in range(len(features)) if self.clf.coef_[0][i] > 0 if features[i] not in hold}
            if positive_to_exclude:
                to_refit = True
                features_to_exclude = {x:ginis[x] for x in positive_to_exclude}
                to_exclude = min(features_to_exclude, key=features_to_exclude.get)
                if verbose:
                    print(f'To drop: {to_exclude}, gini: {round(ginis[to_exclude], self.round_digits)}, coef: {positive_to_exclude[to_exclude]}')
                features.remove(to_exclude)
                features_change.append(f'- {to_exclude}')
            else:
                wald = self.wald_test(clf=self.clf, features=features)
                feature_to_exclude_array = wald[(wald['p-value']>pvalue_threshold) & (wald['p-value']==wald['p-value'].max()) & (wald['feature'].isin(hold + ['intercept']) == False)]['feature'].values
                if feature_to_exclude_array:
                    to_refit = True
                    if verbose:
                        print(f'To drop: {feature_to_exclude_array[0]}, gini: {round(ginis[feature_to_exclude_array[0]], self.round_digits)}, p-value: {wald[wald["feature"]==feature_to_exclude_array[0]]["p-value"].values[0]}')
                    features.remove(feature_to_exclude_array[0])
                    features_change.append(f'- {feature_to_exclude_array[0]}')

        result_features = []
        for i in range(len(self.clf.coef_[0])):
            if self.clf.coef_[0][i] == 0:
                if verbose:
                    print(f'To drop: {features[i]}, gini: {round(ginis[features[i]], self.round_digits)}, coef: 0')
            else:
                result_features.append(features[i])
        if len(features_change) > 1:
            fig = plt.figure(figsize=(max(len(features_change)//2, 5), 3))
            plt.plot(np.arange(len(scores)), scores, 'bo-', linewidth=2.0)
            plt.xticks(np.arange(len(features_change)), features_change, rotation=30, ha='right', fontsize=10)
            plt.tick_params(axis='y', which='both', labelsize=10)
            plt.ylabel(scoring)
            plt.title('Dropping features with positive coefs and high p-values')
            fig.tight_layout()
            if verbose:
                plt.show()
        else:
            fig = None
            if verbose:
                print(f'Nothing to drop')
        return features, fig

    @staticmethod
    def add_feature_stat(df, clf, scoring, cv):
        features = list(df.columns)[1:]
        clf.fit(df[features], df.iloc[:, 0])
        features_to_check = []
        coefs_list = [clf.intercept_[0]]
        for i, feature in enumerate(features):
            if clf.coef_[0][i] != 0:
                features_to_check.append(feature)
                coefs_list.append(clf.coef_[0][i])
            elif i == 0:
                return 1, 0
        try:
            # Calculate matrix of predicted class probabilities.
            # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
            predProbs = np.matrix(clf.predict_proba(df[features]))

            # Design matrix -- add column of 1's at the beginning of your X_train matrix
            X_design = np.hstack((np.ones(shape = (df[features_to_check].shape[0],1)),
                                  df[features_to_check]))
            V=np.multiply(predProbs[:,0], predProbs[:,1]).A1
            covLogit = np.linalg.inv(np.matrix(X_design.T * V) * X_design)
            bse=np.sqrt(np.diag(covLogit))
            wald=(coefs_list / bse) ** 2
            pvalue=chi2.sf(wald, 1)
            if scoring.upper() in ['AIC', 'BIC', 'SIC',  'SBIC']:
                features_kept = []
                weights_crit = [clf.intercept_[0]]
                for i in range(len(features)):
                    if clf.coef_[0][i] != 0:
                        features_kept.append(features[i])
                        weights_crit.append(clf.coef_[0][i])

                intercept_crit = np.ones((df.shape[0], 1))
                features_crit = np.hstack((intercept_crit, df[features_kept]))
                scores_crit = np.dot(features_crit, weights_crit)
                ll = np.sum((df.iloc[:, 0] * scores_crit - np.log(np.exp(scores_crit) + 1)))

                if scoring.upper() == 'AIC':
                    score = 2 * len(weights_crit) - 2 * ll
                else:
                    score = len(weights_crit) * np.log(df.shape[0]) - 2 * ll
                score = -score
            else:
                score = cross_val_score(clf, df[features], df.iloc[:, 0], cv=cv,
                                        scoring=scoring if scoring != 'gini' else 'roc_auc').mean()
                if scoring == 'gini':
                    score = (2 * score - 1)*100
        except:
            return 1, 0
        return pvalue[1], score

    def stepwise_selection(self, ds=None, verbose=False, selection_type='stepwise', features=None, hold=None,
                           score_delta=0.01, scoring='gini', cv=None, pvalue_threshold=0.05, pvalue_priority=False):
        """
         Итерационный отобор. Доступны три типа отбора:
            selection_type='forward' - все доступные факторы помещаются в список кандидатов, на каждом шаге из списка кандидатов определяется лучший* фактор и перемещается в модель
            selection_type='backward' - в модель включаются все доступные факторы, затем на каждом шаге исключается худший* фактор
            selection_type='stepwise' - комбинация 'forward' и 'backward'. Каждый шаг состоит из двух этапов:
                    на первом из списка кандидатов отбирается лучший* фактор в модель,
                    на втором из уже включенных факторов выбирается худший* и исключается

            *Определение лучшего фактора:
            При pvalue_priority=False лучшим фактором считается тот, который увеличивает метрику scoring модели на наибольшую величину.
                Если величина такого увеличения ниже score_delta, то лучший фактора не определяется, и добавления не происходит
            При pvalue_priority=True лучшим фактором считается фактор, который после добавления в модель имеет наименьшее p-value.
                Если величина этого p-value выше pvalue_threshold, то лучший фактора не определяется, и добавления не происходит

            *Определение худшего фактора:
            Худшим фактором в модели считается фактор с наибольшим p-value.
                Если величина этого p-value ниже pvalue_threshold, то худший фактора не определяется, и исключения не происходит
        :param ds: ДатаСэмпл. При None берется self.ds
        :param verbose: флаг для вывода подробных комментариев в процессе работы
        :param selection_type: тип отбора. Варианты 'forward', 'backward', 'stepwise'
        :param features: исходный список переменных. При None берется self.features
        :param hold: список переменных, которые обязательно должны остаться после отбора
        :param score_delta: минимальный прирост метрики
        :param scoring: максимизируемая метрика.
                Варианты значений: 'gini', 'AIC', 'BIC', 'SIC', 'SBIC' + все метрики доступные для вычисления через sklearn.model_selection.cross_val_score.
                Все информационные метрики после вычисления умножаются на -1 для сохранения логики максимизации метрики.
        :param pvalue_threshold: граница значимости по p-value
        :param pvalue_priority: вариант определения лучшего фактора

        :return: кортеж (итоговый список переменных, график со скором в процессе отбора в виде объекта plt.figure)
        """
        def add_feature(features, candidates, clf):
            cvs = {}
            pvalues = {}
            candidates = [f for f in candidates if f not in features]
            if self.ds.n_jobs > 1:
                jobs = {}
                iterations = len(candidates)
                candidates_iter = iter(candidates)
                with futures.ProcessPoolExecutor(max_workers=self.ds.n_jobs) as pool:
                    while iterations:
                        for feature in candidates_iter:
                            jobs[pool.submit(self.add_feature_stat, df=self.ds.samples[self.ds.train_name][[self.ds.target, feature] + features], clf=clf, scoring=scoring, cv=cv)] = feature
                            if len(jobs) > self.ds.max_queue:
                                break
                        for job in futures.as_completed(jobs):
                            iterations -= 1
                            feature = jobs[job]
                            pvalues[feature], cvs[feature] = job.result()
                            del jobs[job]
                            break
            else:
                for feature in candidates:
                    tmp_features = features + [feature]
                    clf.fit(self.ds.samples[self.ds.train_name][tmp_features], self.ds.samples[self.ds.train_name][self.ds.target])
                    cvs[feature] = self.get_score(scoring=scoring, cv=cv, features=tmp_features, clf=clf, fit=False)
                    try:
                        wald = self.wald_test(features=tmp_features, clf=clf, fit=False)
                        pvalues[feature] = wald[wald['feature'] == feature]['p-value'].values[0]
                    except:
                        pvalues[feature] = 1
            if pvalues:
                if pvalue_priority:
                    feature = min(pvalues, key=pvalues.get)
                    if pvalues[feature] > pvalue_threshold:
                        feature = None
                else:
                    features = [f for f in sorted(cvs, key=cvs.get, reverse=True) if
                                pvalue_threshold is None or pvalues[f] < pvalue_threshold]
                    if features and cvs[features[0]] - prev_score > score_delta:
                        feature = features[0]
                    else:
                        feature = None
                if feature:
                    return feature, round(cvs[feature], self.round_digits), pvalues[feature]
            return None, None, None

        def drop_feature(features, clf):
            # searching for the feature that increases the quality most
            wald = self.wald_test(clf=clf, features=features, fit=True)
            wald_to_check = wald[~wald['feature'].isin(hold + ['intercept'])]
            pvalue = wald_to_check['p-value'].max()
            feature = wald_to_check[wald_to_check['p-value'] == pvalue]['feature'].values[0]
            if pvalue < pvalue_threshold:
                feature = None
                score = None
                pvalue = None
            else:
                score = self.get_score(scoring=scoring, cv=cv, features=[f for f in features if f != feature], clf=clf, fit=True)
            return feature, score, pvalue

        if ds is not None:
            self.ds = ds
        if hold is None:
            hold = []
        if features is None:
            features = self.ds.features.copy()
        scores = []
        features_change = []
        # correctness check
        for feature in hold:
            if feature not in self.ds.features:
                print(f'No {feature} in DataSample!')
                return None

        if verbose:
            print(f'{selection_type.capitalize()} feature selection started...')
        # Forward selection
        if selection_type == 'forward':
            candidates = [feature for feature in features if feature not in hold]
            features = hold.copy()
            if len(features) > 0:
                prev_score = self.get_score(scoring=scoring, cv=cv, features=features, fit=True)
                scores.append(prev_score)
                features_change.append('initial')
                if verbose:
                    print('Initial features:', features, '', scoring, 'score', prev_score)
            else:
                prev_score = -1000000
            # maximum number of steps equals to the number of candidates
            for i in range(len(candidates)):
                # cross-validation scores for each of the remaining candidates of the step
                feature, score, pvalue = add_feature(features, candidates, self.clf)
                if feature is None:
                    break
                if verbose:
                    print(f'To add: {feature}, {scoring}: {score}, p-value: {pvalue}')
                prev_score = score
                features.append(feature)
                candidates.remove(feature)
                scores.append(score)
                features_change.append(f'+ {feature}')

        # Backward selection
        elif selection_type == 'backward':
            candidates = [feature for feature in features if feature not in hold]
            if len(features) > 0:
                prev_score = self.get_score(scoring=scoring, cv=cv, features=features, fit=True)
                scores.append(prev_score)
                features_change.append('initial')
                if verbose:
                    print('Initial features:', features, '', scoring, 'score', prev_score)
            else:
                prev_score = -1000000
            # maximum number of steps equals to the number of candidates
            for i in range(len(candidates)):
                feature, score, pvalue = drop_feature(features, self.clf)
                if feature is None:
                    break
                if verbose:
                    print(f'To drop: {feature}, {scoring}: {score}, p-value: {pvalue}')
                prev_score = score
                features.remove(feature)
                scores.append(score)
                features_change.append(f'- {feature}')
        # stepwise
        elif selection_type == 'stepwise':
            candidates = [feature for feature in features if feature not in hold]
            features = hold.copy()
            if len(features) > 0:
                score = self.get_score(scoring=scoring, cv=cv, features=features, fit=True)
                scores.append(score)
                features_change.append('initial')
                feature_sets = [set(features)]
            else:
                score = -1000000
                feature_sets = []
            to_continue = True
            prev_score = score
            while to_continue and candidates:
                to_continue = False
                result_features = features.copy()
                feature, score, pvalue = add_feature(features, candidates, self.clf)
                if feature:
                    if verbose:
                        print(f'To add: {feature}, {scoring}: {score}, p-value: {pvalue}')
                    features.append(feature)
                    scores.append(score)
                    prev_score = score
                    features_change.append(f'+ {feature}')
                    if set(features) in feature_sets:
                        if verbose:
                            print('Feature selection entered loop: terminating feature selection')
                        break
                    to_continue = True
                    feature_sets.append(set(features))
                if features == result_features:
                    if verbose:
                        print('No significant features to add were found')
                # the least significant feature is removed
                # if it is Step1 then no removal
                if len(features) > 1:
                    feature, score, pvalue = drop_feature(features, self.clf)
                    if feature:
                        if verbose:
                            print(f'To drop: {feature}, {scoring}: {score}, p-value: {pvalue}')
                        scores.append(score)
                        prev_score = score
                        features.remove(feature)
                        features_change.append(f'- {feature}')
                        if set(features) in feature_sets:
                            if verbose:
                                print('Feature selection entered loop: terminating feature selection')
                            break
                        to_continue = True
                        feature_sets.append(set(features))
            features = sorted(list(feature_sets[-1]))
        else:
            print('Incorrect kind of selection. Please use backward, forward or stepwise.')
            return None
        fig = plt.figure(figsize=(max(len(features_change)//2, 5), 3))
        plt.plot(np.arange(len(scores)), scores, 'bo-', linewidth=2.0)
        plt.xticks(np.arange(len(features_change)), features_change, rotation=30, ha='right', fontsize=10)
        plt.tick_params(axis='y', which='both', labelsize=10)
        plt.ylabel(scoring)
        plt.title(f'{selection_type.capitalize()} score changes')
        fig.tight_layout()
        if verbose:
            plt.show()
        return features, fig

    def tree_selection(self, ds=None, selection_type='forward', model_type='xgboost', result_file='tree_selection.xlsx',
                       plot_pdp=False, verbose=False):
        """
        Выполняет отбо факторов методом autobinary.AutoSelection. На отоборанных факторах строится модель на бустинге и логреге.
        Если self.transformer не задан, то для отобранных факторов дополнительно делается автобиннинг
        :param ds: ДатаСэмпл с нетрансформированными переменными. При None берется self.ds
        :param selection_type: тип отбора. Варианты 'forward', 'backward', 'deep_backward'
        :param model_type: тип модели для классификации. Варианты 'xgboost', 'lightboost', 'catboost'
        :param result_file: файл, в который будут сохраняться результаты
        :param plot_pdp: флаг для построение графиков PDP. Нужна библиотека pdpbox
        :param verbose: флаг вывода комментариев в процессе работы
        """
        from ..autobinary.auto_permutation import PermutationSelection
        from ..autobinary.auto_selection import AutoSelection
        from ..autobinary.auto_trees import AutoTrees
        from ..autobinary.base_pipe import base_pipe
        if plot_pdp:
            try:
                from ..autobinary.auto_pdp import PlotPDP
            except Exception as e:
                print(e)
                plot_pdp = False
        if ds is None:
            ds = self.ds
        prep_pipe = base_pipe(
            num_columns=[f for f in ds.features if f not in ds.cat_columns],
            cat_columns=ds.cat_columns,
            kind='all')

        if model_type == 'xgboost':
            import xgboost
            params = {'eta': 0.01,
                      'n_estimators': 500,
                      'subsample': 0.9,
                      'max_depth': 6,
                      'objective': 'binary:logistic',
                      'n_jobs': ds.n_jobs,
                      'random_state': ds.random_state,
                      'eval_metric': 'logloss'}
            clf = xgboost.XGBClassifier(**params)
            fit_params = {
                'early_stopping_rounds': 100,
                'eval_metric': ['logloss', 'aucpr', 'auc'],
                'verbose': 25}
        elif model_type == 'lightboost':
            import lightgbm
            params = {'learning_rate': 0.01,
                      'n_estimators': 500,
                      'subsample': 0.9,
                      'max_depth': 6,
                      'objective': 'binary',
                      'metric': 'binary_logloss',
                      'n_jobs': ds.n_jobs,
                      'random_state': ds.random_state,
                      'verbose': -1}
            clf = lightgbm.LGBMClassifier(**params)
            fit_params = {
                'early_stopping_rounds': 100,
                'eval_metric': ['logloss', 'auc'],
                'verbose': -1}
        elif model_type == 'catboost':
            import catboost
            params = {'learning_rate': 0.01,
                      'iterations': 500,
                      'subsample': 0.9,
                      'depth': 6,
                      'loss_function': 'Logloss',
                      'thread_count': ds.n_jobs,
                      'random_state': ds.random_state,
                      'verbose': 0}
            clf = catb.CatBoostClassifier(**params)

            fit_params = {
                'use_best_model': True,
                'early_stopping_rounds': 200,
                'verbose': 50,
                'plot': False}
        else:
            print('Wrong model_type!')
            return None

        X_train = ds.samples[ds.train_name][ds.features]
        y_train = ds.samples[ds.train_name][ds.target]
        prep_pipe = base_pipe(
            num_columns=[f for f in ds.features if f not in ds.cat_columns],
            cat_columns=ds.cat_columns,
            kind='all')
        prep_pipe.fit(X_train, y_train)
        new_X_train = prep_pipe.transform(X_train)

        perm_imp = PermutationSelection(
            model_type=model_type,
            model_params=params,
            task_type='classification')
        fi, fi_rank, depth_features, rank_features = perm_imp.depth_analysis(new_X_train, y_train, list(new_X_train.columns), 5)
        # задаем стратегию проверки
        strat = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=ds.random_state)

        selection = AutoSelection(
            base_pipe=base_pipe,
            num_columns=[f for f in depth_features if f not in ds.cat_columns],
            cat_columns=[f for f in depth_features if f in ds.cat_columns],
            main_fit_params=fit_params,
            main_estimator=clf,
            X_train=X_train[depth_features],
            y_train=y_train,
            main_metric='gini',
            model_type=model_type)
        if selection_type == 'forward':
            selection_res = selection.forward_selection(strat=strat)
        elif selection_type == 'backward':
            selection_res = selection.backward_selection(strat=strat, first_degradation=True)
        elif selection_type == 'deep_backward':
            selection_res = selection.deep_backward_selection(strat=strat, tol=0.001)
        else:
            print('Wrong selection_type!')
            return None
        features = selection_res['features_stack']

        model = AutoTrees(
            main_estimator=clf,
            main_fit_params=fit_params,
            main_prep_pipe=prep_pipe,
            main_features=features,
            X_train=X_train[features],
            y_train=y_train,
            main_metric='gini',
            model_type=model_type)

        model.model_fit_cv(strat=strat)

        if plot_pdp:
            clf.fit(new_X_train, y_train);
            pdp_plot = PlotPDP(model=clf, X=new_X_train[features], main_features=features)
            pdp_plot.create_feature_plot(save=True, path=ds.result_folder + 'pdp_ice_plots', frac_to_plot=0.05)

        with pd.ExcelWriter(ds.result_folder + result_file, engine='xlsxwriter') as writer:
            samples = {}
            for name, sample in ds.samples.items():
                samples[name] = prep_pipe.transform(sample[features])
                samples[name][ds.target] = sample[ds.target].reset_index(drop=True)
            ds_tmp = DataSamples(samples=samples, target=ds.target, features=features, cat_columns=[], n_jobs=1)
            if ds.bootstrap_base is not None:
                bootstrap_base = prep_pipe.transform(ds.bootstrap_base[features])
                bootstrap_base[ds.target] = ds.bootstrap_base[ds.target].reset_index(drop=True)
                ds_tmp.bootstrap_base = bootstrap_base
                ds_tmp.bootstrap = ds.bootstrap
            gini_df = ds_tmp.calc_gini(add_description=True)

            gini_df.to_excel(writer, sheet_name=model_type, startrow=1, startcol=0)
            ws = writer.sheets[model_type]
            adjust_cell_width(ws, gini_df)
            ws.write(0, 0, 'Features in model:')
            ws.write(0, len(gini_df.columns) + 3, 'Metrics:')
            model.get_extra_scores().round(self.round_digits).\
                to_excel(writer, sheet_name=model_type, startrow=1, startcol=len(gini_df.columns) + 3, index=False)
            if plot_pdp:
                for i, feature in enumerate(features):
                    ws.insert_image(len(features) + 5 + i * 28, 0, f'{ds.result_folder}pdp_ice_plots/PDP_{feature}.png',
                                    {'x_scale': 0.75, 'y_scale': 0.75})
            fig_to_excel(model.get_rocauc_plots(), ws, row=0,  col=len(gini_df.columns) + 10)

            if self.transformer is None:
                self.transformer = WOE(ds=ds, features=features)
                self.transformer.auto_fit(plot_flag=False, SM_on=False, BL_allow_Vlogic_to_increase_gini=20, G_on=False)
            ds = self.transformer.transform(ds, features=features, verbose=verbose)
            self.fit(ds, features=ds.features)
            self.report(ds, out=writer, sheet_name='logreg', pvalue_threshold=0.05, PSI=False, verbose=verbose)

    def fit(self, ds=None, sample_name=None, features=None):
        """
        Обучение модели
        :param ds: ДатаСэмпл. При None берется self.ds
        :param sample_name: название сэмпла на котором проводится обучение. При None берется ds.train_sample
        :param features: список переменных. При None берется self.features
        """
        if ds is None:
            ds = self.ds
        self.coefs = {}
        self.intercept = None
        if features:
            self.features = features
        elif not self.features:
            self.features = ds.features

        if sample_name is None:
            sample_name = ds.train_name

        try:
            self.clf.fit(ds.samples[sample_name][self.features], ds.samples[sample_name][ds.target])
        except Exception as e:
            print(e)
            print('Fit failed!')
            return None
        self.intercept = round(self.clf.intercept_[0], self.round_digits)
        for i in range(len(self.features)):
            self.coefs[self.features[i]] = round(self.clf.coef_[0][i], self.round_digits)
        print(f'intercept = {self.intercept}')
        print(f'coefs = {self.coefs}')

    def inf_criterion(self, ds=None, sample_name=None, clf=None, features=None, criterion='AIC'):
        """
        Расчет информационного критерия модели на заданной выборке
        :param ds: ДатаСэмпл. При None берется self.ds
        :param sample_name: название сэмпла. При None берется ds.train_sample
        :param clf: модель (объект класса LogisticRegression). При None берется self.model
        :param features: список переменных. При None берется self.features
        :param criterion: критерий для расчета. Доступны варианты 'AIC', 'BIC'
        :return: значение заданного криетрия
        """
        if ds is None:
            ds = self.ds
        if features is None:
            if self.features:
                features = self.features
            else:
                features = ds.features

        if clf is None:
            clf = self.clf

        if sample_name is None:
            sample_name = ds.train_name

        features_kept = []
        weights_crit = [clf.intercept_[0]]
        for i in range(len(features)):
            if clf.coef_[0][i] != 0:
                features_kept.append(features[i])
                weights_crit.append(clf.coef_[0][i])

        intercept_crit = np.ones((ds.samples[sample_name].shape[0], 1))
        features_crit = np.hstack((intercept_crit, ds.samples[sample_name][features_kept]))
        scores_crit = np.dot(features_crit, weights_crit)
        ll = np.sum(ds.samples[sample_name][ds.target] * scores_crit - np.log(np.exp(scores_crit) + 1))
        if criterion in ['aic', 'AIC']:
            return 2 * len(weights_crit) - 2 * ll
        elif criterion in ['bic', 'BIC', 'sic', 'SIC', 'sbic', 'SBIC']:
            return len(weights_crit) * np.log(ds.samples[sample_name].shape[0]) - 2 * ll

    def get_score(self, ds=None, sample_name=None, clf=None, cv=None, scoring='gini', features=None, fit=True):
        """
        Вычисление скора модели на заданной выборке
        :param ds: ДатаСэмпл. При None берется self.ds
        :param sample_name: название сэмпла. При None берется ds.train_sample
        :param clf: классификатор модели. При None берется self.clf
        :param cv: параметр cv для вычисления скора sklearn.model_selection.cross_val_score
        :param scoring: рассчитываемый скор. Варианты значений: 'gini', 'AIC', 'BIC' + все метрики доступные для вычисления через sklearn.model_selection.cross_val_score
        :param features: список переменных. При None берется self.features
        :param fit: флаг для обучения модели заново на текущих данных

        :return: рассчитанный скор
        """
        if ds is None:
            ds = self.ds

        if features is None:
            if self.features:
                features = self.features
            else:
                features = ds.features
        
        if features is None:
            print ('No features, how can that happen? :(')
            return None
        
        if sample_name is None:
            sample_name = ds.train_name

        if clf is None:
            clf = self.clf
        if scoring.upper() in ['AIC', 'BIC', 'SIC',  'SBIC']:
            if fit:
                clf.fit(ds.samples[sample_name][features], ds.samples[sample_name][ds.target])
            features_kept = []
            weights_crit = [clf.intercept_[0]]
            for i in range(len(features)):
                if clf.coef_[0][i] != 0:
                    features_kept.append(features[i])
                    weights_crit.append(clf.coef_[0][i])

            intercept_crit = np.ones((ds.samples[sample_name].shape[0], 1))
            features_crit = np.hstack((intercept_crit, ds.samples[sample_name][features_kept]))
            scores_crit = np.dot(features_crit, weights_crit)
            ll = np.sum(ds.samples[sample_name][ds.target] * scores_crit - np.log(np.exp(scores_crit) + 1))

            if scoring.upper() == 'AIC':
                score = 2 * len(weights_crit) - 2 * ll
            else:
                score = len(weights_crit) * np.log(ds.samples[sample_name].shape[0]) - 2 * ll
            score = -score
        else:
            score = cross_val_score(clf, ds.samples[sample_name][features], ds.samples[sample_name][ds.target], cv=cv,
                                    scoring=scoring if scoring != 'gini' else 'roc_auc').mean()
            if scoring == 'gini':
                score = abs(round((2 * score - 1)*100, self.round_digits))
        return score

    def scoring(self, data=None, score_field='score', pd_field='pd', scale_field=None):
        """
        Скоринг выборки.
        :param data: ДатаСэмпл или ДатаФрейм. Возвращается объект того же типа
        :param score_field: поле, в которое должен быть записан посчитанный скор
        :param pd_field: поле, в которое должен быть записан посчитанный PD
        :param scale_field: поле, в которое должен быть записан посчитанный грейд

        :return: ДатаСэмпл или ДатаФрейм с добавленными полями скоров, PD и грейда
        """
        
        def score_df(df):
            df[score_field] = self.intercept + np.dot(df[list(self.coefs.keys())], list(self.coefs.values()))
            if pd_field:
                df[pd_field] = 1 / (1 + np.exp(-df[score_field]))
                if scale_field and self.scale:
                    df[scale_field] = df[pd_field].apply(self.to_scale)
            if self.calibration:
                df[f'{score_field}_calibr'] = self.calibration[0] + self.calibration[1] * df[score_field]
                if pd_field:
                    df[f'{pd_field}_calibr'] = 1 / (1 + np.exp(-df[f'{score_field}_calibr']))
                    if scale_field and self.scale:
                        df[f'{scale_field}_calibr'] = df[f'{pd_field}_calibr'].apply(self.to_scale)
            return df

        if not self.coefs or not self.intercept:
            print(f'No calculated intercept and coefs in self, use self.fit() before. Return None')
        if data is not None:
            if isinstance(data, pd.DataFrame):
                ds = DataSamples(samples={'train': data}, features=[rem_suffix(f) for f in self.coefs], cat_columns=[])
            else:
                ds = copy.deepcopy(data)
        else:
            ds = copy.deepcopy(self.ds)
        if self.transformer is not None:
            ds = self.transformer.transform(ds, features=list(self.coefs.keys()))
        for name in ds.samples:
            ds.samples[name] = score_df(ds.samples[name])
        if ds.bootstrap_base is not None:
            ds.bootstrap_base = score_df(ds.bootstrap_base)
        if isinstance(data, pd.DataFrame):
            return ds.to_df(sample_field=None)
        else:
            return ds

    def get_code(self, score_field='score', pd_field=None, scale_field=None):
        result = '\n'
        if self.transformer is not None:
            for f_WOE in self.features:
                f = f_WOE.rsplit('_WOE', maxsplit=1)[0]
                result += self.transformer.feature_woes[f].get_transform_func(f"df['{f_WOE}'] = ", f"df['{f}']",
                                                                              self.transformer.feature_woes[f].groups,
                                                                              self.transformer.feature_woes[f].woes,
                                                                              self.transformer.feature_woes[f].others_woe) + "\n"
        result += f"df[{score_field}] = {self.intercept} + {' + '.join([f'''({c}) * df['{f}']''' for f, c in self.coefs.items()])}\n"
        if pd_field:
            result += f"df[{pd_field}] = 1 / (1 + np.exp(-df[{score_field}]))\n"
            if scale_field:
                result += f"df[{scale_field}] = df[{pd_field}].apply(to_scale)\n"
        if self.calibration:
            result += f"df[{score_field} + '_calibr'] = {self.calibration[0]} + {self.calibration[1]} * df[{score_field}]\n"
            if pd_field:
                result += f"df[{pd_field} + '_calibr'] = 1 / (1 + np.exp(-df[{score_field} + '_calibr']))\n"
                if scale_field:
                    result += f"df[{scale_field} + '_calibr'] = df[{pd_field} + '_calibr'].apply(to_scale)\n"
        return result

    def to_py(self, file_name='model.py', score_field='score', pd_field='pd', scale_field=None):
        """
        Генерация хардкода функции scoring
        :param file_name: название питоновского файла, куда должен быть сохранен код
        :param score_field: поле, в которое должен быть записан посчитанный скор
        :param pd_field:  поле, в которое должен быть записан посчитанный PD
        :param scale_field:  поле, в которое должен быть записан посчитанный грейд
        """
        result = "import pandas as pd\nimport numpy as np\n\n"
        if scale_field:
            result += f'''
def to_scale(PD):
    scale = {self.scale}
    for s in scale:
        if PD < scale[s]:
            return s
    return 'MSD'\n\n'''

        result += f'''
def scoring(df, score_field='score', pd_field='pd', scale_field=None):
    """
    Функция скоринга выборки
    Arguments:
        df: [pd.DataFrame] входной ДатаФрейм, должен содержать все нетрансформированные переменные модели
        score_field: [str] поле, в которое должен быть записан посчитанный скор
        pd_field: [str] поле, в которое должен быть записан посчитанный PD
        scale_field: [str] поле, в которое должен быть записан посчитанный грейд
    Returns:
        df: [pd.DataFrame] выходной ДатаФрейм с добавленными полями трансформированных переменных, скоров, PD и грейда
    """
'''
        result += self.get_code(score_field='score_field' if score_field else None,
                                pd_field='pd_field' if pd_field else None,
                                scale_field='scale_field' if scale_field else None).replace('\n','\n    ')
        result += f"    return df\n\n"
        result += f'''df = scoring(df, score_field={f"'{score_field}'" if score_field else None}, pd_field={f"'{pd_field}'" if pd_field else None}, scale_field={f"'{scale_field}'" if scale_field else None})'''
        if file_name:
            if self.ds is not None:
                file_name = self.ds.result_folder + file_name
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(result)
            print(f'The model code for implementation saved to file {file_name}')
        print(result)

    def draw_coefs(self, filename=None):
        """
        Отрисовка гистограммы коэффициентов модели
        :param filename: название файла для сохранения
        """
        with plt.style.context(('seaborn-deep')):
            plt.figure(figsize=(5,3))
            feature_list = [f for f in self.coefs]
            coefs_list = [self.coefs[f] for f in self.coefs]
            plt.barh(range(len(coefs_list)), [coefs_list[i] for i in np.argsort(coefs_list)])
            plt.yticks(range(len(coefs_list)), [feature_list[i] for i in np.argsort(coefs_list)])
            plt.tick_params(axis='both', which='both', labelsize=10)
            plt.tight_layout()
            if filename is not None:
                plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.show()
#------------------------------------------------------------------------------------------------------------------


class OrdinalRegressionModel:
    '''
    Ordinal logistic regression for acceptable probability models (mainly for income prediction). It predicts a probability
    that real value is less or equal then the specified value or a value that would be less or equal then the real value with
    the specified probability.

    There is no ordinal regression in sklearn, so this class doesn't inherit sklearn's interface, but it has a similar one.

    An object of this class can:
    1) transform input continuous dependent variable into ordinal by defining classes,
    2) fit separate logistic regressions for analysing the possibility of using ordinal regression (all coefficients by the same
        variables must be "close" to each other) and guessing of initital parameters for ordinal regression,
    3) fit ordinal regression, using scipy.optimize.minimize with SLSQP method and constraints, that class estimates should
        be monotonically increasing, minimizing negative log-likelihood of separate logistic regressions,
    4) fit linear regression for estimating class estimates by the natural logarithm of predicted value, using
        scipy.optimize.minimize with the specified method, minimizing negative R-squared of this linear regression,
    5) predict a probability that real value is less or equal then the specified value,
    6) predict a value that would be less or equal then the real value with the specified probability,
    7) create SAS code for calculating income, using fitted ordinal logistic and linear regressions.
    '''

    def __init__(self, X, y, classes=None, coefficients=None, intercepts=None, bias=None,
                 log_intercept=None, log_coefficient=None, alpha=None):
        '''
        Parameters
        -----------
        X: pandas.Series containing the predictor variables
        y: pandas.Series containing the dependent variable
        classes: an array-like, containing the upper edges of classes of dependent variable
        coefficients: the coefficients by the predictor variables
        intercepts: an array-like with classes estimates
        bias: a shift of the dependent variable used for more accurate prediction of classes estimates
        log_intercept: an intercept for linear regression of classes estimates by classes upper edges
        log_coefficient: a coefficient for linear regression of classes estimates by classes upper edges
        alpha: a probability value used for predicting dependent variable lower edge
        '''
        if len(X.shape)==1:
            X=pd.DataFrame(X)
        if classes is None:
            classes=[x*10000 for x in range(1,11)]

        self.X=X
        self.y=y
        self.classes=classes
        self.coefficients=coefficients
        self.intercepts=intercepts
        self.bias=bias
        self.log_intercept=log_intercept
        self.log_coefficient=log_coefficient
        self.alpha=alpha


    def y_to_classes(self, y=None, classes=None, replace=True):
        '''
        Transform continuous dependent variable to ordinal variable

        Parameters
        -----------
        y: pandas.Series containing the dependent variable
        classes: an array-like, containing the upper edges of classes of dependent variable
        replace: should new ordinal variable replace initial continuous one in self

        Returns
        -----------
        pandas.Series of transformed ordinal dependent variable
        '''
        if y is None:
            y=self.y
        if classes is None:
            classes=self.classes

        bins = pd.IntervalIndex.from_tuples([(-np.inf if i==0 else classes[i-1], np.inf if i==len(classes) else classes[i]) \
                                             for i in range(len(classes)+1)], closed ='left')
        #display(bins)
        new_y=pd.cut(y, bins=bins).apply(lambda x: x.right).astype(float)
        if replace:
            self.y=new_y
        return new_y


    def fit(self, X=None, y=None, params=None, verbose=True):
        '''
        Fit ordinal logistic regression, using scipy.optimize.minimize with SLSQP method and constraints
        of inequality defined as differences between the estimates of the next and current classes (these
        differences will stay non-negative for SLSQP method)

        Parameters
        -----------
        X: pandas.Series containing the predictor variables
        y: pandas.Series containing the dependent variable
        params: an array-like with predictor's coefficient and classes estimates
        verbose: should detailed information be printed
        '''

        def get_guess(X, y, verbose=True):
            '''
            TECH

            Find initial coefficients and classes estimates values by fitting separate logistic regressions

            Parameters
            -----------
            X: pandas.Series containing the predictor variables
            y: pandas.Series containing the dependent variable
            verbose: should detailed information be printed

            Returns
            -----------
            an array-like with predictors' coefficients and classes estimates initial values
            '''
            lr_coefs=pd.DataFrame(columns=list(X.columns)+['intercept', 'amount', 'gini'])
            for target in self.classes:
                check_class=(y<=target)
                lr=LogisticRegression(C=100000, random_state=40)
                lr.fit(X, check_class)
                fpr, tpr, _ = roc_curve(check_class, lr.predict_proba(X)[:,1])

                lr_coefs.loc[target]=list(lr.coef_[0])+[lr.intercept_[0], check_class.sum(), 2*auc(fpr, tpr)-1]
            if verbose:
                for pred in X.columns:
                    plt.figure(figsize = (10,5))
                    plt.barh(range(lr_coefs.shape[0]), lr_coefs[pred].tolist())
                    for i in range(lr_coefs.shape[0]):
                        plt.text(lr_coefs[pred].tolist()[i], i, str(round(lr_coefs[pred].tolist()[i],5)),
                                 horizontalalignment='left', verticalalignment='center', fontweight='bold', fontsize=10)
                    plt.yticks(range(lr_coefs.shape[0]), lr_coefs.index.tolist())
                    plt.suptitle('Coefficients of separate logistic regressions')
                    plt.xlabel('Coefficients for predictor '+pred)
                    plt.ylabel('Predicted value caps (classes)')
                    plt.margins(x=0.1)
                    plt.show()
                plt.figure(figsize = (10,5))
                plt.bar(range(lr_coefs.shape[0]), lr_coefs.gini.tolist())
                for i in range(lr_coefs.shape[0]):
                    plt.text(i, lr_coefs.gini.tolist()[i], str(round(lr_coefs.gini.tolist()[i],4)),
                             horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=10)
                plt.xticks(range(lr_coefs.shape[0]), lr_coefs.index.tolist())
                plt.suptitle('Gini values of separate logistic regressions')
                plt.ylabel('Gini value')
                plt.xlabel('Predicted value caps (classes)')
                plt.margins(y=0.1)
                plt.show()
            return np.array(lr_coefs[list(X.columns)].mean().tolist()+lr_coefs.intercept.tolist())

        if X is None:
            X=self.X
        if y is None:
            y=self.y

        if params is None:
            if verbose:
                print('No initial parameters specified. Guessing approximate parameters by fitting separate logistic regressions..')
            guess=get_guess(X, y, verbose=verbose)
            if verbose:
                print('Initial guess:')
                print('\tCoefficients (<predictor> = <coefficient>):')
                for i in range(len(X.columns)):
                    print('\t',X.columns[i],'=',guess[i])
                print('\tClass estimates (<class> = <estimate>):')
                for i in range(len(self.classes)):
                    print('\t',self.classes[i],'=',guess[i+len(X.columns)])
                print()
        else:
            guess=params.copy()


        cons=tuple({'type': 'ineq', 'fun': lambda x:  x[i+1] - x[i]} for i in range(len(X.columns),len(X.columns)+len(self.classes)-1))

        results = minimize(self.OrdinalRegressionLL, guess, args=(X, y),
                           method = 'SLSQP', options={'disp': verbose}, constraints=cons)

        self.coefficients=results['x'][:len(X.columns)]
        self.intercepts=list(results['x'][len(X.columns):])
        if verbose:
            print()
            print('Fitted values:')
            print('\tCoefficients (<predictor> = <coefficient>):')
            for i in range(len(X.columns)):
                print('\t',X.columns[i],'=',self.coefficients[i])
            print('\tClass estimates (<class> = <estimate>):')
            for i in range(len(self.classes)):
                print('\t',self.classes[i],'=',self.intercepts[i])


    def fit_class_estimates(self, bias=None, verbose=True, method='Nelder-Mead'):
        '''
        Fit linear regression, using scipy.optimize.minimize with the specified method. Nelder-Mead showed
        the best results during testing.

        Parameters
        -----------
        bias: a shift of the dependent variable used for more accurate prediction of classes estimates
        verbose: should detailed information be printed
        method: optimization method for finding the best bias value
        '''
        to_log=pd.DataFrame([self.classes, self.intercepts], index=['value', 'estimate']).T

        if bias is None:
            if verbose:
                print('Searching for an optimal bias value..')
            results = minimize(self.LinearRegressionR2, [0], args=(to_log.value, to_log.estimate),
                               method=method, options={'disp': verbose})
            self.bias=results['x'][0]
        else:
            self.bias=bias

        biased_X=np.log(to_log.value+self.bias)
        lir=LinearRegression()
        lir.fit(biased_X.reshape(-1,1), to_log.estimate)
        self.log_intercept=lir.intercept_
        self.log_coefficient=lir.coef_[0]

        r2=r2_score(to_log.estimate, lir.predict(biased_X.reshape(-1,1)))

        if verbose:
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            ax.scatter(y=to_log.estimate, x=to_log.value, label='Actual estimates')
            ax.plot(to_log.value, lir.predict(biased_X.reshape(-1,1)), 'r-', label='Predicted estimates')
            ax.text(0.95, 0.1,
                    'y = '+str(round(self.log_coefficient,5))+'*ln(x+'+str(round(self.bias,5))+') '+\
                        ('+ ' if self.log_intercept>=0 else '')+str(round(self.log_intercept,5)),
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.7))
            ax.set_xlabel('Predicted value caps (classes)')
            ax.set_ylabel('Classes estimates')
            ax.legend()
            plt.show()
            print('Bias =', self.bias, 'R2_score =', r2)


    def predict(self, alpha=None, X=None):
        '''
        Predict a value that would be less or equal then the real value with the specified probability

        Parameters
        -----------
        X: pandas.Series containing the predictor variable
        alpha: a probability value used for predicting dependent variable lower edge

        Returns
        -----------
        pandas.Series of predicted values
        '''
        if X is None:
            X=self.X
        if alpha is not None:
            self.alpha=alpha
        elif alpha is None:
            alpha=self.alpha
        return np.exp((np.log((1-alpha)/alpha) - (self.coefficients*X).sum(axis=1) - self.log_intercept)/self.log_coefficient) - self.bias


    def predict_proba(self, predicted_value, X=None):
        '''
        Predict a probability that real value is less or equal then the specified value

        Parameters
        -----------
        predicted_value: a value to be used as the upper edge of predicted values in probability calculation
        X: pandas.Series containing the predictor variable

        Returns
        -----------
        pandas.Series of predicted probabilities
        '''
        if X is None:
            X=self.X

        if self.log_intercept is None or self.log_coefficient is None:
            print('Class estimates were not fitted, only initial class values are available for prediction.')
            class_num=np.argmin(np.abs(np.array(self.classes)-predicted_value))
            predicted_value=self.classes[class_num]
            intercept=self.intercepts[class_num]
            print('Using', predicted_value,'instead of input value.')
            return 1/(1+np.exp(-((self.coefficients*X).sum(axis=1)+intercept)))
        else:
            intercept=self.log_coefficient*np.log(predicted_value+self.bias)+self.log_intercept
            return 1/(1+np.exp(-((self.coefficients*X).sum(axis=1)+intercept)))


    def to_sas(self, alpha=None):
        '''
        Print SAS code with income calculation formula

        Parameters
        -----------
        alpha: a probability value used for predicting dependent variable lower edge
        '''
        if alpha is None:
            alpha=self.alpha
        print('BIAS =', self.bias,';')
        print('INCOME_INTERCEPT =', self.log_intercept,';')
        print('INCOME_COEF =', self.log_coefficient,';')
        for i in range(len(self.X.columns)):
            print(self.X.columns[i]+'_COEF =', self.coefficients[i],';')
        print('ALPHA =', alpha,';')
        print()
        final_formula='INCOME_FORECAST = exp((log((1-ALPHA)/ALPHA)'
        for i in range(len(self.X.columns)):
            final_formula+=' - '+self.X.columns[i]+'_COEF*'+self.X.columns[i]
        final_formula+=' - INCOME_INTERCEPT)/INCOME_COEF) - BIAS;'
        print(final_formula)


    def OrdinalRegressionLL(self, params, X=None, y=None):
        '''
        Returns negative summed up log-likelihood of separate logistic regressions for MLE

        Parameters
        -----------
        params: an array-like with predictors' coefficients and classes estimates
        X: pandas.Series containing the predictor variables
        y: pandas.Series containing the dependent variable

        Returns
        -----------
        negative summed up log-likelihood
        '''
        if X is None:
            X=self.X
        if y is None:
            y=self.y

        negLL = 0
        #calculate and sum up log-likelyhood for each class as if it is a separate model
        for i in range(len(self.classes)):
            #taking class intercept and the common coefficients
            weights=[params[i+len(self.X.columns)]]
            for p in range(len(self.X.columns)):
                weights.append(params[p])

            intercept = np.ones((X.shape[0], 1))
            features = np.hstack((intercept, X))
            scores = np.dot(features, weights)
            negLL -= np.sum((y<=self.classes[i])*scores - np.log(np.exp(scores) + 1))
        # return negative LL
        return negLL

    def LinearRegressionR2(self, params, X, y):
        '''
        Returns negative R-squared of linear regression for minimizing

        Parameters
        -----------
        params: an array-like with the bias value
        X: pandas.Series containing the predictor variable
        y: pandas.Series containing the dependent variable

        Returns
        -----------
        negative R-squared
        '''
        biased_X=np.log(X+params[0])

        lir=LinearRegression()
        lir.fit(biased_X.reshape(-1,1), y)

        return - r2_score(y, lir.predict(biased_X.values.reshape(-1,1)))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
