# -*- coding: utf-8 -*-

from ..data import DataSamples
from ..woe import WOE
from ..bankmetrics import *
from .._utils import fig_to_excel, adjust_cell_width, add_suffix, rem_suffix, is_cross, cross_name, cross_split, add_ds_folder
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.metrics import roc_curve, auc, r2_score, make_scorer
from scipy.stats import chi2, chisquare, ks_2samp, ttest_ind, kstest
import warnings
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
plt.rc('font', family='Verdana', size=11)
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)
gc.enable()


class LogisticRegressionModel:
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
        if clf is not None:
            self.clf = clf
        else:
            self.clf = SGDClassifier(loss='log', penalty='l2', max_iter=1000, alpha=0.001,
                                     random_state=self.ds.random_state if self.ds is not None else 0)
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
        if self.transformer is not None and self.coefs:
            model['scorecard'] = base64.b64encode(cloudpickle.dumps(self.transformer.export_scorecard(features=[rem_suffix(f) for f in self.coefs], full=False).drop(['target_rate', 'sample_part'], axis=1))).decode()
        if file_name is not None:
            file_name = add_ds_folder(self.ds, file_name)
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
            self.transformer = WOE(ds=self.ds, scorecard=scorecard)
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
                self.ds = DataSamples(samples={'Train': data}, target=target, time_column=time_column, id_column=id_column,
                                 feature_descriptions=feature_descriptions, result_folder=result_folder, n_jobs=n_jobs)
                self.ds.samples_split()
            else:
                self.ds = copy.deepcopy(data)
                if n_jobs is not None:
                    self.ds.n_jobs = n_jobs
                if result_folder and isinstance(result_folder, str):
                    if not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    self.ds.result_folder = result_folder + ('/' if result_folder and not result_folder.endswith('/') else '')

        if self.transformer is None:
            self.transformer = WOE(ds=self.ds)
            self.transformer.auto_fit(plot_flag=-1, method=method)
        else:
            print('Using existing self.transformer.')
        if not self.coefs:
            self.ds = self.transformer.transform(self.ds, verbose=True)
            self.mfa(gini_threshold=10, result_file=f'{self.name + "_" if self.name else ""}mfa.xlsx')
        else:
            print('Using existing self.coefs.')
            self.report(out=out, sheet_name=None, pvalue_threshold=0.05, verbose=False)

        if save_model:
            self.save_model(file_name=f'{self.ds.result_folder}{save_model}')
        if validate:
            self.validate(result_file='auto_validation.xlsx')

    def mfa(self, ds=None, features=None, hold=None, features_ini=None, limit_to_add=100, gini_threshold=5,
            corr_method='pearson', corr_threshold=0.75, drop_with_most_correlations=False, selection_type='stepwise',
            pvalue_threshold=0.05, pvalue_priority=False, scoring='gini', score_delta=0.1, cv=None,
            drop_positive_coefs=True, crosses_simple=True, crosses_max_num=10,
            verbose=True, result_file='mfa.xlsx', metrics=None, metrics_cv=None):
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
        :param features_ini: список переменных, с которых стартует процедура отбора. Они могут быть исключены в процессе отбора
        :param limit_to_add: максимальное кол-во переменных, которые могут быть добавлены к модели
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
        :param cv: параметр cv для вызова sklearn.model_selection.cross_val_score. При None берется StratifiedKFold(5, shuffle=True)
        :param drop_positive_coefs: флаг для выполнения этапа 4

        --- Кросс переменные ---
        :param crosses_simple: True  - после трансформации кросс-переменные учавствут в отборе наравне со всеми переменными
                               False - сначала выполняется отбор только на основных переменных,
                                       затем в модель добавляются по тем же правилам кросс переменные, но не более, чем crosses_max_num штук
        :param crosses_max_num: максимальное кол-во кросс переменных в модели. учитывается только при crosses_simple=False

        --- Отчет ---
        :param verbose: флаг для вывода подробных комментариев в процессе работы
        :param result_file: название файла отчета. При None результаты не сохраняются
        :param metrics: список метрик/тестов, результы расчета которых должны быть включены в отчет.
                          Элементы списка могут иметь значения (не чувствительно к регистру):
                              'ontime': расчет динамики джини по срезам,
                              'vif'   : расчет Variance Inflation Factor,
                              'psi'   : расчет Population Population Stability Index,
                              'wald'  : тест Вальда,
                              'ks'    : тест Колмогорова-Смирнова,
                              func    : пользовательская функция, которая принимает целевую и зависимую переменную,
                                        и возвращает числовое значение метрики

                                        Например,
                                        def custom_metric(y_true, y_pred):
                                            from sklearn.metrics import roc_curve, f1_score
                                            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                                            thres = thresholds[np.argmax(tpr * (1 - fpr))]
                                            return f1_score(y_true, (y_pred > thres).astype(int))
                                        metrics = ['vif', 'ks', 'psi', custom_metric]
        :param metrics_cv: список метрик, рассчитываемых через sklearn.model_selection.cross_val_score.
                          Аналогично параметру metrics элементами могут быть строки, поддерживаемые cross_val_score, либо пользовательские функции
                          Например, ['roc_auc', 'neg_log_loss', 'gini', 'f1', 'accuracy', custom_metric]
        """
        print(f"\n{' MFA ':-^150}\n")
        if ds is not None:
            self.ds = copy.deepcopy(ds)
        if self.transformer is not None:
            self.ds = self.transformer.transform(self.ds, features=features, verbose=verbose)
        if features is None or self.transformer is not None:
            features = self.ds.features.copy()
        if features_ini is None:
            features_ini = []
        hold = set() if hold is None else {add_suffix(f) if add_suffix(f) in features else f for f in hold}
        features_ini = [add_suffix(f) for f in features_ini]
        gini_df = self.ds.calc_gini(add_description=True, features=features)
        if not crosses_simple:
            cross_features = [f for f in features if is_cross(f)]
            features = [f for f in features if not is_cross(f)]
        drop_features_gini, drop_features_corr, selection_fig, regularized_fig = \
            self.mfa_steps(features=features, hold=hold, gini_threshold=gini_threshold, corr_method=corr_method,
                           corr_threshold=corr_threshold, drop_with_most_correlations=drop_with_most_correlations,
                           selection_type=selection_type, pvalue_threshold=pvalue_threshold, pvalue_priority=pvalue_priority,
                           scoring=scoring, score_delta=score_delta, cv=cv, drop_positive_coefs=drop_positive_coefs,
                           features_ini=features_ini, limit_to_add=limit_to_add, verbose=verbose, gini_df=gini_df)
        if not crosses_simple and crosses_max_num > 0 and cross_features:
            print(f"\n{' MFA for cross features ':-^125}\n")
            cross_drop_features_gini, cross_drop_features_corr, cross_selection_fig, cross_regularized_fig = \
                self.mfa_steps(features=cross_features, hold=hold, gini_threshold=gini_threshold, corr_method=corr_method,
                               corr_threshold=corr_threshold, drop_with_most_correlations=drop_with_most_correlations,
                               selection_type='stepwise', pvalue_threshold=pvalue_threshold, pvalue_priority=pvalue_priority,
                               scoring=scoring, score_delta=score_delta, cv=cv, drop_positive_coefs=drop_positive_coefs,
                               features_ini=self.features, limit_to_add=crosses_max_num, verbose=verbose, gini_df=gini_df)
        else:
            cross_drop_features_gini, cross_drop_features_corr, cross_selection_fig, cross_regularized_fig = [], {}, None, None
        if result_file is not None:
            with pd.ExcelWriter(self.ds.result_folder + result_file,  engine='xlsxwriter') as writer:
                gini_df[gini_df.index.isin(drop_features_gini)].to_excel(writer, sheet_name='Gini below threshold')
                adjust_cell_width(writer.sheets['Gini below threshold'], gini_df)
                gini_df['Drop reason'] = gini_df.index.map(drop_features_corr)
                self.ds.corr_mat(sample_name=self.ds.train_name, features=features, description_df=gini_df).to_excel(writer, sheet_name='Correlation analysis')
                adjust_cell_width(writer.sheets['Correlation analysis'], gini_df)
                add_figs = [selection_fig, regularized_fig]
                if not crosses_simple and crosses_max_num > 0 and cross_features:
                    gini_df[gini_df.index.isin(cross_drop_features_gini)].to_excel(writer, sheet_name='Cross Gini below threshold')
                    adjust_cell_width(writer.sheets['Cross Gini below threshold'], gini_df)
                    gini_df['Drop reason'] = gini_df.index.map(cross_drop_features_corr)
                    self.ds.corr_mat(sample_name=self.ds.train_name, features=cross_features, description_df=gini_df).to_excel(writer, sheet_name='Cross Correlation analysis')
                    adjust_cell_width(writer.sheets['Cross Correlation analysis'], gini_df)
                    add_figs += [cross_selection_fig, cross_regularized_fig]
                self.report(out=writer, sheet_name=selection_type, pvalue_threshold=pvalue_threshold, verbose=verbose,
                            add_figs=add_figs, gini_df=gini_df.drop(['Drop reason'], axis=1), metrics=metrics, metrics_cv=metrics_cv, cv=cv)

    def mfa_steps(self, features=None, hold=None, gini_threshold=5, verbose=False,
                  corr_method='pearson', corr_threshold=0.75, drop_with_most_correlations=False, 
                  selection_type='stepwise', pvalue_threshold=0.05, pvalue_priority=False,
                  scoring='gini', score_delta=0.1, cv=None, drop_positive_coefs=True, 
                  features_ini=None, limit_to_add=100, gini_df=None):
        if cv is None:
            cv = StratifiedKFold(5, shuffle=True, random_state=self.ds.random_state)
        if gini_df is None:
            gini_df = self.ds.calc_gini(add_description=True, features=features)
        ginis = gini_df[self.ds.train_name].to_dict()
        features = sorted(features, key=ginis.get(0), reverse=True)
        drop_features_gini = [f for f in features if abs(ginis.get(f, 0)) < gini_threshold and f not in hold]
        features = [f for f in features if f not in drop_features_gini]
        if verbose:
            print(f"\n{' Step 1 ':-^100}")
            print(f'Dropped features with gini lower {gini_threshold}: {drop_features_gini}')
        if not features:
            print('Features set is empty. Break.')
            self.features = []
            return drop_features_gini, {}, None, None
        if verbose:
            print(f"\n{' Step 2 ':-^100}")
        drop_features_corr = self.ds.CorrelationAnalyzer(sample_name=self.ds.train_name, features=features, hold=hold,
                                                         scores=ginis, drop_with_most_correlations=drop_with_most_correlations,
                                                         method=corr_method, threshold=corr_threshold, verbose=verbose)
        features = [f for f in features if f not in drop_features_corr]
        if verbose:
            print(f"\n{' Step 3 ':-^100}")
        if selection_type in ['stepwise', 'forward', 'backward']:
            features, selection_fig = self.stepwise_selection(features=features, hold=hold, features_ini=features_ini, 
                                                              limit_to_add=limit_to_add, verbose=verbose,
                                                              score_delta=score_delta, scoring=scoring, cv=cv,
                                                              pvalue_threshold=pvalue_threshold,
                                                              pvalue_priority=pvalue_priority,
                                                              selection_type=selection_type)
        else:
            if selection_type != 'regularized':
                print('Incorrect selection_type value. Set to regularized.')
            selection_type = 'regularized'
            selection_fig = None
        if drop_positive_coefs:
            if verbose:
                print(f"\n{' Step 4 ':-^100}")
            features, regularized_fig = self.regularized_selection(features=features, hold=hold, scoring=scoring, cv=cv,
                                                                   pvalue_threshold=pvalue_threshold, verbose=verbose)
        else:
            regularized_fig = None
        features = sorted([f for f in features if not is_cross(f)], key=ginis.get(0), reverse=True) + sorted([f for f in features if is_cross(f)], key=ginis.get(0), reverse=True)
        print(f"\n{' Final model ':-^100}")
        self.fit(features=features)
        return drop_features_gini, drop_features_corr, selection_fig, regularized_fig

    def report(self, ds=None, out='report.xlsx', sheet_name=None, pvalue_threshold=0.05, verbose=False, add_figs=None,
               gini_df=None, metrics=None, metrics_cv=None, cv=None):
        """
        Генерация отчета по обученной модели.
        :param ds: ДатаСэмпл. В случае, если он не содержит трансформированные переменные, то выполняется трансформация трансформером self.transformer
        :param out: либо строка с названием эксель файла, либо объект pd.ExcelWriter для сохранения отчета
        :param sheet_name: название листа в экселе
        :param pvalue_threshold: граница по p-value. Используется только для выделения значений p-value цветом
        :param verbose: флаг вывода комментариев в процессе работы
        :param add_figs: список из графиков, которые должны быть добавлены в отчет
        :param gini_df: датафрейм с джини всех переменных модели
        :param metrics: список метрик/тестов, результы расчета которых должны быть включены в отчет.
                          Элементы списка могут иметь значения (не чувствительно к регистру):
                              'ontime': расчет динамики джини по срезам,
                              'vif'   : расчет Variance Inflation Factor,
                              'psi'   : расчет Population Population Stability Index,
                              'wald'  : тест Вальда,
                              'ks'    : тест Колмогорова-Смирнова,
                              func    : пользовательская функция, которая принимает целевую и зависимую переменную,
                                        и возвращает числовое значение метрики

                                        Например,
                                        def custom_metric(y_true, y_pred):
                                            from sklearn.metrics import roc_curve, f1_score
                                            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                                            thres = thresholds[np.argmax(tpr * (1 - fpr))]
                                            return f1_score(y_true, (y_pred > thres).astype(int))
                                        metrics = ['vif', 'ks', 'psi', custom_metric]
        :param metrics_cv: список метрик, рассчитываемых через sklearn.model_selection.cross_val_score.
                          Аналогично параметру metrics элементами могут быть строки, поддерживаемые cross_val_score, либо пользовательские функции
                          Например, ['roc_auc', 'neg_log_loss', 'gini', 'f1', 'accuracy', custom_metric]
        :param cv: параметр cv для вызова sklearn.model_selection.cross_val_score
        """
        if ds is None:
            ds = copy.deepcopy(self.ds)
        if sheet_name is None:
            sheet_name = self.name if self.name else 'model'
        if not self.coefs or not self.features:
            print('Please fit your model before calling this method.')
            return None
        if isinstance(out, str):
            writer = pd.ExcelWriter(add_ds_folder(ds, out), engine='xlsxwriter')
        elif isinstance(out, pd.ExcelWriter):
            writer = out
        else:
            print('Parameter out must have str or pd.ExcelWriter type.')
            return None
        print('Generating report...')
        if metrics is None:
            metrics = ['wald', 'ks', 'vif', 'iv', 'psi', 'ontime']
        metrics = [t.lower() if isinstance(t, str) else t for t in metrics]
        if verbose:
            self.draw_coefs()
        score_field = 'model_score'
        if self.transformer is not None:
            ds = self.transformer.transform(ds, features=self.features, verbose=verbose)
        ds = self.scoring(ds, score_field=score_field)
        if gini_df is None:
            gini_df = ds.calc_gini(add_description=True, features=self.features)
        gini_df = gini_df.merge(pd.DataFrame.from_dict({**self.coefs, **{'intercept': self.intercept}}, orient='index', columns=['coefficient']), left_index=True, right_index=True, how='right')
        model_gini = ds.calc_gini(features=[score_field], abs=True)
        gini_df = pd.concat([gini_df, model_gini]).rename_axis('feature')
        gini_df.columns = [('Gini', c) if c in list(ds.samples) + ['Bootstrap mean', 'Bootstrap std'] else (c, '') for c in gini_df.columns]
        for m in metrics:
            if m == 'wald':
                wald_df = pd.concat([self.wald_test(ds, sample_name=name, features=self.features)[['se', 'p-value'] if name == ds.train_name else ['p-value']].rename({'p-value': name}, axis=1) for name, sample in ds.samples.items()], axis=1)
                wald_df.columns = [('Wald p-value', c) if c in list(ds.samples) else (c, '') for c in wald_df.columns]
                gini_df = gini_df.merge(wald_df, left_index=True, right_index=True, how='left')
            elif m == 'ks':
                gini_df = gini_df.merge(pd.concat([ds.KS_test(sample_name=name, features=self.features + [score_field]).set_axis([('Kolmogorov-Smirnov test', name)], axis=1) for name in ds.samples], axis=1), left_index=True, right_index=True, how='left')
            elif m == 'vif':
                gini_df = gini_df.merge(pd.concat([ds.VIF(features=self.features, sample_name=name).set_axis([('VIF', name)], axis=1) for name in ds.samples], axis=1), left_index=True, right_index=True, how='left')
            elif m == 'iv':
                gini_df = gini_df.merge(pd.concat([ds.IV_test(features=self.features, sample_name=name).set_axis([('IV', name)], axis=1) for name in ds.samples], axis=1), left_index=True, right_index=True, how='left')
            elif callable(m):
                gini_df = gini_df.merge(pd.concat([pd.DataFrame.from_dict({f: m(sample[ds.target], sample[f]) for f in self.features + [score_field]}, orient='index', columns=[(m.__name__, name)]) for name, sample in ds.samples.items()], axis=1), left_index=True, right_index=True, how='left')

        gini_df.columns = pd.MultiIndex.from_tuples(gini_df.columns)
        gini_df.rename(index={score_field: 'model'}).style.applymap(lambda x: 'color: red' if x > pvalue_threshold else 'color: orange' if x > pvalue_threshold / 5 else 'color: black',
                               subset=pd.IndexSlice[:, [f for f in gini_df if f[0] == 'Wald p-value']]) \
            .to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, float_format=f'%0.{self.round_digits}f')
        ds.corr_mat(features=self.features).to_excel(writer, sheet_name=sheet_name, startrow=3, startcol=len(gini_df.columns) + 3)
        ws = writer.sheets[sheet_name]
        descr_len = len(ds.feature_descriptions.columns) if ds.feature_descriptions is not None else 0
        ws.set_column(0, 0 + descr_len, 30)
        ws.set_column(1 + descr_len, gini_df.shape[1], 15)
        ws.set_column(len(gini_df.columns) + 3, gini_df.shape[1] + 3, 30)
        ws.write(0, 0, 'Features in model:')
        ws.write(0, len(gini_df.columns) + 3, 'Correlations matrix:')
        m_col = 10
        if metrics_cv:
            model_metrics = []
            for m in metrics_cv:
                if callable(m):
                    try:
                        metric = {'Metric': m.__name__}
                        m = make_scorer(m)
                    except:
                        metric = {'Metric': str(m)}
                else:
                    metric = {'Metric': str(m)}
                for name, sample in ds.samples.items():
                    scores = cross_val_score(self.clf, sample[self.features], sample[ds.target], cv=cv, scoring=m if m != 'gini' else 'roc_auc')
                    if m == 'gini':
                        scores = np.array([2*x - 1 for x in scores])
                    metric[f'{name} mean'] = round(scores.mean(), self.round_digits)
                    metric[f'{name} std'] = round(scores.std(), self.round_digits)
                model_metrics.append(metric)
            ws.write(len(self.features) + 7, m_col, 'Model metrics on cross-validation:')
            pd.DataFrame(model_metrics).set_index('Metric').to_excel(writer, sheet_name=sheet_name, startrow=len(self.features) + 8, startcol=m_col)
            m_col += len(ds.samples)*2 + 3
        fig_to_excel(self.roc_curve(ds, verbose=verbose, score_field=score_field), ws, row=0,
                     col=gini_df.shape[1] + max(len(self.features), len(ds.samples)) + 6)
        if add_figs:
            row = 0
            for fig in add_figs:
                if fig:
                    fig_to_excel(fig, ws, row=row, col=len(gini_df.columns) + max(len(self.features), len(ds.samples)) + 16)
                    row = +16
        print(model_gini.rename(index={score_field: 'Gini'}))
        if 'ontime' in metrics and ds.time_column:
            ws.write(len(self.features) + 7, m_col, 'Model Gini dynamics:')
            features_gini = ds.calc_gini_in_time(features=self.features + [score_field], abs=True)
            model_gini = features_gini.iloc[:, features_gini.columns.get_level_values(0) == score_field].copy()
            model_gini.columns = model_gini.columns.droplevel()
            model_gini.to_excel(writer, sheet_name=sheet_name, startrow=len(self.features) + 8, startcol=m_col)
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
            fig_to_excel(fig, ws, row=len(self.features) + len(model_gini) + 11, col=m_col - 1, scale=0.75)
            ws.write(len(self.features) + len(model_gini) + 36, m_col, 'Features Gini dynamics:')
            features_gini.iloc[:, features_gini.columns.get_level_values(0) != score_field].to_excel(writer, sheet_name=sheet_name,
                                    startrow=len(self.features) + len(model_gini) + 28, startcol=m_col)

        if self.transformer is not None:
            ws.write(len(self.features) + 7, 0, 'Scorecard:')
            scorecard = self.transformer.export_scorecard(features=[rem_suffix(f) for f in self.features], full=False)
            scorecard.set_index('feature').to_excel(writer, sheet_name=sheet_name,
                                                    startrow=len(self.features) + 8, startcol=0,
                                                    float_format=f'%0.{self.round_digits}f')
            figs = self.transformer.plot_bins(features=[rem_suffix(f) for f in self.features], folder=None, plot_flag=verbose)
            for i, fig in enumerate(figs):
                fig_to_excel(fig, ws,
                             row=len(scorecard) + len(self.features) + 11 + i * (20 if ds.time_column is None else 30),
                             col=0, scale=0.7)
        else:
            scorecard = None

        if 'psi' in metrics:
            result, figs = DataSamples(samples={'Train': ds.to_df(sample_field='sample')[self.features + [ds.target, 'sample']]},
                                       target=ds.target, features=self.features, cat_columns=[],
                                       time_column='sample').psi(scorecard=scorecard)
            result.to_excel(writer, sheet_name='PSI_samples')
            ws = writer.sheets['PSI_samples']
            ws.set_column(0, 0, 40)
            ws.set_column(1, len(result.columns) + 1, 12)
            for i, fig in enumerate(figs):
                fig_to_excel(fig, ws, row=i * (20 + 1) + len(self.features) + 3, col=0, scale=0.9)
            if ds.time_column:
                result, figs = ds.psi(features=self.features, scorecard=scorecard)
                result.to_excel(writer, sheet_name='PSI_time')
                ws = writer.sheets['PSI_time']
                ws.set_column(0, 0, 40)
                ws.set_column(1, len(result.columns) + 1, 12)
                for i, fig in enumerate(figs):
                    fig_to_excel(fig, ws, row=i * (20 + 1) + len(self.features) + 3, col=0, scale=0.9)
        if isinstance(out, str):
            writer.close()
        plt.close('all')

    def calc_gini(self, ds=None, score_field='model_score'):
        if ds is None:
            ds = self.ds
        if any([score_field not in sample.columns for sample in ds.samples.values()]):
            ds = self.scoring(ds, score_field=score_field)
        return ds.calc_gini(features=[score_field], abs=True)

    def roc_curve(self, ds=None, score_field=None, verbose=True):
        """
        Рассчет джини модели на всех сэмплах и построение ROC-кривой
        :param ds: ДатаСэмпл
        :param verbose: флаг для вывода ROC-кривой в аутпут

        :return: кортеж (словарь {сэмпл: джини}, ROC-кривая в виде plt.fig)
        """
        if ds is None:
            ds = self.ds
        if score_field is None:
            score_field = 'score'
            ds = self.scoring(ds, score_field=score_field)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        for name, sample in ds.samples.items():
            fpr, tpr, _ = roc_curve(sample[ds.target], sample[score_field])
            ax.plot(fpr, tpr, label=f'{name} (Gini = {round((auc(fpr, tpr) * 2 - 1) * 100, 2)})')
        ax.plot(tpr, tpr, 'r')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        if verbose:
            plt.show()
        return fig

    def validate(self, ds=None, result_file='validation.xlsx', score_field='score', pd_field='pd', scale_field=None):
        """
        Валидационные тесты модели на заданном ДатаСэмпле библиотекой bankmetrics
        :param ds: ДатаСэмпл
        :param result_file: результирующий эксель файл
        :param score_field: поле со расчитанным скором (при отсутcnвии этого поля в выборке будет вызван метод self.scoring)
        :param pd_field: поле с расчитанным PD
        :param scale_field: поле с расчитанным грейдом
        """
        print(f"\n{' Validation ':-^150}\n")
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
                gini = DataSamples.get_f_gini(df[ds.target], df[score_field])
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
                   on='feature', how='left').set_index('feature')

    def regularized_selection(self, ds=None, features=None, hold=None, scoring='gini', cv=None, pvalue_threshold=0.05, verbose=False):
        """
        Отбор факторов на основе регуляризации - строится модель на всех переменных, затем итерационно исключаются
        переменные с нулевыми или положительными коэффициентами и низкой значимостью
        :param ds: ДатаСэмпл. При None берется self.ds
        :param features: исходный список переменных. При None берется self.features
        :param hold: список/сет переменных, которые обязательно должны остаться после отбора
        :param scoring: расчитываемый скор модели
        :param pvalue_threshold: граница значимости по p-value
        :param verbose: флаг для вывода подробных комментариев в процессе работы

        :return: кортеж (итоговый список переменных, график со скором в процессе отбора в виде объекта plt.figure)
        """
        if ds is not None:
            self.ds = ds

        if features is None:
            if self.features:
                features = self.features.copy()
            else:
                features = self.ds.features.copy()
        features = [add_suffix(f) if add_suffix(f) in self.ds.features else f for f in features]
        if hold is None:
            hold = set()
        else:
            hold = {add_suffix(f) if add_suffix(f) in features else f for f in hold}

        # correctness check
        for f in features:
            if f not in self.ds.samples[self.ds.train_name].columns:
                print(f'No {f} in DataSample!')
                return None

        ginis = self.ds.calc_gini(features=features, samples=[self.ds.train_name])[self.ds.train_name].to_dict()
        scores = []
        features_change = ['Initial']
        to_refit = True
        if verbose:
            print(f'Dropping features with positive coefs and high p-values...')
        while to_refit:
            to_refit = False
            self.clf.fit(self.ds.samples[self.ds.train_name][features], self.ds.samples[self.ds.train_name][self.ds.target])
            new_score = self.get_score(scoring=scoring, cv=cv, features=features, fit=False)
            scores.append(new_score)
            positive_to_exclude = {f: c for f, c in zip(features, self.clf.coef_[0]) if c > 0 and f not in hold}
            if positive_to_exclude:
                to_refit = True
                features_to_exclude = {x: ginis[x] for x in positive_to_exclude}
                to_exclude = min(features_to_exclude, key=features_to_exclude.get)
                if verbose:
                    print(f'To drop: {to_exclude}, gini: {ginis[to_exclude]}, coef: {positive_to_exclude[to_exclude]}')
                features.remove(to_exclude)
                features_change.append(f'- {to_exclude}')
            else:
                wald = self.wald_test(clf=self.clf, features=features)
                feature_to_exclude_array = wald[(wald['p-value'] > pvalue_threshold) & (wald['p-value'] == wald['p-value'].max()) & (wald.index.isin(list(hold) + ['intercept']) == False)].index.values
                if feature_to_exclude_array:
                    to_refit = True
                    if verbose:
                        print(f'To drop: {feature_to_exclude_array[0]}, gini: {ginis[feature_to_exclude_array[0]]}, p-value: {wald[wald.index == feature_to_exclude_array[0]]["p-value"].values[0]}')
                    features.remove(feature_to_exclude_array[0])
                    features_change.append(f'- {feature_to_exclude_array[0]}')

        features = [f for f, c in zip(features, self.clf.coef_[0]) if c != 0]
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
        if clf.coef_[0][0] == 0:
            return 1, 0
        coefs_list = [clf.intercept_[0]] + [c for c in clf.coef_[0] if c != 0]
        features_to_check = [f for f, c in zip(features, clf.coef_[0]) if c != 0]
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
                intercept_crit = np.ones((df.shape[0], 1))
                features_crit = np.hstack((intercept_crit, df[features_to_check]))
                scores_crit = np.dot(features_crit, coefs_list)
                ll = np.sum((df.iloc[:, 0] * scores_crit - np.log(np.exp(scores_crit) + 1)))
                if scoring.upper() == 'AIC':
                    score = 2 * len(coefs_list) - 2 * ll
                else:
                    score = len(coefs_list) * np.log(df.shape[0]) - 2 * ll
                score = -score
            else:
                score = cross_val_score(clf, df[features], df.iloc[:, 0], cv=cv, scoring=scoring if scoring != 'gini' else 'roc_auc').mean()
                if scoring == 'gini':
                    score = round((2 * score - 1)*100, 3)
        except:
            return 1, 0
        return pvalue[1], score

    def stepwise_selection(self, ds=None, verbose=False, selection_type='stepwise', features=None, hold=None,
                           features_ini=None, limit_to_add=100, score_delta=0.01, scoring='gini', cv=None,
                           pvalue_threshold=0.05, pvalue_priority=False):
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
        :param features_ini: список переменных, с которых стартует отбор. Они могут быть исключены в процессе отбора
        :param limit_to_add: максимальное кол-во переменных, которые могут быть добавлены к модели
        :param score_delta: минимальный прирост метрики
        :param scoring: максимизируемая метрика.
                Варианты значений: 'gini', 'AIC', 'BIC', 'SIC', 'SBIC' + все метрики доступные для вычисления через sklearn.model_selection.cross_val_score.
                Все информационные метрики после вычисления умножаются на -1 для сохранения логики максимизации метрики.
        :param cv: параметр cv для вычисления скора sklearn.model_selection.cross_val_score
        :param pvalue_threshold: граница значимости по p-value
        :param pvalue_priority: вариант определения лучшего фактора

        :return: кортеж (итоговый список переменных, график со скором в процессе отбора в виде объекта plt.figure)
        """
        def add_feature(features, candidates, score, clf):
            cvs = {}
            pvalues = {}
            candidates = candidates - features
            if self.ds.n_jobs > 1:
                jobs = {}
                iterations = len(candidates)
                candidates_iter = iter(candidates)
                with futures.ProcessPoolExecutor(max_workers=self.ds.n_jobs) as pool:
                    while iterations:
                        for feature in candidates_iter:
                            jobs[pool.submit(self.add_feature_stat, df=self.ds.samples[self.ds.train_name][[self.ds.target, feature] + list(features)], clf=clf, scoring=scoring, cv=cv)] = feature
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
                    tmp_features = list(features) + [feature]
                    clf.fit(self.ds.samples[self.ds.train_name][tmp_features], self.ds.samples[self.ds.train_name][self.ds.target])
                    cvs[feature] = self.get_score(scoring=scoring, cv=cv, features=tmp_features, clf=clf, fit=False)
                    try:
                        wald = self.wald_test(features=tmp_features, clf=clf, fit=False)
                        pvalues[feature] = wald[wald.index == feature]['p-value'].values[0]
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
                    if features and cvs[features[0]] - score > score_delta:
                        feature = features[0]
                    else:
                        feature = None
                if feature:
                    return feature, cvs[feature], pvalues[feature]
            return None, None, None

        def drop_feature(features, hold, clf):
            # searching for the feature that increases the quality most
            wald = self.wald_test(clf=clf, features=list(features), fit=True)
            wald_to_check = wald[~wald.index.isin(list(hold) + ['intercept'])]
            pvalue = wald_to_check['p-value'].max()
            if wald_to_check.empty or pvalue < pvalue_threshold:
                return None, None, None
            feature = wald_to_check[wald_to_check['p-value'] == pvalue].index.values[0]
            score = self.get_score(scoring=scoring, cv=cv, features=list(features - {feature}), clf=clf, fit=True)
            return feature, score, pvalue

        def ini_score(features):
            if features:
                score = self.get_score(scoring=scoring, cv=cv, features=list(features), fit=True)
                graph = [(score, 'initial')]
                if verbose:
                    print(f'Initial features: {list(features)}, {scoring} score {score}')
            else:
                score = -1000000
                graph = []
            return score, graph

        if ds is not None:
            self.ds = ds

        hold = set() if hold is None else set(hold)
        features = set(self.ds.features) if features is None else set(features)
        features_ini = hold if features_ini is None else set(features_ini) | hold
        candidates = features - features_ini
        if selection_type != 'backward' or features_ini != hold:
            features = features_ini.copy()
        score, graph = ini_score(features)
        # correctness check
        for f in features | features_ini:
            if f not in self.ds.samples[self.ds.train_name].columns:
                print(f'No {f} in DataSample!')
                return None

        if verbose:
            print(f'{selection_type.capitalize()} feature selection started...')
        # Forward selection
        if selection_type == 'forward':
            # maximum number of steps equals to the number of candidates
            for i in range(len(candidates)):
                # cross-validation scores for each of the remaining candidates of the step
                feature, score, pvalue = add_feature(features, candidates, score, self.clf)
                if feature is None:
                    break
                if verbose:
                    print(f'To add: {feature}, {scoring}: {score}, p-value: {pvalue}')
                features.add(feature)
                graph.append((score, f'+ {feature}'))
                if len(features - features_ini) >= limit_to_add:
                    if verbose:
                        print(f'Reached the limit of the number of added features in the model. Selection stopped.')
                    break

        # Backward selection
        elif selection_type == 'backward':
            # maximum number of steps equals to the number of candidates
            for i in range(len(candidates)):
                feature, score, pvalue = drop_feature(features, hold, self.clf)
                if feature is None:
                    break
                if verbose:
                    print(f'To drop: {feature}, {scoring}: {score}, p-value: {pvalue}')
                features.remove(feature)
                graph.append((score, f'- {feature}'))
                if len(features - features_ini) >= limit_to_add:
                    if verbose:
                        print(f'Reached the limit of the number of added features in the model. Selection stopped.')
                    break

        # stepwise
        elif selection_type == 'stepwise':
            feature_sets = [features.copy()]
            for i in range(len(candidates)):
                feature, score, pvalue = add_feature(features, candidates, score, self.clf)
                if feature:
                    if verbose:
                        print(f'To add: {feature}, {scoring}: {score}, p-value: {pvalue}')
                    features.add(feature)
                    graph.append((score, f'+ {feature}'))
                elif verbose:
                    print('No significant features to add were found')
                if features:
                    feature, drop_score, pvalue = drop_feature(features, hold, self.clf)
                    if feature:
                        score = drop_score
                        if verbose:
                            print(f'To drop: {feature}, {scoring}: {score}, p-value: {pvalue}')
                        features.remove(feature)
                        graph.append((score, f'- {feature}'))
                if features in feature_sets:
                    break
                feature_sets.append(features.copy())
                if len(features - features_ini) >= limit_to_add:
                    if verbose:
                        print(f'Reached the limit of the number of added features in the model. Selection stopped.')
                    break
            features = feature_sets[-1]
        else:
            print('Incorrect kind of selection. Please use backward, forward or stepwise.')
            return None
        fig = plt.figure(figsize=(max(len(graph)//2, 5), 3))
        plt.plot(np.arange(len(graph)), [x[0] for x in graph], 'bo-', linewidth=2.0)
        plt.xticks(np.arange(len(graph)), [x[1] for x in graph], rotation=30, ha='right', fontsize=10)
        plt.tick_params(axis='y', which='both', labelsize=10)
        plt.ylabel(scoring)
        plt.title(f'{selection_type.capitalize()} score changes')
        fig.tight_layout()
        if verbose:
            plt.show()
        return list(features), fig

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
        if ds is not None:
            self.ds = ds
        if features:
            self.features = features
        elif not self.features:
            self.features = self.ds.features
        if sample_name is None:
            sample_name = self.ds.train_name
        try:
            self.clf.fit(self.ds.samples[sample_name][self.features], self.ds.samples[sample_name][self.ds.target])
            self.intercept = round(self.clf.intercept_[0], self.round_digits)
            self.coefs = {f: round(c, self.round_digits) for f, c in zip(self.features, self.clf.coef_[0])}
            print(f'intercept = {self.intercept}')
            print(f'coefs = {self.coefs}')
        except Exception as e:
            self.intercept = None
            self.coefs = {}
            print(e)
            print('Fit failed!')

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
        if sample_name is None:
            sample_name = ds.train_name
        if clf is None:
            clf = self.clf

        if scoring.upper() in ['AIC', 'BIC', 'SIC',  'SBIC']:
            if fit:
                clf.fit(ds.samples[sample_name][features], ds.samples[sample_name][ds.target])
            features_kept = [f for f, c in zip(features, clf.coef_[0]) if c != 0]
            weights_crit = [clf.intercept_[0]] + [c for c in clf.coef_[0] if c != 0]
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
                ds = DataSamples(samples={'train': data}, features=[], cat_columns=[])
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
                f = rem_suffix(f_WOE)
                if f in self.transformer.feature_woes:
                    result += self.transformer.feature_woes[f].get_transform_func(f"df['{f_WOE}'] = ") + "\n"
                elif is_cross(f):
                    f1, f2 = cross_split(f)
                    result += self.transformer.feature_crosses[f1].get_transform_func(f2, f"df['{f_WOE}'] = ") + "\n"
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
        result += f"return df\n\n"
        result += f'''df = scoring(df, score_field={f"'{score_field}'" if score_field else None}, pd_field={f"'{pd_field}'" if pd_field else None}, scale_field={f"'{scale_field}'" if scale_field else None})'''
        if file_name:
            file_name = add_ds_folder(self.ds, file_name)
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
